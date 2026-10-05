// The caller holds MLX_EXECUTION_LOCK throughout graph construction and evaluation.
// Custom kernels keep the recurrence and its bounded jet workspace on Metal.
use std::ffi::CString;

use mlx_rs::{Array, Dtype, StreamOrDevice};
use mlx_sys::*;

use super::super::CholeskyAdKind;

fn check(status: i32, operation: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "MLX Metal Cholesky {operation} failed with status {status}"
        ))
    }
}

// Each wrapper owns exactly one C handle. C constructors copy borrowed names
// and arrays; output extraction creates a separate owned mlx_array handle.
macro_rules! owned_handle {
    ($name:ident, $raw:ty, $free:ident) => {
        struct $name($raw);
        impl Drop for $name {
            fn drop(&mut self) {
                // SAFETY: this handle is owned and is freed exactly once.
                unsafe {
                    $free(self.0);
                }
            }
        }
    };
}
owned_handle!(Names, mlx_vector_string, mlx_vector_string_free);
owned_handle!(Arrays, mlx_vector_array, mlx_vector_array_free);
owned_handle!(Kernel, mlx_fast_metal_kernel, mlx_fast_metal_kernel_free);
owned_handle!(
    Config,
    mlx_fast_metal_kernel_config,
    mlx_fast_metal_kernel_config_free
);

pub(super) fn evaluate(
    inputs: &[&Array],
    shape: &[usize],
    kind: Option<CholeskyAdKind>,
    stream: &StreamOrDevice,
) -> Result<Array, String> {
    let n = shape[shape.len() - 1];
    let batches = shape[..shape.len() - 2].iter().product::<usize>();
    let (mut body, header) = if let Some(kind) = kind {
        ad_source(n, kind)
    } else {
        (format!("const ulong n={n};\nfor(ulong i=0;i<n*n;++i)out[i]=0.0f;\nfor(ulong row=0;row<n;++row){{for(ulong col=0;col<=row;++col){{float reduced=input[row*n+col];for(ulong k=0;k<col;++k){{float product=out[row*n+k]*out[col*n+k];reduced=reduced-product;}}out[row*n+col]=row==col?metal::precise::sqrt(reduced):reduced/out[col*n+col];}}}}"), String::new())
    };
    let mut prefix = format!(
        "const ulong batch=thread_position_in_grid.x;\nconst ulong offset=batch*{n}*{n};\n"
    );
    for name in ["input", "other", "third", "fourth"]
        .iter()
        .take(inputs.len())
    {
        prefix.push_str(&format!("{name} += offset;\n"));
    }
    prefix.push_str("out += offset;\n");
    if let Some(kind) = kind {
        prefix.push_str(&format!("scratch += offset*{};\n", scratch_lanes(kind)));
    }
    body.insert_str(0, &prefix);
    let body = CString::new(body).map_err(|e| e.to_string())?;
    let header = CString::new(header).map_err(|e| e.to_string())?;
    let names = [c"input", c"other", c"third", c"fourth"];
    let input_names = names[..inputs.len()]
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();
    let output_names = if kind.is_some() {
        vec![c"out".as_ptr(), c"scratch".as_ptr()]
    } else {
        vec![c"out".as_ptr()]
    };
    let handles = inputs
        .iter()
        .map(|array| array.as_ptr())
        .collect::<Vec<_>>();
    let shape = shape
        .iter()
        .map(|extent| i32::try_from(*extent).map_err(|_| "Cholesky matrix exceeds MLX dimensions"))
        .collect::<Result<Vec<_>, _>>()?;
    let batches = i32::try_from(batches).map_err(|_| "Cholesky batches exceed MLX dimensions")?;
    // SAFETY: all pointers reference live, correctly sized buffers. MLX copies
    // the names/input handles. RAII frees handles on every error path, and the
    // extracted output is separately owned before its vector is dropped.
    unsafe {
        let input_names = Names(mlx_vector_string_new_data(
            input_names.as_ptr().cast_mut(),
            input_names.len(),
        ));
        let output_names = Names(mlx_vector_string_new_data(
            output_names.as_ptr().cast_mut(),
            output_names.len(),
        ));
        if input_names.0.ctx.is_null() || output_names.0.ctx.is_null() {
            return Err("MLX Metal Cholesky name allocation failed".into());
        }
        let kernel = Kernel(mlx_fast_metal_kernel_new(
            c"quabla_cholesky".as_ptr(),
            input_names.0,
            output_names.0,
            body.as_ptr(),
            header.as_ptr(),
            true,
            false,
        ));
        if kernel.0.ctx.is_null() {
            return Err("MLX Metal Cholesky kernel creation failed".into());
        }
        let config = Config(mlx_fast_metal_kernel_config_new());
        if config.0.ctx.is_null() {
            return Err("MLX Metal Cholesky configuration creation failed".into());
        }
        check(
            mlx_fast_metal_kernel_config_add_output_arg(
                config.0,
                shape.as_ptr(),
                shape.len(),
                Dtype::Float32.into(),
            ),
            "output shape",
        )?;
        if let Some(kind) = kind {
            let lanes = i32::try_from(scratch_lanes(kind))
                .map_err(|_| "Cholesky workspace exceeds MLX dimensions")?;
            let mut scratch_shape = shape.clone();
            scratch_shape.insert(0, lanes);
            check(
                mlx_fast_metal_kernel_config_add_output_arg(
                    config.0,
                    scratch_shape.as_ptr(),
                    scratch_shape.len(),
                    Dtype::Float32.into(),
                ),
                "workspace shape",
            )?;
        }
        check(
            mlx_fast_metal_kernel_config_set_grid(config.0, batches, 1, 1),
            "grid",
        )?;
        check(
            mlx_fast_metal_kernel_config_set_thread_group(config.0, 1, 1, 1),
            "thread group",
        )?;
        let inputs = Arrays(mlx_vector_array_new_data(handles.as_ptr(), handles.len()));
        let mut outputs = Arrays(mlx_vector_array_new());
        if inputs.0.ctx.is_null() || outputs.0.ctx.is_null() {
            return Err("MLX Metal Cholesky array-vector allocation failed".into());
        }
        check(
            mlx_fast_metal_kernel_apply(
                &mut outputs.0,
                kernel.0,
                inputs.0,
                config.0,
                stream.as_ref().as_ptr(),
            ),
            "apply",
        )?;
        let mut output = mlx_array_new();
        let status = mlx_vector_array_get(&mut output, outputs.0, 0);
        if status != 0 {
            mlx_array_free(output);
            check(status, "output extraction")?;
        }
        Ok(Array::from_ptr(output))
    }
}

pub(super) fn scratch_lanes(kind: CholeskyAdKind) -> usize {
    let lanes = match kind {
        CholeskyAdKind::Vjp => 1,
        CholeskyAdKind::Mixed => 4,
        _ => 2,
    };
    lanes
        * match kind {
            CholeskyAdKind::Vjp => 2,
            CholeskyAdKind::VjpJvp => 3,
            _ => 1,
        }
}

fn ad_source(n: usize, kind: CholeskyAdKind) -> (String, String) {
    let function = "quabla_cholesky";
    let first = kind != CholeskyAdKind::Vjp;
    let mixed = kind == CholeskyAdKind::Mixed;
    let reverse = matches!(kind, CholeskyAdKind::Vjp | CholeskyAdKind::VjpJvp);
    let prefix = format!("{function}_helpers");
    let fields = if mixed {
        "float v,d,e,h;"
    } else if first {
        "float v,d;"
    } else {
        "float v;"
    };
    let mut source = format!("namespace {prefix} {{\nstruct Jet {{ {fields} }};\n");
    source.push_str("float addf(float a,float b){return (a+b);}\nfloat mulf(float a,float b){return (a*b);}\nfloat subf(float a,float b){return (a-b);}\nfloat divf(float a,float b){return (a/b);}\n");
    source.push_str("Jet add(Jet a,Jet b){Jet c;c.v=addf(a.v,b.v);");
    if first {
        source.push_str("c.d=addf(a.d,b.d);");
    }
    if mixed {
        source.push_str("c.e=addf(a.e,b.e);c.h=addf(a.h,b.h);");
    }
    source.push_str("return c;}\nJet neg(Jet a){a.v=-a.v;");
    if first {
        source.push_str("a.d=-a.d;");
    }
    if mixed {
        source.push_str("a.e=-a.e;a.h=-a.h;");
    }
    source.push_str("return a;}\nJet sub(Jet a,Jet b){return add(a,neg(b));}\nJet mul(Jet a,Jet b){Jet c;c.v=mulf(a.v,b.v);");
    if first {
        source.push_str("c.d=addf(mulf(a.d,b.v),mulf(a.v,b.d));");
    }
    if mixed {
        source.push_str("c.e=addf(mulf(a.e,b.v),mulf(a.v,b.e));c.h=addf(addf(addf(mulf(a.h,b.v),mulf(a.d,b.e)),mulf(a.e,b.d)),mulf(a.v,b.h));");
    }
    source.push_str("return c;}\nJet divide(Jet a,Jet b){Jet c;c.v=divf(a.v,b.v);float denominator=mulf(b.v,b.v);\n");
    if first {
        source.push_str(
            "float numerator=subf(mulf(a.d,b.v),mulf(a.v,b.d));c.d=divf(numerator,denominator);",
        );
    }
    if mixed {
        source.push_str("c.e=divf(subf(mulf(a.e,b.v),mulf(a.v,b.e)),denominator);float numerator_direction=subf(addf(mulf(a.h,b.v),mulf(a.d,b.e)),addf(mulf(a.e,b.d),mulf(a.v,b.h)));float denominator_direction=addf(mulf(b.e,b.v),mulf(b.v,b.e));c.h=divf(subf(mulf(numerator_direction,denominator),mulf(numerator,denominator_direction)),mulf(denominator,denominator));");
    }
    source.push_str("return c;}\nfloat derivative(float x,int order){if(x==0.0f)return 0.0f;if(x<0.0f)return NAN;float coefficient=1.0f;for(int i=0;i<order;++i)coefficient=mulf(coefficient,0.5f-(float)i);return mulf(coefficient,metal::precise::pow(x,0.5f-(float)order));}\nJet root(Jet a,int order){Jet c;c.v=order==0?metal::precise::sqrt(a.v):derivative(a.v,order);");
    if first {
        source.push_str("c.d=mulf(a.d,derivative(a.v,order+1));");
    }
    if mixed {
        source.push_str("c.e=mulf(a.e,derivative(a.v,order+1));c.h=addf(mulf(a.h,derivative(a.v,order+1)),mulf(mulf(a.d,a.e),derivative(a.v,order+2)));");
    }
    source.push_str("return c;}\n}\n");
    let header = std::mem::take(&mut source);
    source.push_str(&format!("using namespace {prefix};\nconst ulong n={n},count=n*n;\ndevice Jet* values=(device Jet*)scratch;\n"));
    if reverse {
        // Descending traversal consumes each cotangent before writing its
        // gradient; subsequent updates target earlier, unconsumed elements.
        source.push_str(if kind == CholeskyAdKind::Vjp {
            "device Jet* residuals=values+count;device Jet* cotangents=(device Jet*)out;\n"
        } else {
            "device Jet* residuals=values+count;device Jet* cotangents=residuals+count;\n"
        });
    }
    source.push_str("for(ulong i=0;i<count;++i){values[i]=Jet{};out[i]=0.0f;");
    if reverse {
        let upstream = if kind == CholeskyAdKind::Vjp {
            "i/n>=i%n?other[i]:0.0f"
        } else {
            "third[i]"
        };
        source.push_str(&format!(
            "cotangents[i]=Jet{{}};cotangents[i].v={upstream};"
        ));
        if first {
            source.push_str("cotangents[i].d=fourth[i];");
        }
    }
    source.push_str("}\nfor(ulong row=0;row<n;++row){for(ulong col=0;col<=row;++col){ulong i=row*n+col;Jet reduced{};reduced.v=input[i];");
    if first {
        source.push_str("reduced.d=other[i];");
    }
    if mixed {
        source.push_str("reduced.e=third[i];reduced.h=fourth[i];");
    }
    source.push_str(
        "for(ulong k=0;k<col;++k)reduced=sub(reduced,mul(values[row*n+k],values[col*n+k]));",
    );
    if reverse {
        source.push_str("residuals[i]=reduced;");
    }
    source.push_str("values[i]=row==col?root(reduced,0):divide(reduced,values[col*n+col]);");
    if !reverse {
        source.push_str(if mixed {
            "out[i]=values[i].h;"
        } else {
            "out[i]=values[i].d;"
        });
    }
    source.push_str("}}\n");
    if reverse {
        source.push_str("for(ulong r=n;r>0;--r){ulong row=r-1;for(ulong c=row+1;c>0;--c){ulong col=c-1,i=row*n+col;Jet upstream=cotangents[i],reduced;if(row==col){reduced=mul(upstream,root(residuals[i],1));}else{ulong diagonal=col*n+col;Jet contribution=neg(divide(mul(upstream,residuals[i]),mul(values[diagonal],values[diagonal])));cotangents[diagonal]=add(cotangents[diagonal],contribution);reduced=divide(upstream,values[diagonal]);}");
        source.push_str(if first {
            "out[i]=reduced.d;"
        } else {
            "out[i]=reduced.v;"
        });
        source.push_str("for(ulong k=col;k>0;--k){ulong inner=k-1,left=row*n+inner,right=col*n+inner;Jet product=neg(reduced);cotangents[left]=add(cotangents[left],mul(product,values[right]));cotangents[right]=add(cotangents[right],mul(product,values[left]));}}}\n");
    }
    (source, header)
}

#[cfg(test)]
mod workspace_tests {
    use super::*;

    #[test]
    fn vjp_consumes_output_cotangents_without_a_third_workspace_matrix() {
        let (body, _) = ad_source(8, CholeskyAdKind::Vjp);
        assert_eq!(scratch_lanes(CholeskyAdKind::Vjp), 2);
        assert!(body.contains("cotangents=(device Jet*)out"));
        assert!(!body.contains("cotangents=residuals+count"));
        assert!(body.contains("i/n>=i%n?other[i]:0.0f"));
        let (body, _) = ad_source(8, CholeskyAdKind::VjpJvp);
        assert_eq!(scratch_lanes(CholeskyAdKind::VjpJvp), 6);
        assert!(body.contains("cotangents=residuals+count"));
    }
}
