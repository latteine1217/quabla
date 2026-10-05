// The caller holds MLX_EXECUTION_LOCK throughout graph construction and evaluation.
// Custom kernels keep the recurrence and its bounded jet workspace on Metal.
// One threadgroup owns each matrix: the forward sweep runs column by column
// (diagonal, then the rows below it in parallel) and the reverse sweep runs
// columns in descending order, so every accumulator receives its updates in
// the same order as the sequential row-major recurrence and results stay
// bitwise identical to a single-thread sweep.
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
        let entry = "float reduced=input[row*n+col];for(ulong k=0;k<col;++k){float product=out[row*n+k]*out[col*n+k];reduced=reduced-product;}out[row*n+col]=row==col?metal::precise::sqrt(reduced):reduced/out[col*n+col];";
        (
            format!(
                "const ulong n={n};\nfor(ulong i=tid;i<n*n;i+=T)out[i]=0.0f;\n{BARRIER}\nfor(ulong col=0;col<n;++col){{{}}}",
                column_sweep(entry)
            ),
            String::new(),
        )
    };
    let mut prefix = format!(
        "const ulong batch=threadgroup_position_in_grid.x;\nconst ulong tid=thread_position_in_threadgroup.x,T=threads_per_threadgroup.x;\nconst ulong offset=batch*{n}*{n};\n"
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
    let threads = batches
        .checked_mul(THREADGROUP)
        .and_then(|threads| i32::try_from(threads).ok())
        .ok_or("Cholesky batches exceed MLX dimensions")?;
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
            mlx_fast_metal_kernel_config_set_grid(config.0, threads, 1, 1),
            "grid",
        )?;
        check(
            mlx_fast_metal_kernel_config_set_thread_group(config.0, THREADGROUP as i32, 1, 1),
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

// Threads cooperating on one matrix. MLX grids count threads (dispatchThreads),
// so the grid spans batches*THREADGROUP threads and each threadgroup is a batch.
const THREADGROUP: usize = 256;
// All shared state lives in device buffers, so phases synchronize device memory.
const BARRIER: &str = "threadgroup_barrier(metal::mem_flags::mem_device);";

// One forward column step: the diagonal entry first, then the entries below it
// in parallel. Every entry reads only columns finished in earlier steps plus
// this column's diagonal, so its arithmetic matches the sequential sweep.
fn column_sweep(entry: &str) -> String {
    format!(
        "if(tid==0){{ulong row=col;{entry}}}\n{BARRIER}\nfor(ulong row=col+1+tid;row<n;row+=T){{{entry}}}\n{BARRIER}\n"
    )
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
        // Descending column traversal consumes each cotangent before writing
        // its gradient; subsequent updates target earlier, unconsumed elements.
        source.push_str(if kind == CholeskyAdKind::Vjp {
            "device Jet* residuals=values+count;device Jet* cotangents=(device Jet*)out;\n"
        } else {
            "device Jet* residuals=values+count;device Jet* cotangents=residuals+count;\n"
        });
    }
    source.push_str("for(ulong i=tid;i<count;i+=T){values[i]=Jet{};out[i]=0.0f;");
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
    source.push_str(&format!("}}\n{BARRIER}\n"));
    let mut entry = String::from("ulong i=row*n+col;Jet reduced{};reduced.v=input[i];");
    if first {
        entry.push_str("reduced.d=other[i];");
    }
    if mixed {
        entry.push_str("reduced.e=third[i];reduced.h=fourth[i];");
    }
    entry.push_str(
        "for(ulong k=0;k<col;++k)reduced=sub(reduced,mul(values[row*n+k],values[col*n+k]));",
    );
    if reverse {
        entry.push_str("residuals[i]=reduced;");
    }
    entry.push_str("values[i]=row==col?root(reduced,0):divide(reduced,values[col*n+col]);");
    if !reverse {
        entry.push_str(if mixed {
            "out[i]=values[i].h;"
        } else {
            "out[i]=values[i].d;"
        });
    }
    source.push_str(&format!(
        "for(ulong col=0;col<n;++col){{{}}}\n",
        column_sweep(&entry)
    ));
    if reverse {
        let lane = if first { "d" } else { "v" };
        // Column j of the sequential sweep, regrouped by accumulator:
        // phase A finishes the entries below the diagonal and applies their
        // updates to their own rows; phase B applies their updates to row j
        // and the diagonal in descending row order; phase C finishes the
        // diagonal and applies its two updates per entry of row j. Each
        // accumulator therefore sees the same sequence of additions as in
        // the sequential descending row-major traversal.
        source.push_str(&format!("threadgroup Jet diagonal_reduced;\nfor(ulong c=n;c>0;--c){{ulong j=c-1,diagonal=j*n+j;\nfor(ulong r=j+1+tid;r<n;r+=T){{ulong i=r*n+j;Jet upstream=cotangents[i];Jet contribution=neg(divide(mul(upstream,residuals[i]),mul(values[diagonal],values[diagonal])));Jet reduced=divide(upstream,values[diagonal]);out[i]=reduced.{lane};for(ulong k=j;k>0;--k){{ulong inner=k-1,left=r*n+inner,right=j*n+inner;Jet product=neg(reduced);cotangents[left]=add(cotangents[left],mul(product,values[right]));}}cotangents[i]=reduced;residuals[i]=contribution;}}\n{BARRIER}\nfor(ulong t=tid;t<=j;t+=T){{if(t<j){{ulong right=j*n+t;Jet accumulated=cotangents[right];for(ulong r=n-1;r>j;--r){{Jet product=neg(cotangents[r*n+j]);accumulated=add(accumulated,mul(product,values[r*n+t]));}}cotangents[right]=accumulated;}}else{{Jet accumulated=cotangents[diagonal];for(ulong r=n-1;r>j;--r)accumulated=add(accumulated,residuals[r*n+j]);cotangents[diagonal]=accumulated;}}}}\n{BARRIER}\nif(tid==0){{Jet reduced=mul(cotangents[diagonal],root(residuals[diagonal],1));out[diagonal]=reduced.{lane};diagonal_reduced=reduced;}}\nthreadgroup_barrier(metal::mem_flags::mem_threadgroup);\nfor(ulong k=tid;k<j;k+=T){{ulong left=j*n+k;Jet product=neg(diagonal_reduced);cotangents[left]=add(cotangents[left],mul(product,values[left]));cotangents[left]=add(cotangents[left],mul(product,values[left]));}}\n{BARRIER}\n}}\n"));
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
