// The caller holds MLX_EXECUTION_LOCK throughout graph construction and evaluation.
// MLX's own scatter_add combines duplicate indices with atomic adds, whose order
// is unspecified. This Metal kernel instead gives every output element one
// thread that adds its contributions sequentially in update order, so results
// are bitwise identical to the CPU reference's chain of rounded f32 adds.
use std::ffi::CString;

use mlx_rs::{Array, StreamOrDevice};
use mlx_sys::*;

fn check(status: i32, operation: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "MLX Metal scatter_add {operation} failed with status {status}"
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

const THREADGROUP: i32 = 256;

/// `base` (shape `shape`) with `updates` added at `indices` along `axis`;
/// see `TensorOp::ScatterAdd`. Contributions are grouped per destination
/// (CSR offsets and sources) on the host, keeping update order within a
/// destination, so each thread reads only its own contributions.
pub(super) fn evaluate(
    base: &Array,
    updates: &Array,
    indices: &[usize],
    axis: usize,
    shape: &[usize],
    stream: &StreamOrDevice,
) -> Result<Array, String> {
    let total = shape.iter().product::<usize>();
    if total == 0 {
        return Ok(base.clone());
    }
    let extent = shape[axis];
    let inner = shape[axis + 1..].iter().product::<usize>();
    let count = indices.len();
    let to_u32 = |value: usize| {
        u32::try_from(value).map_err(|_| "scatter_add index table exceeds u32".to_string())
    };
    let mut offsets = vec![0_u32; extent + 1];
    for index in indices {
        offsets[index + 1] += 1;
    }
    for destination in 0..extent {
        offsets[destination + 1] += offsets[destination];
    }
    let mut cursor = offsets[..extent].to_vec();
    let mut sources = vec![0_u32; count];
    for (source, index) in indices.iter().enumerate() {
        sources[cursor[*index] as usize] = to_u32(source)?;
        cursor[*index] += 1;
    }
    let offsets = Array::from_slice(
        &offsets,
        &[i32::try_from(extent + 1).map_err(|e| e.to_string())?],
    );
    let sources = Array::from_slice(
        &sources,
        &[i32::try_from(count).map_err(|e| e.to_string())?],
    );

    // `addf` keeps each sum a separate rounded f32 add.
    let header =
        "namespace quabla_scatter_helpers {\nfloat addf(float a,float b){return (a+b);}\n}\n";
    let body = format!(
        "const ulong e=thread_position_in_grid.x;\n\
         if(e>={total}UL)return;\n\
         const ulong outer=e/{span}UL;\n\
         const ulong row=(e/{inner}UL)%{extent}UL;\n\
         const ulong lane=e%{inner}UL;\n\
         float value=base[e];\n\
         for(uint k=offsets[row];k<offsets[row+1];++k){{value=quabla_scatter_helpers::addf(value,updates[(outer*{count}UL+sources[k])*{inner}UL+lane]);}}\n\
         out[e]=value;\n",
        span = extent * inner,
    );
    let body = CString::new(body).map_err(|e| e.to_string())?;
    let header = CString::new(header).map_err(|e| e.to_string())?;
    let input_names = [c"base", c"updates", c"offsets", c"sources"]
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();
    let output_names = [c"out".as_ptr()];
    let handles = [base, updates, &offsets, &sources]
        .iter()
        .map(|array| array.as_ptr())
        .collect::<Vec<_>>();
    let output_shape = shape
        .iter()
        .map(|extent| i32::try_from(*extent).map_err(|_| "scatter_add exceeds MLX dimensions"))
        .collect::<Result<Vec<_>, _>>()?;
    let threads = i32::try_from(total).map_err(|_| "scatter_add exceeds MLX dimensions")?;
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
            return Err("MLX Metal scatter_add name allocation failed".into());
        }
        let kernel = Kernel(mlx_fast_metal_kernel_new(
            c"quabla_scatter_add".as_ptr(),
            input_names.0,
            output_names.0,
            body.as_ptr(),
            header.as_ptr(),
            true,
            false,
        ));
        if kernel.0.ctx.is_null() {
            return Err("MLX Metal scatter_add kernel creation failed".into());
        }
        let config = Config(mlx_fast_metal_kernel_config_new());
        if config.0.ctx.is_null() {
            return Err("MLX Metal scatter_add configuration creation failed".into());
        }
        check(
            mlx_fast_metal_kernel_config_add_output_arg(
                config.0,
                output_shape.as_ptr(),
                output_shape.len(),
                mlx_rs::Dtype::Float32.into(),
            ),
            "output shape",
        )?;
        check(
            mlx_fast_metal_kernel_config_set_grid(config.0, threads, 1, 1),
            "grid",
        )?;
        check(
            mlx_fast_metal_kernel_config_set_thread_group(config.0, THREADGROUP.min(threads), 1, 1),
            "thread group",
        )?;
        let inputs = Arrays(mlx_vector_array_new_data(handles.as_ptr(), handles.len()));
        let mut outputs = Arrays(mlx_vector_array_new());
        if inputs.0.ctx.is_null() || outputs.0.ctx.is_null() {
            return Err("MLX Metal scatter_add array-vector allocation failed".into());
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
