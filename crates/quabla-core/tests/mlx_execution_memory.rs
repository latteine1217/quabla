// MLX peak memory of a chain of loop regions. This file holds one test so that
// no other test shares the process-wide MLX memory counters.
#![cfg(all(feature = "mlx", target_os = "macos"))]

use std::collections::BTreeMap;

use quabla_core::tensor_ir::{
    DynamicTensor, MlxBackend, TensorBackend, TensorForiExecutionPlan, TensorIr,
};

/// Elements of every carry: 4 MiB of f32, far above MLX's small-array noise.
const N: usize = 1 << 20;
const CARRY_MIB: f64 = (N * 4) as f64 / (1 << 20) as f64;

/// `sum(fori(... fori(c) ...))` with `loops` chained two-step loops of
/// `carry * scale`, and the gradient of that sum with respect to `c`.
fn chain(loops: usize) -> Result<(TensorIr, usize, TensorIr, usize), String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![N])?;
    body.input("index", vec![])?;
    let scale = body.input("scale", vec![N])?;
    let next = body.mul(carry, scale)?;
    let plan = TensorForiExecutionPlan::new(0, 2, body.compile_region(next)?, "carry", "index")?;
    let mut graph = TensorIr::new();
    let mut c = graph.input("c", vec![N])?;
    let s = graph.input("s", vec![N])?;
    for _ in 0..loops {
        c = graph.fori(c, plan.clone(), vec![("scale".to_string(), s)])?;
    }
    let total = graph.sum(c)?;
    let vjp = graph.symbolic_vjp(total, "ct")?;
    let gradient = vjp.gradients["c"];
    Ok((graph, total, vjp.graph, gradient))
}

/// Peak MLX memory above the memory already in use while `run` executes.
fn peak_mib(run: impl FnOnce() -> Result<(), String>) -> Result<f64, String> {
    mlx_rs::memory::reset_peak_memory().map_err(|error| error.to_string())?;
    let before = mlx_rs::memory::active_memory().map_err(|error| error.to_string())?;
    run()?;
    let peak = mlx_rs::memory::peak_memory().map_err(|error| error.to_string())?;
    Ok(peak.saturating_sub(before) as f64 / (1 << 20) as f64)
}

/// Peak memory of the forward program and of its gradient for one chain length.
fn chain_peaks(loops: usize) -> Result<(f64, f64), String> {
    let (graph, total, reverse, gradient) = chain(loops)?;
    let forward = graph.compile_cpu(total)?;
    let reverse = reverse.compile_cpu(gradient)?;
    let inputs = BTreeMap::from([
        ("c".to_string(), DynamicTensor::filled(vec![N], 1.0)?),
        ("s".to_string(), DynamicTensor::filled(vec![N], 1.0)?),
        ("ct".to_string(), DynamicTensor::filled(vec![], 1.0)?),
    ]);
    // Warm up the kernels and the allocator cache before measuring.
    MlxBackend.execute(&forward, &inputs)?;
    MlxBackend.execute(&reverse, &inputs)?;
    let forward = peak_mib(|| MlxBackend.execute(&forward, &inputs).map(drop))?;
    let reverse = peak_mib(|| MlxBackend.execute(&reverse, &inputs).map(drop))?;
    Ok((forward, reverse))
}

/// A loop's carry is evaluated inside its region and read only by the next
/// loop, so once that loop has run nothing but the value table holds it. The
/// executor releases such values after their last use: the forward peak does
/// not grow with the length of the chain, and the gradient's peak grows by at
/// most the one primal carry per loop that its reverse pass reads back.
/// Keeping every value made them grow by one and by four carries per loop.
#[test]
fn mlx_loop_chain_peak_memory_grows_only_by_the_reverse_tape() -> Result<(), String> {
    const SHORT: usize = 4;
    const LONG: usize = 12;
    let (short_forward, short_reverse) = chain_peaks(SHORT)?;
    let (long_forward, long_reverse) = chain_peaks(LONG)?;
    println!("=== MLX peak above baseline, {CARRY_MIB} MiB carries ===");
    println!("forward: {SHORT} loops {short_forward:.1} MiB, {LONG} loops {long_forward:.1} MiB");
    println!("reverse: {SHORT} loops {short_reverse:.1} MiB, {LONG} loops {long_reverse:.1} MiB");
    let slack = 2.0 * CARRY_MIB;
    let tape = (LONG - SHORT) as f64 * CARRY_MIB;
    assert!(
        long_forward <= short_forward + slack,
        "forward peak grew from {short_forward} to {long_forward} MiB"
    );
    assert!(
        long_reverse <= short_reverse + tape + slack,
        "reverse peak grew from {short_reverse} to {long_reverse} MiB"
    );
    Ok(())
}
