// Compare checkpoint replay with the original complete-tape fallback on real Metal arrays.
use super::*;
use crate::tensor_ir::{
    TensorForiExecutionPlan, TensorForiVjpJvpExecutionPlan, TensorIr, TensorScanExecutionPlan,
    TensorScanVjpJvpExecutionPlan,
};

fn with_full_tape<T>(run: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    crate::tensor_ir::LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.set(true));
    let result = run();
    crate::tensor_ir::LOOP_CHECKPOINT_TEST_FULL_TAPE.with(|value| value.set(false));
    result
}

#[allow(clippy::too_many_arguments)]
fn full_tape_mlx_fori_value_and_vjp(
    backend: &MlxBackend,
    loop_plan: &TensorForiExecutionPlan,
    initial_carry: Array,
    external_captures: &BTreeMap<String, Array>,
    output_cotangent: Array,
) -> Result<MlxForiVjpEvaluation, String> {
    with_full_tape(|| {
        mlx_fori_value_and_vjp(
            backend,
            loop_plan,
            initial_carry,
            external_captures,
            output_cotangent,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn full_tape_mlx_scan_value_and_vjp(
    backend: &MlxBackend,
    scan_plan: &TensorScanExecutionPlan,
    initial_carry: Array,
    external_captures: &BTreeMap<String, Array>,
    final_carry_cotangent: Array,
    output_cotangent: Array,
) -> Result<MlxScanVjpEvaluation, String> {
    with_full_tape(|| {
        mlx_scan_value_and_vjp(
            backend,
            scan_plan,
            initial_carry,
            external_captures,
            final_carry_cotangent,
            output_cotangent,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn full_tape_mlx_fori_vjp_jvp(
    backend: &MlxBackend,
    plan: &TensorForiVjpJvpExecutionPlan,
    initial_carry: Array,
    initial_tangent: Array,
    external_captures: &BTreeMap<String, Array>,
    external_tangents: &BTreeMap<String, Array>,
    output_cotangent: Array,
    output_cotangent_tangent: Array,
) -> Result<BTreeMap<String, Array>, String> {
    with_full_tape(|| {
        mlx_fori_vjp_jvp(
            backend,
            plan,
            initial_carry,
            initial_tangent,
            external_captures,
            external_tangents,
            output_cotangent,
            output_cotangent_tangent,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn full_tape_mlx_scan_vjp_jvp(
    backend: &MlxBackend,
    plan: &TensorScanVjpJvpExecutionPlan,
    initial_carry: Array,
    initial_tangent: Array,
    external_captures: &BTreeMap<String, Array>,
    external_tangents: &BTreeMap<String, Array>,
    final_carry_cotangent: Array,
    final_carry_cotangent_tangent: Array,
    output_cotangent: Array,
    output_cotangent_tangent: Array,
) -> Result<BTreeMap<String, Array>, String> {
    with_full_tape(|| {
        mlx_scan_vjp_jvp(
            backend,
            plan,
            initial_carry,
            initial_tangent,
            external_captures,
            external_tangents,
            final_carry_cotangent,
            final_carry_cotangent_tangent,
            output_cotangent,
            output_cotangent_tangent,
        )
    })
}

fn fixture(
    n: usize,
    steps: usize,
) -> Result<(TensorForiExecutionPlan, TensorScanExecutionPlan), String> {
    let mut graph = TensorIr::new();
    let c = graph.input_typed("c", vec![n], TensorDType::F32)?;
    let w = graph.input_typed("w", vec![n], TensorDType::F32)?;
    let next = graph.mul(c, w)?;
    let next = graph.tanh(next)?;
    let output = graph.sin(next)?;
    Ok((
        TensorForiExecutionPlan::new(3, 3 + steps, graph.compile_cpu(next)?, "c", "i")?,
        TensorScanExecutionPlan::new(
            3,
            3 + steps,
            graph.compile_cpu_many(&[next, output])?.0,
            "c",
            "i",
        )?,
    ))
}

fn array(shape: Vec<usize>, value: f64) -> Result<Array, String> {
    mlx_array_from_dynamic(&DynamicTensor::filled(shape, value)?)
}

fn bits(value: &Array) -> Result<Vec<u32>, String> {
    transforms::eval([value]).map_err(|error| error.to_string())?;
    Ok(value
        .as_slice::<f32>()
        .iter()
        .map(|value| value.to_bits())
        .collect())
}

#[test]
fn checkpoint_matches_original_metal_loops_and_higher_order_bits() -> Result<(), String> {
    let backend = MlxBackend;
    for steps in [1, 8, 17, 64] {
        let (fori, scan) = fixture(4, steps)?;
        let external = BTreeMap::from([("w".into(), array(vec![4], 0.9)?)]);
        let directions = BTreeMap::from([("w".into(), array(vec![4], 0.03)?)]);
        let initial = array(vec![4], 0.1)?;
        let tangent = array(vec![4], 0.02)?;
        let seed = array(vec![4], 1.0)?;
        let seed_tangent = array(vec![4], 0.01)?;
        let output_seed = array(vec![steps, 4], 0.2)?;
        let output_seed_tangent = array(vec![steps, 4], 0.04)?;
        let old = full_tape_mlx_fori_value_and_vjp(
            &backend,
            &fori,
            initial.clone(),
            &external,
            seed.clone(),
        )?;
        let new =
            mlx_fori_value_and_vjp(&backend, &fori, initial.clone(), &external, seed.clone())?;
        assert_eq!(bits(&new.carry_gradient)?, bits(&old.carry_gradient)?);
        assert_eq!(
            bits(&new.external_gradients["w"])?,
            bits(&old.external_gradients["w"])?
        );
        let old = full_tape_mlx_scan_value_and_vjp(
            &backend,
            &scan,
            initial.clone(),
            &external,
            seed.clone(),
            output_seed.clone(),
        )?;
        let new = mlx_scan_value_and_vjp(
            &backend,
            &scan,
            initial.clone(),
            &external,
            seed.clone(),
            output_seed.clone(),
        )?;
        assert_eq!(bits(&new.carry_gradient)?, bits(&old.carry_gradient)?);
        assert_eq!(
            bits(&new.external_gradients["w"])?,
            bits(&old.external_gradients["w"])?
        );
        let fori = TensorForiVjpJvpExecutionPlan::new(fori, "test")?;
        let old = full_tape_mlx_fori_vjp_jvp(
            &backend,
            &fori,
            initial.clone(),
            tangent.clone(),
            &external,
            &directions,
            seed.clone(),
            seed_tangent.clone(),
        )?;
        let new = mlx_fori_vjp_jvp(
            &backend,
            &fori,
            initial.clone(),
            tangent.clone(),
            &external,
            &directions,
            seed.clone(),
            seed_tangent.clone(),
        )?;
        for name in ["c", "w"] {
            assert_eq!(bits(&new[name])?, bits(&old[name])?);
        }
        let scan = TensorScanVjpJvpExecutionPlan::new(scan, "test")?;
        let old = full_tape_mlx_scan_vjp_jvp(
            &backend,
            &scan,
            initial.clone(),
            tangent.clone(),
            &external,
            &directions,
            seed.clone(),
            seed_tangent.clone(),
            output_seed.clone(),
            output_seed_tangent.clone(),
        )?;
        let new = mlx_scan_vjp_jvp(
            &backend,
            &scan,
            initial,
            tangent,
            &external,
            &directions,
            seed,
            seed_tangent,
            output_seed,
            output_seed_tangent,
        )?;
        for name in ["c", "w"] {
            assert_eq!(bits(&new[name])?, bits(&old[name])?);
        }
    }
    Ok(())
}

#[test]
#[ignore = "isolated complete Metal loop VJP timing and active-memory probe"]
fn profile_checkpoint_metal_vjp() -> Result<(), String> {
    let backend = MlxBackend;
    let cases = if let Ok(steps) = std::env::var("QUABLA_CHECKPOINT_PROFILE_STEPS") {
        vec![steps.parse::<usize>().map_err(|error| error.to_string())?]
    } else {
        vec![64, 128, 256, 512]
    };
    for steps in cases {
        let n = 8192;
        let (plan, _) = fixture(n, steps)?;
        let external = BTreeMap::from([("w".into(), array(vec![n], 0.9)?)]);
        let initial = array(vec![n], 0.1)?;
        let seed = array(vec![n], 1.0)?;
        transforms::eval([&initial, &seed, &external["w"]]).map_err(|e| e.to_string())?;
        let modes = if std::env::var_os("QUABLA_CHECKPOINT_FIRST").is_some() {
            [false, true]
        } else {
            [true, false]
        };
        for full_tape in modes {
            if let Ok(mode) = std::env::var("QUABLA_CHECKPOINT_PROFILE_MODE") {
                if (mode == "full_tape") != full_tape {
                    continue;
                }
            }
            let mut times = Vec::new();
            let mut peaks = Vec::new();
            for _ in 0..5 {
                mlx_rs::memory::reset_peak_memory().map_err(|e| e.to_string())?;
                let baseline = mlx_rs::memory::active_memory().map_err(|e| e.to_string())?;
                let start = std::time::Instant::now();
                let result = if full_tape {
                    full_tape_mlx_fori_value_and_vjp(
                        &backend,
                        &plan,
                        initial.clone(),
                        &external,
                        seed.clone(),
                    )?
                } else {
                    mlx_fori_value_and_vjp(
                        &backend,
                        &plan,
                        initial.clone(),
                        &external,
                        seed.clone(),
                    )?
                };
                transforms::eval(
                    std::iter::once(&result.carry_gradient)
                        .chain(result.external_gradients.values()),
                )
                .map_err(|e| e.to_string())?;
                // Host access waits for GPU completion before timing and allocator observations.
                let checksum = result
                    .carry_gradient
                    .as_slice::<f32>()
                    .iter()
                    .map(|v| f64::from(*v))
                    .sum::<f64>()
                    + result.external_gradients["w"]
                        .as_slice::<f32>()
                        .iter()
                        .map(|v| f64::from(*v))
                        .sum::<f64>();
                std::hint::black_box(checksum);
                times.push(start.elapsed().as_nanos());
                let peak = mlx_rs::memory::peak_memory().map_err(|e| e.to_string())?;
                let active = mlx_rs::memory::active_memory().map_err(|e| e.to_string())?;
                println!("{{\"sample\":true,\"steps\":{steps},\"full_tape\":{full_tape},\"baseline\":{baseline},\"peak\":{peak},\"active_after\":{active}}}");
                // Absolute active-memory peaks remain meaningful when prior command buffers
                // release asynchronously; subtracting a moving baseline does not.
                peaks.push(peak);
            }
            times.sort();
            peaks.sort();
            println!("{{\"steps\":{steps},\"full_tape\":{full_tape},\"median_ns\":{},\"peak_active_device_bytes\":{}}}", times[2], peaks[2]);
        }
    }
    Ok(())
}
