use quabla_core::tensor_ir::{
    DynamicTensor, TensorCondExecutionPlan, TensorDType, TensorForiExecutionPlan, TensorIr,
    TensorScanExecutionPlan,
};
use std::collections::BTreeMap;

type ScanResult = (
    (DynamicTensor, DynamicTensor),
    (DynamicTensor, DynamicTensor),
);

// Reconstruct the previous per-output numeric execution rather than a symbolic derivative.
fn separate_jvps(
    body: &TensorIr,
    outputs: [usize; 2],
    steps: usize,
    mut carry: DynamicTensor,
    mut tangent: DynamicTensor,
    external: &BTreeMap<String, DynamicTensor>,
    directions: &BTreeMap<String, DynamicTensor>,
) -> Result<ScanResult, String> {
    let mut values = Vec::new();
    let mut derivatives = Vec::new();
    for index in 0..steps {
        let mut inputs = external.clone();
        inputs.insert("carry".into(), carry);
        inputs.insert("index".into(), DynamicTensor::filled(vec![], index as f64)?);
        let mut tangents = directions.clone();
        tangents.insert("carry".into(), tangent);
        tangents.insert("index".into(), DynamicTensor::filled(vec![], 0.0)?);
        (carry, tangent) = body.jvp(outputs[0], &inputs, &tangents)?;
        let (value, derivative) = body.jvp(outputs[1], &inputs, &tangents)?;
        values.push(value);
        derivatives.push(derivative);
    }
    fn stack(values: Vec<DynamicTensor>) -> Result<DynamicTensor, String> {
        let first = values.first().ok_or("scan requires at least one output")?;
        let shape = [vec![values.len()], first.shape().to_vec()].concat();
        // The existing stack helper concatenates through DynamicTensor::new (f64).
        let data = values.iter().flat_map(|v| v.storage().iter()).collect();
        DynamicTensor::new(shape, data)
    }
    Ok(((carry, stack(values)?), (tangent, stack(derivatives)?)))
}

fn assert_bits(actual: &DynamicTensor, expected: &DynamicTensor) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    assert_eq!(
        actual
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        expected
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}

fn check(
    body: TensorIr,
    outputs: [usize; 2],
    steps: usize,
    carry: DynamicTensor,
    tangent: DynamicTensor,
    external: BTreeMap<String, DynamicTensor>,
    directions: BTreeMap<String, DynamicTensor>,
) -> Result<ScanResult, String> {
    let scan = TensorScanExecutionPlan::new(
        0,
        steps,
        body.compile_cpu_many(&outputs)?.0,
        "carry",
        "index",
    )?;
    let expected = separate_jvps(
        &body,
        outputs,
        steps,
        carry.clone(),
        tangent.clone(),
        &external,
        &directions,
    );
    let actual = scan.jvp(carry, tangent, &external, &directions);
    match (&actual, &expected) {
        (Ok(((ac, av), (at, ad))), Ok(((ec, ev), (et, ed)))) => {
            for (a, e) in [(ac, ec), (av, ev), (at, et), (ad, ed)] {
                assert_bits(a, e);
            }
        }
        (Err(a), Err(e)) => assert_eq!(a, e),
        _ => panic!("different results: {actual:?} vs {expected:?}"),
    }
    actual
}

#[test]
fn shared_numeric_scan_jvp_matches_separate_outputs_and_analytic_capture_derivative(
) -> Result<(), String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![])?;
    let index = body.input("index", vec![])?;
    let scale = body.input("scale", vec![])?;
    let product = body.mul(carry, scale)?;
    let next = body.add(product, index)?;
    let output = body.mul(next, next)?;
    let result = check(
        body,
        [next, output],
        3,
        DynamicTensor::filled(vec![], 1.0)?,
        DynamicTensor::filled(vec![], 0.0)?,
        BTreeMap::from([("scale".into(), DynamicTensor::filled(vec![], 2.0)?)]),
        BTreeMap::from([("scale".into(), DynamicTensor::filled(vec![], 1.0)?)]),
    )?;
    assert_eq!(result.0 .0.data().as_ref(), &[12.0]);
    assert_eq!(result.1 .0.data().as_ref(), &[13.0]);
    assert_eq!(result.1 .1.data().as_ref(), &[4.0, 40.0, 312.0]);
    Ok(())
}

#[test]
fn numeric_scan_jvp_preserves_f32_rounding_repeated_outputs_and_passthrough() -> Result<(), String>
{
    for dtype in [TensorDType::F32, TensorDType::F64] {
        for repeated in [false, true] {
            let mut body = TensorIr::new();
            let carry = body.input_typed("carry", vec![3], dtype)?;
            let scale = body.input_typed("scale", vec![3], dtype)?;
            let shared = body.mul(carry, scale)?;
            let nonlinear = body.sin(shared)?;
            let next = body.div(nonlinear, scale)?;
            let output = if repeated { next } else { carry };
            check(
                body,
                [next, output],
                4,
                DynamicTensor::with_dtype(vec![3], vec![0.123456789, -0.0, 0.7], dtype)?,
                DynamicTensor::with_dtype(vec![3], vec![0.7654321, 0.3, -0.2], dtype)?,
                BTreeMap::from([(
                    "scale".into(),
                    DynamicTensor::with_dtype(vec![3], vec![0.3, 1.1, 1.7], dtype)?,
                )]),
                BTreeMap::from([(
                    "scale".into(),
                    DynamicTensor::with_dtype(vec![3], vec![0.2, -0.1, 0.05], dtype)?,
                )]),
            )?;
        }
    }
    Ok(())
}

#[test]
fn numeric_scan_jvp_preserves_nested_scan_results() -> Result<(), String> {
    let mut inner = TensorIr::new();
    let carry = inner.input("carry", vec![])?;
    let scale = inner.input("scale", vec![])?;
    let next = inner.mul(carry, scale)?;
    let inner_scan = TensorScanExecutionPlan::new(
        0,
        2,
        inner.compile_cpu_many(&[next, carry])?.0,
        "carry",
        "index",
    )?;
    let mut outer = TensorIr::new();
    let carry = outer.input("carry", vec![])?;
    let scale = outer.input("scale", vec![])?;
    let (next, values) = outer.scan(carry, inner_scan, vec![("scale".into(), scale)])?;
    let output = outer.sum(values)?;
    check(
        outer,
        [next, output],
        3,
        DynamicTensor::filled(vec![], 0.7)?,
        DynamicTensor::filled(vec![], 0.2)?,
        BTreeMap::from([("scale".into(), DynamicTensor::filled(vec![], 1.1)?)]),
        BTreeMap::from([("scale".into(), DynamicTensor::filled(vec![], 0.03)?)]),
    )?;
    Ok(())
}

#[test]
fn numeric_scan_jvp_preserves_nonfinite_values_and_validation_order() -> Result<(), String> {
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![3])?;
    let next = body.sqrt(carry)?;
    check(
        body,
        [next, carry],
        2,
        DynamicTensor::new(vec![3], vec![-1.0, f64::INFINITY, -0.0])?,
        DynamicTensor::filled(vec![3], 0.0)?,
        BTreeMap::new(),
        BTreeMap::new(),
    )?;
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![])?;
    let scale = body.input("scale", vec![])?;
    let next = body.mul(carry, scale)?;
    let scan = TensorScanExecutionPlan::new(
        0,
        1,
        body.compile_cpu_many(&[next, carry])?.0,
        "carry",
        "index",
    )?;
    let scalar = DynamicTensor::filled(vec![], 1.0)?;
    let external = BTreeMap::from([("scale".into(), scalar.clone())]);
    assert_eq!(
        scan.jvp(scalar.clone(), scalar.clone(), &external, &BTreeMap::new())
            .unwrap_err(),
        "scan external capture \"scale\" has an invalid shape"
    );
    let inputs = BTreeMap::from([
        ("carry".into(), scalar.clone()),
        ("scale".into(), scalar.clone()),
    ]);
    assert_eq!(
        body.jvp(usize::MAX, &BTreeMap::new(), &BTreeMap::new())
            .unwrap_err(),
        format!("node {} does not exist", usize::MAX)
    );
    assert_eq!(
        body.jvp(next, &inputs, &BTreeMap::new()).unwrap_err(),
        "missing input tangent \"carry\""
    );
    let wrong = BTreeMap::from([
        ("carry".into(), DynamicTensor::filled(vec![1], 0.0)?),
        ("scale".into(), scalar.clone()),
    ]);
    assert!(body
        .jvp(next, &inputs, &wrong)
        .unwrap_err()
        .contains("input tangent \"carry\" has shape"));
    let empty = TensorScanExecutionPlan::new(
        0,
        0,
        body.compile_cpu_many(&[next, carry])?.0,
        "carry",
        "index",
    )?;
    assert_eq!(
        empty
            .jvp(scalar.clone(), scalar, &external, &external)
            .unwrap_err(),
        "scan requires at least one output"
    );
    Ok(())
}

// Count requested heap bytes, not RSS: this exposes removed graph/tangent traversals directly.
struct CountingAllocator;
static COUNTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static ALLOCATIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// SAFETY: every operation delegates to System with the original pointer/layout; counters only
// observe allocation requests and never alter allocation ownership or lifetime.
unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if COUNTING.load(std::sync::atomic::Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: GlobalAlloc supplies a valid allocation layout.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: the pointer/layout come from the same System allocation delegate.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        if COUNTING.load(std::sync::atomic::Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            BYTES.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: GlobalAlloc supplies the live pointer, its layout, and valid new size.
        unsafe { std::alloc::System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
#[ignore = "CPU release probe; run alone with --ignored --nocapture --test-threads=1"]
fn scan_shared_numeric_jvp_cpu_release_probe() -> Result<(), String> {
    use std::sync::atomic::Ordering::Relaxed;
    fn measure(
        mut run: impl FnMut() -> Result<ScanResult, String>,
    ) -> Result<(f64, usize, usize), String> {
        ALLOCATIONS.store(0, Relaxed);
        BYTES.store(0, Relaxed);
        COUNTING.store(true, Relaxed);
        let start = std::time::Instant::now();
        let result = run();
        COUNTING.store(false, Relaxed);
        let elapsed = start.elapsed().as_secs_f64();
        std::hint::black_box(result?);
        Ok((elapsed, ALLOCATIONS.load(Relaxed), BYTES.load(Relaxed)))
    }
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![512])?;
    let scale = body.input("scale", vec![512])?;
    let mut next = carry;
    for _ in 0..24 {
        let product = body.mul(next, scale)?;
        next = body.sin(product)?;
    }
    let output = body.mul(next, next)?;
    let steps = 20;
    let scan = TensorScanExecutionPlan::new(
        0,
        steps,
        body.compile_cpu_many(&[next, output])?.0,
        "carry",
        "index",
    )?;
    let carry = DynamicTensor::filled(vec![512], 0.3)?;
    let tangent = DynamicTensor::filled(vec![512], 0.2)?;
    let external = BTreeMap::from([("scale".into(), DynamicTensor::filled(vec![512], 1.001)?)]);
    let directions = BTreeMap::from([("scale".into(), DynamicTensor::filled(vec![512], 0.01)?)]);
    let old = || {
        separate_jvps(
            &body,
            [next, output],
            steps,
            carry.clone(),
            tangent.clone(),
            &external,
            &directions,
        )
    };
    let new = || scan.jvp(carry.clone(), tangent.clone(), &external, &directions);
    let expected = old()?;
    let actual = new()?;
    for (a, e) in [
        (&actual.0 .0, &expected.0 .0),
        (&actual.0 .1, &expected.0 .1),
        (&actual.1 .0, &expected.1 .0),
        (&actual.1 .1, &expected.1 .1),
    ] {
        assert_bits(a, e);
    }
    println!("probe=scan_numeric_jvp lanes=512 depth=24 steps=20 rounds=12 parity=bitwise_passed");
    println!("allocation_metric=requested_heap_bytes reference_stack=direct_flatten_conservative");
    let mut old_times = Vec::new();
    let mut new_times = Vec::new();
    for round in 0..12 {
        let (old_result, new_result) = if round % 2 == 0 {
            (measure(old)?, measure(new)?)
        } else {
            let new_result = measure(new)?;
            (measure(old)?, new_result)
        };
        println!(
            "round={round} old_ms={:.6} new_ms={:.6} old_allocations={} new_allocations={} old_requested_bytes={} new_requested_bytes={}",
            old_result.0 * 1000.0,
            new_result.0 * 1000.0,
            old_result.1,
            new_result.1,
            old_result.2,
            new_result.2
        );
        assert!(new_result.1 < old_result.1);
        assert!(new_result.2 < old_result.2);
        old_times.push(old_result.0);
        new_times.push(new_result.0);
    }
    old_times.sort_by(f64::total_cmp);
    new_times.sort_by(f64::total_cmp);
    let old_median = (old_times[5] + old_times[6]) / 2.0;
    let new_median = (new_times[5] + new_times[6]) / 2.0;
    println!(
        "old_median_ms={:.6} new_median_ms={:.6} speedup={:.6}",
        old_median * 1000.0,
        new_median * 1000.0,
        old_median / new_median
    );
    Ok(())
}

#[test]
fn numeric_scan_jvp_preserves_nested_fori_and_cond_branches() -> Result<(), String> {
    let mut loop_body = TensorIr::new();
    let carry = loop_body.input("carry", vec![])?;
    let scale = loop_body.input("scale", vec![])?;
    let next = loop_body.mul(carry, scale)?;
    let fori = TensorForiExecutionPlan::new(0, 2, loop_body.compile_cpu(next)?, "carry", "index")?;
    let mut on_true = TensorIr::new();
    let carry = on_true.input("carry", vec![])?;
    let scale = on_true.input("scale", vec![])?;
    let next = on_true.fori(carry, fori, vec![("scale".into(), scale)])?;
    let mut on_false = TensorIr::new();
    let carry = on_false.input("carry", vec![])?;
    let scale = on_false.input("scale", vec![])?;
    let other = on_false.div(carry, scale)?;
    let branches =
        TensorCondExecutionPlan::new(on_true.compile_cpu(next)?, on_false.compile_cpu(other)?)?;
    let mut body = TensorIr::new();
    let carry = body.input("carry", vec![])?;
    let _scale = body.input("scale", vec![])?;
    let index = body.input("index", vec![])?;
    let half = body.scalar_constant(0.5);
    let predicate = body.greater(index, half)?;
    let next = body.cond(predicate, branches)?;
    let output = body.add(carry, next)?;
    check(
        body,
        [next, output],
        3,
        DynamicTensor::filled(vec![], 0.7)?,
        DynamicTensor::filled(vec![], 0.2)?,
        BTreeMap::from([("scale".into(), DynamicTensor::filled(vec![], 1.1)?)]),
        BTreeMap::from([("scale".into(), DynamicTensor::filled(vec![], 0.03)?)]),
    )?;
    Ok(())
}

#[test]
fn single_output_numeric_jvp_keeps_bool_tangent_and_primal_validation() -> Result<(), String> {
    let mut body = TensorIr::new();
    let flag = body.input_typed("flag", vec![], TensorDType::Bool)?;
    let inputs = BTreeMap::from([(
        "flag".into(),
        DynamicTensor::with_dtype(vec![], vec![1.0], TensorDType::Bool)?,
    )]);
    let (value, tangent) = body.jvp(flag, &inputs, &BTreeMap::new())?;
    assert_eq!(value.dtype(), TensorDType::Bool);
    assert_eq!(tangent.dtype(), TensorDType::F64);
    assert_eq!(tangent.data().as_ref(), &[0.0]);
    let zero = BTreeMap::from([("flag".into(), DynamicTensor::filled(vec![], 0.0)?)]);
    assert_bits(&body.jvp(flag, &inputs, &zero)?.1, &tangent);
    let nonzero = BTreeMap::from([("flag".into(), DynamicTensor::filled(vec![], 1.0)?)]);
    assert!(body
        .jvp(flag, &inputs, &nonzero)
        .unwrap_err()
        .contains("flag"));
    assert_eq!(
        body.jvp(flag, &BTreeMap::new(), &nonzero).unwrap_err(),
        "missing input \"flag\""
    );
    Ok(())
}
