# Quabla

Quabla is a JAX-like array runtime written in Rust for scientific machine
learning: Python functions are traced into a tensor program, transformed
(differentiated, batched), and executed on the CPU, CUDA, or MLX.

## Programs

**Tensor program**:
A directed acyclic graph of ops over rank-N arrays, built by tracing a
function; its Rust type is `TensorIr`.
_Avoid_: graph (alone), IR (alone), expression

**Op**:
One node kind of a tensor program, such as `Add`, `Matmul`, or `Gather`; a
`TensorOp` variant.
_Avoid_: primitive, kernel, instruction

**Op family**:
A group of ops that share every rule except their value and derivative and
are therefore one op with a kind, such as the unary math functions.
_Avoid_: op category, op class

**Kind**:
The member of an op family that a node computes, such as `Exp` of the unary
math family or `Pow` of the binary math family.
_Avoid_: subtype, flavour

**Region**:
An op that owns nested tensor programs and runs them under control flow:
a conditional, a loop, a scan, or one of their derivatives.
_Avoid_: block, subgraph, control-flow node

**Carry tape**:
The loop states a region's reverse pass reads back, from the last step to the
first: the state before every step, or one checkpoint per block of steps that
is replayed forward when its block is reached; its Rust type is `CarryTape`.
_Avoid_: checkpoint list, tape (alone)

**Region kind**:
Which control flow a region performs, including the derivative it was
built for, such as `Fori`, `ForiVjp`, or `ScanVjpJvp`.

## Transforms

**Trace**:
The act of running a Python function on tracers to record a tensor program.
_Avoid_: capture, record

**Tracer**:
A stand-in value that records each op applied to it into the tensor program
being traced; its Python type is `TraceTensor`.
_Avoid_: symbolic tensor, placeholder

**Eager**:
Executing an op immediately on concrete values instead of recording it.

**Symbolic derivative**:
A derivative (JVP or VJP) produced as another tensor program, so it can be
differentiated again and run on every backend.
_Avoid_: runtime AD, tape AD

## Execution

**Backend**:
A target that executes tensor programs: CPU, CUDA, or MLX.
_Avoid_: device (a backend may drive several devices), engine

**Execution plan**:
A tensor program prepared for one backend, ready to run on inputs.
_Avoid_: executable (the Python handle), compiled program

**Fusion admission**:
Whether a backend's fused path accepts an op into one generated kernel. It
differs between backends, because admitting an op changes which neighbouring
ops share a kernel and so how their floating-point results round.

**CPU reference**:
The float64 CPU result that every other path of an op is compared against.
_Avoid_: ground truth, oracle

**Golden baseline**:
A recorded set of program outputs that a change must reproduce bit for bit
on each backend.
_Avoid_: snapshot (alone), fixture
