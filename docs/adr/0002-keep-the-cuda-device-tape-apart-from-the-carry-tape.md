# Keep the CUDA device tape apart from the carry tape

Every Fori and Scan reverse pass on the CPU and MLX records and replays its
loop through `CarryTape` (`tensor_ir/carry_tape.rs`), while the CUDA host loop
keeps its own `DeviceTape` (`tensor_ir/cuda_host_loop.rs`). Both follow the
same `checkpoint_block_size` rule; only the tape that holds the states differs.

The device tape is separate because device states are owned buffers, not
shared references. Cloning a CPU `DynamicTensor` or an MLX `Array` only adds
a reference, so `CarryTape` clones each state it keeps and moves the state
into the step. Cloning a `CudaSlice` allocates and copies device memory, so
`DeviceTape` keeps the buffers it records and steps from a borrowed state.
Every buffer it drops (an intermediate state between checkpoints, and each
state once the reverse step has read it) goes back to the loop's buffer pool
through the host run, which the step function and the reverse step also need
mutably. The device tape also skips the loop's final forward step, which no
CUDA reverse pass reads. Serving all of this from `CarryTape` would add a
context argument to every callback, a borrowed step, a discard hook, and a
skip-the-final-step option to the interface the CPU and MLX callers use,
while the logic it would absorb is about forty lines of block arithmetic that
the shared rule already pins.

Revisit this if a second backend with owned, pooled device buffers appears,
or if the CPU and MLX passes move to borrowed steps for their own reasons.
