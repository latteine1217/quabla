//! Helpers shared by the integration tests, and by the library's unit tests
//! through a `#[path]` module in `src/lib.rs`.

// Each test crate uses a different subset.
#![allow(dead_code)]

/// An opt-in hardware test suite. A gate is on exactly when its environment
/// variable is the string "1"; any other value, "0" and the empty string
/// included, leaves it off. The Python suites apply the same rule
/// (`tests/python/_support.py`).
#[derive(Clone, Copy, Debug)]
pub enum Gate {
    /// MLX on Apple silicon (`--features mlx`).
    Mlx,
    /// One CUDA GPU (`--features cuda`).
    Cuda,
    /// Two CUDA GPUs and a loadable NCCL library (`--features cuda-nccl`).
    CudaNccl,
}

impl Gate {
    pub const fn variable(self) -> &'static str {
        match self {
            Gate::Mlx => "QUABLA_MLX_TEST",
            Gate::Cuda => "QUABLA_CUDA_TEST",
            Gate::CudaNccl => "QUABLA_CUDA_NCCL_TEST",
        }
    }

    pub fn enabled(self) -> bool {
        std::env::var_os(self.variable()).is_some_and(|value| value == "1")
    }
}
