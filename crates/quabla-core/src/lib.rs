// Every unsafe block and impl must document its soundness invariant.
#![warn(clippy::undocumented_unsafe_blocks)]

pub mod autodiff;
pub mod compiler;
pub mod models;
pub mod ode;
pub mod optim;
pub mod tensor;
pub mod tensor_ir;

// The integration tests' hardware gates, so unit tests read them the same way.
#[cfg(test)]
#[path = "../tests/support/mod.rs"]
mod test_support;

pub use autodiff::{Dual, ForwardGradient};
pub use compiler::{
    QuablaCapability, QuablaCompiler, QuablaExecutable, QuablaJvpProgram,
    QuablaMultiOutputExecutable, QuablaMultiOutputProgram, QuablaPrecision, QuablaProgram,
    QuablaTarget, QuablaVjpProgram,
};
pub use quabla_macros::{forward_diff, forward_gradient};
