//! Error type for the crate.
//!
//! Mirrors the lightweight `thiserror`-free style used by `rvector`: we keep the
//! dependency surface minimal for a quantization primitive, so the enum
//! implements `std::error::Error` and `Display` by hand.

use std::fmt;

/// Errors returned by rotational quantization operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A code passed to a distance routine did not have the dimension this
    /// quantizer produces.
    DimensionMismatch {
        /// Dimension expected by the quantizer (the rotation's output dim).
        expected: usize,
        /// Dimension actually found in the supplied code.
        actual: usize,
    },
    /// The requested bit-width is not supported by this build.
    UnsupportedBits(u32),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DimensionMismatch { expected, actual } => {
                write!(f, "code dimensions don't match: expected {expected}, got {actual}")
            }
            Error::UnsupportedBits(bits) => {
                write!(f, "unsupported bit width: {bits} (only 8-bit is implemented)")
            }
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
