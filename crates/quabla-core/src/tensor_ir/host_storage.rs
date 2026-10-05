use std::borrow::Cow;
use std::sync::Arc;

use super::TensorDType;

/// Immutable, dtype-sized host storage. Clones and shape-only views share bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum HostTensorStorage {
    F64(Arc<Vec<f64>>),
    F32(Arc<Vec<f32>>),
    Bool(Arc<Vec<u8>>),
}

impl HostTensorStorage {
    pub fn from_f64(values: Vec<f64>, dtype: TensorDType) -> Self {
        match dtype {
            TensorDType::F64 => Self::F64(Arc::new(values)),
            TensorDType::F32 => Self::F32(Arc::new(values.into_iter().map(|v| v as f32).collect())),
            TensorDType::Bool => Self::Bool(Arc::new(
                values.into_iter().map(|v| u8::from(v != 0.0)).collect(),
            )),
        }
    }

    /// Adopts device F32 readback without a widened intermediate buffer.
    pub fn from_f32(mut values: Vec<f32>, dtype: TensorDType) -> Self {
        match dtype {
            TensorDType::F32 => {
                // Match the historical F32 -> F64 -> F32 conversion, including NaN quieting.
                for value in &mut values {
                    *value = f64::from(*value) as f32;
                }
                Self::F32(Arc::new(values))
            }
            TensorDType::F64 => Self::F64(Arc::new(values.into_iter().map(f64::from).collect())),
            TensorDType::Bool => Self::Bool(Arc::new(
                values.into_iter().map(|v| u8::from(v != 0.0)).collect(),
            )),
        }
    }

    /// Borrows existing F32 storage when widening would leave its bits unchanged.
    pub fn to_f32(&self) -> Cow<'_, [f32]> {
        if let Self::F32(values) = self {
            let signaling_nan = values.iter().any(|value| {
                let bits = value.to_bits();
                bits & 0x7f800000 == 0x7f800000 && bits & 0x007fffff != 0 && bits & 0x00400000 == 0
            });
            if !signaling_nan {
                return Cow::Borrowed(values);
            }
        }
        Cow::Owned(self.iter().map(|value| value as f32).collect())
    }

    pub fn dtype(&self) -> TensorDType {
        match self {
            Self::F64(_) => TensorDType::F64,
            Self::F32(_) => TensorDType::F32,
            Self::Bool(_) => TensorDType::Bool,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::Bool(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> f64 {
        match self {
            Self::F64(values) => values[index],
            Self::F32(values) => f64::from(values[index]),
            Self::Bool(values) => f64::from(values[index] != 0),
        }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = f64> + DoubleEndedIterator + '_ {
        (0..self.len()).map(|index| self.get(index))
    }

    pub fn first(&self) -> Option<f64> {
        (!self.is_empty()).then(|| self.get(0))
    }

    pub fn contains(&self, value: &f64) -> bool {
        self.iter().any(|element| element == *value)
    }

    pub fn to_f64(&self) -> Cow<'_, [f64]> {
        match self {
            Self::F64(values) => Cow::Borrowed(values),
            _ => Cow::Owned(self.iter().collect()),
        }
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.iter().collect()
    }

    pub fn into_f64(self) -> Vec<f64> {
        match self {
            Self::F64(values) => {
                Arc::try_unwrap(values).unwrap_or_else(|values| values.as_ref().clone())
            }
            _ => self.to_vec(),
        }
    }

    pub fn into_dtype(self, dtype: TensorDType) -> Self {
        if self.dtype() == dtype {
            self
        } else {
            // Convert each lane through F64 without retaining an array-sized widened copy.
            match dtype {
                TensorDType::F64 => Self::F64(Arc::new(self.iter().collect())),
                TensorDType::F32 => Self::F32(Arc::new(self.iter().map(|v| v as f32).collect())),
                TensorDType::Bool => {
                    Self::Bool(Arc::new(self.iter().map(|v| u8::from(v != 0.0)).collect()))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_payload_matches_dtype_without_a_widened_shadow() {
        let count = 4096;
        for (dtype, bytes_per_element) in [
            (TensorDType::F64, 8),
            (TensorDType::F32, 4),
            (TensorDType::Bool, 1),
        ] {
            let storage = HostTensorStorage::from_f64(vec![1.0; count], dtype);
            let bytes = match &storage {
                HostTensorStorage::F64(values) => values.capacity() * size_of::<f64>(),
                HostTensorStorage::F32(values) => values.capacity() * size_of::<f32>(),
                HostTensorStorage::Bool(values) => values.capacity() * size_of::<u8>(),
            };
            assert_eq!(bytes, count * bytes_per_element);
            assert_eq!(storage.iter().sum::<f64>(), count as f64);
        }
    }

    #[test]
    fn typed_storage_preserves_rounding_zero_sign_and_boolean_conversion() {
        let values = vec![-0.0, 0.1, f64::INFINITY, f64::NAN];
        let storage = HostTensorStorage::from_f64(values.clone(), TensorDType::F32);
        for (actual, expected) in storage.iter().zip(values) {
            assert_eq!(actual.to_bits(), f64::from(expected as f32).to_bits());
        }
        let mask = HostTensorStorage::from_f64(vec![-0.0, 0.0, -1.0, f64::NAN], TensorDType::Bool);
        assert_eq!(mask.to_vec(), vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn consuming_f64_storage_moves_unique_data_and_protects_shared_aliases() {
        let unique = HostTensorStorage::from_f64(vec![1.0, 2.0], TensorDType::F64);
        let HostTensorStorage::F64(values) = &unique else {
            unreachable!()
        };
        let address = values.as_ptr();
        let moved = unique.into_f64();
        assert_eq!(moved.as_ptr(), address);

        let shared = HostTensorStorage::from_f64(vec![1.0, 2.0], TensorDType::F64);
        let alias = shared.clone();
        let mut scratch = shared.into_f64();
        scratch[0] = 9.0;
        assert_eq!(alias.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn cross_dtype_conversion_matches_widened_reference_and_preserves_aliases() {
        for source_dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
            let source = HostTensorStorage::from_f64(
                vec![-0.0, 0.1, f64::INFINITY, f64::from_bits(0x7ff0000000000001)],
                source_dtype,
            );
            let original = source.iter().map(f64::to_bits).collect::<Vec<_>>();
            for target_dtype in [TensorDType::F64, TensorDType::F32, TensorDType::Bool] {
                let expected = HostTensorStorage::from_f64(source.to_vec(), target_dtype);
                let actual = source.clone().into_dtype(target_dtype);
                assert_eq!(actual.dtype(), target_dtype);
                assert_eq!(
                    actual.iter().map(f64::to_bits).collect::<Vec<_>>(),
                    expected.iter().map(f64::to_bits).collect::<Vec<_>>()
                );
                assert_eq!(
                    source.iter().map(f64::to_bits).collect::<Vec<_>>(),
                    original
                );
            }
        }
    }
}

#[cfg(test)]
mod device_conversion_tests {
    use super::*;
    #[test]
    fn typed_device_conversion_preserves_bits_and_avoids_widened_buffers() {
        let source = vec![
            -0.0f32,
            0.1,
            f32::INFINITY,
            f32::from_bits(0x7f800001),
            f32::from_bits(0xff800001),
        ];
        for dtype in [TensorDType::F32, TensorDType::F64, TensorDType::Bool] {
            let reference =
                HostTensorStorage::from_f64(source.iter().copied().map(f64::from).collect(), dtype);
            let actual = HostTensorStorage::from_f32(source.clone(), dtype);
            assert_eq!(
                actual.iter().map(f64::to_bits).collect::<Vec<_>>(),
                reference.iter().map(f64::to_bits).collect::<Vec<_>>()
            );
            assert_eq!(
                actual
                    .to_f32()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .iter()
                    .map(|v| (v as f32).to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let storage = HostTensorStorage::F32(Arc::new(vec![0.1, -0.0]));
        assert!(matches!(storage.to_f32(), Cow::Borrowed(_)));
        let signaling = HostTensorStorage::F32(Arc::new(source));
        assert!(matches!(signaling.to_f32(), Cow::Owned(_)));
    }
}
