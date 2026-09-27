use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

use crate::tensor::PyTensor;

#[pyclass(name = "Adam", skip_from_py_object)]
pub struct PyAdam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    step: u64,
    moments: BTreeMap<String, (Vec<f64>, Vec<f64>)>,
}

#[pyfunction]
#[pyo3(signature = (gradient_sets, parameter_names = None))]
pub fn sum_gradients(
    gradient_sets: &Bound<'_, PyList>,
    parameter_names: Option<Vec<String>>,
) -> PyResult<BTreeMap<String, PyTensor>> {
    if gradient_sets.is_empty() {
        return Err(PyValueError::new_err(
            "sum_gradients requires at least one gradient dictionary",
        ));
    }

    let first_item = gradient_sets.get_item(0)?;
    let first = first_item.cast::<PyDict>()?;
    let strict_keys = parameter_names.is_none();
    let names = match parameter_names {
        Some(names) => names,
        None => {
            let mut names = Vec::with_capacity(first.len());
            for (key, _) in first.iter() {
                names.push(key.cast::<PyString>()?.to_str()?.to_string());
            }
            names
        }
    };
    let mut total = BTreeMap::new();
    for name in names {
        let tensor = first
            .get_item(&name)?
            .ok_or_else(|| PyValueError::new_err(format!("missing gradient for {name:?}")))?
            .extract::<PyRef<'_, PyTensor>>()?
            .clone();
        total.insert(name, tensor);
    }

    for index in 1..gradient_sets.len() {
        let gradient_item = gradient_sets.get_item(index)?;
        let gradients = gradient_item.cast::<PyDict>()?;
        if strict_keys && gradients.len() != total.len() {
            return Err(PyValueError::new_err(
                "gradient dictionaries must have identical keys",
            ));
        }
        for (name, accumulated) in &mut total {
            let gradient = gradients
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing gradient for {name:?}")))?
                .extract::<PyRef<'_, PyTensor>>()?;
            *accumulated = accumulated
                .try_add(&gradient)
                .map_err(PyValueError::new_err)?;
        }
    }
    Ok(total)
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (learning_rate = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8))]
    fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> PyResult<Self> {
        if !(learning_rate > 0.0
            && (0.0..1.0).contains(&beta1)
            && (0.0..1.0).contains(&beta2)
            && epsilon > 0.0)
        {
            return Err(PyValueError::new_err(
                "Adam hyperparameters must be positive and betas must be in [0, 1)",
            ));
        }
        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            step: 0,
            moments: BTreeMap::new(),
        })
    }

    fn step(
        &mut self,
        parameters: &Bound<'_, PyDict>,
        gradients: &Bound<'_, PyDict>,
    ) -> PyResult<BTreeMap<String, PyTensor>> {
        let mut validated = Vec::with_capacity(parameters.len());
        for (key, value) in parameters.iter() {
            let name = key.cast::<PyString>()?.to_str()?.to_string();
            let parameter = value.extract::<PyRef<'_, PyTensor>>()?.clone();
            let gradient = gradients
                .get_item(&name)?
                .ok_or_else(|| PyValueError::new_err(format!("missing gradient for {name:?}")))?
                .extract::<PyRef<'_, PyTensor>>()?
                .clone();
            let (parameter_shape, parameter_data) = parameter.shape_data();
            let (gradient_shape, gradient_data) = gradient.shape_data();
            if parameter_shape != gradient_shape || parameter_data.len() != gradient_data.len() {
                return Err(PyValueError::new_err(format!(
                    "gradient shape does not match parameter {name:?}"
                )));
            }
            if let Some((first, _)) = self.moments.get(&name) {
                if first.len() != parameter_data.len() {
                    return Err(PyValueError::new_err(format!(
                        "parameter shape changed for {name:?}"
                    )));
                }
            }
            validated.push((name, parameter, gradient));
        }

        self.step += 1;
        let correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let correction2 = 1.0 - self.beta2.powi(self.step as i32);
        let mut updated = BTreeMap::new();
        for (name, parameter, gradient) in validated {
            let (shape, parameter_data) = parameter.shape_data();
            let (_, gradient_data) = gradient.shape_data();
            let (first, second) = self.moments.entry(name.clone()).or_insert_with(|| {
                (
                    vec![0.0; parameter_data.len()],
                    vec![0.0; parameter_data.len()],
                )
            });
            let mut values = Vec::with_capacity(parameter_data.len());
            for ((value, gradient), (first, second)) in parameter_data
                .iter()
                .zip(gradient_data)
                .zip(first.iter_mut().zip(second.iter_mut()))
            {
                *first = self.beta1 * *first + (1.0 - self.beta1) * *gradient;
                *second = self.beta2 * *second + (1.0 - self.beta2) * gradient * gradient;
                values.push(
                    *value
                        - self.learning_rate * (*first / correction1)
                            / ((*second / correction2).sqrt() + self.epsilon),
                );
            }
            // 動量狀態維持 f64；更新後的參數保留原 dtype。
            updated.insert(
                name,
                PyTensor::from_shape_data_typed(shape.to_vec(), values, parameter.dtype())
                    .map_err(PyValueError::new_err)?,
            );
        }
        Ok(updated)
    }
}
