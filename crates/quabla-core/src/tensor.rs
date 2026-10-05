#[derive(Clone, Debug, PartialEq)]
pub struct Tensor2<const ROWS: usize, const COLS: usize> {
    data: Vec<f64>,
}

impl<const ROWS: usize, const COLS: usize> Tensor2<ROWS, COLS> {
    pub fn from_array(values: [[f64; COLS]; ROWS]) -> Self {
        let mut data = Vec::with_capacity(ROWS * COLS);

        for row in values {
            data.extend(row);
        }

        Self { data }
    }

    pub fn shape(&self) -> [usize; 2] {
        [ROWS, COLS]
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        assert!(row < ROWS, "row index out of bounds");
        assert!(col < COLS, "column index out of bounds");

        self.data[row * COLS + col]
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn to_array(&self) -> [[f64; COLS]; ROWS] {
        std::array::from_fn(|row| std::array::from_fn(|col| self.get(row, col)))
    }

    pub fn map<F>(&self, mut f: F) -> Self
    where
        F: FnMut(f64) -> f64,
    {
        Self {
            data: self.data.iter().map(|value| f(*value)).collect(),
        }
    }

    pub fn matmul<const OUT_COLS: usize>(
        &self,
        rhs: &Tensor2<COLS, OUT_COLS>,
    ) -> Tensor2<ROWS, OUT_COLS> {
        let mut data = vec![0.0; ROWS * OUT_COLS];
        if data.is_empty() {
            return Tensor2 { data };
        }

        for row in 0..ROWS {
            let output_row = &mut data[row * OUT_COLS..(row + 1) * OUT_COLS];
            for inner in 0..COLS {
                let lhs = self.data[row * COLS + inner];
                let rhs_row = &rhs.data[inner * OUT_COLS..(inner + 1) * OUT_COLS];
                for (output, &value) in output_row.iter_mut().zip(rhs_row) {
                    *output += lhs * value;
                }
            }
        }

        Tensor2 { data }
    }
}
