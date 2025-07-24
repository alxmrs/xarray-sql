use crate::table_provider::ZarrTableProvider;
use pyo3::prelude::*;

pub mod table_provider;

#[pymodule]
fn zarrquet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZarrTableProvider>()?;
    Ok(())
}
