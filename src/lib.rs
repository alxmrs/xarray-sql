use pyo3::prelude::*;
use crate::table_provider::ZarrTableProvider;

pub(crate) mod table_provider;


#[pymodule]
fn zarrquet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ZarrTableProvider>()?;
    Ok(())
}
