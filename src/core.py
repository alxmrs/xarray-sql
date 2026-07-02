def dataset_to_table(ds, table_names):
    """
    Convierte un dataset de Xarray a una tabla SQL con todas las dimensiones como columnas.
    """
    # Obtener todas las dimensiones y variables
    columns = list(ds.dims.keys()) + list(ds.data_vars.keys())
    
    # Crear la tabla con todas las columnas
    # ...
    return table