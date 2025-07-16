use zarrquet::table_provider::ZarrTableProvider;
use arrow_array::{Int64Array, Float64Array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Validating Multi-Variable Output ===\n");
    
    let provider = ZarrTableProvider::from_path("./test_data/multi_var.zarr".to_string())?;
    
    // Test schema
    println!("📋 Validating schema...");
    let schema = provider.infer_schema()?;
    
    assert_eq!(schema.fields().len(), 5, "Should have 5 fields");
    assert_eq!(schema.field(0).name(), "dim_0", "First field should be dim_0");
    assert_eq!(schema.field(1).name(), "dim_1", "Second field should be dim_1");
    assert_eq!(schema.field(2).name(), "dim_2", "Third field should be dim_2");
    
    // Data variables should be in the schema (order may vary)
    let field_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    assert!(field_names.contains(&"/pressure".to_string()), "Should contain pressure variable");
    assert!(field_names.contains(&"/temperature".to_string()), "Should contain temperature variable");
    
    println!("✅ Schema validation passed!");
    
    // Test data content
    println!("📊 Validating data content...");
    let chunk_indices = vec![0u64, 0u64, 0u64];
    let batch = provider.chunk_to_record_batch(&chunk_indices)?;
    
    assert_eq!(batch.num_rows(), 12, "Should have 12 rows (3×2×2)");
    assert_eq!(batch.num_columns(), 5, "Should have 5 columns");
    
    // Validate coordinate columns
    let dim_0 = batch.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let dim_1 = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    let dim_2 = batch.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
    
    // Check first few rows for correct coordinate progression
    assert_eq!(dim_0.value(0), 0, "Row 0: dim_0 should be 0");
    assert_eq!(dim_1.value(0), 0, "Row 0: dim_1 should be 0");
    assert_eq!(dim_2.value(0), 0, "Row 0: dim_2 should be 0");
    
    assert_eq!(dim_0.value(1), 0, "Row 1: dim_0 should be 0");
    assert_eq!(dim_1.value(1), 0, "Row 1: dim_1 should be 0");
    assert_eq!(dim_2.value(1), 1, "Row 1: dim_2 should be 1");
    
    assert_eq!(dim_0.value(4), 1, "Row 4: dim_0 should be 1");
    assert_eq!(dim_1.value(4), 0, "Row 4: dim_1 should be 0");
    assert_eq!(dim_2.value(4), 0, "Row 4: dim_2 should be 0");
    
    // Validate data values match expected pattern
    // Based on our Python script: temperature = 1.0, 2.0, 3.0, ..., 12.0
    // pressure = 100.0, 200.0, 300.0, ..., 1200.0
    
    // Find pressure and temperature columns (order may vary)
    let mut pressure_col_idx = None;
    let mut temp_col_idx = None;
    
    for (idx, field) in schema.fields().iter().enumerate() {
        if field.name() == "/pressure" {
            pressure_col_idx = Some(idx);
        } else if field.name() == "/temperature" {
            temp_col_idx = Some(idx);
        }
    }
    
    let pressure_col_idx = pressure_col_idx.expect("Should find pressure column");
    let temp_col_idx = temp_col_idx.expect("Should find temperature column");
    
    let pressure_array = batch.column(pressure_col_idx).as_any().downcast_ref::<Float64Array>().unwrap();
    let temp_array = batch.column(temp_col_idx).as_any().downcast_ref::<Float64Array>().unwrap();
    
    // Check specific values
    assert_eq!(temp_array.value(0), 1.0, "First temperature should be 1.0");
    assert_eq!(pressure_array.value(0), 100.0, "First pressure should be 100.0");
    
    assert_eq!(temp_array.value(1), 2.0, "Second temperature should be 2.0");
    assert_eq!(pressure_array.value(1), 200.0, "Second pressure should be 200.0");
    
    assert_eq!(temp_array.value(11), 12.0, "Last temperature should be 12.0");
    assert_eq!(pressure_array.value(11), 1200.0, "Last pressure should be 1200.0");
    
    println!("✅ Data validation passed!");
    
    // Print summary
    println!("\n🎯 Validation Summary:");
    println!("   ✅ Schema has correct structure (5 fields)");
    println!("   ✅ Coordinate columns progress correctly");
    println!("   ✅ Data variables contain expected values");
    println!("   ✅ Cartesian product structure is correct (12 rows)");
    println!("   ✅ Multi-variable implementation working perfectly!");
    
    println!("\n📊 Sample output verification:");
    for row in 0..4 {
        let t = dim_0.value(row);
        let lat = dim_1.value(row);
        let lon = dim_2.value(row);
        let temp = temp_array.value(row);
        let pressure = pressure_array.value(row);
        
        println!("   Row {}: [time={}, lat={}, lon={}, temp={:.1}, pressure={:.1}]", 
                 row, t, lat, lon, temp, pressure);
    }
    
    Ok(())
}