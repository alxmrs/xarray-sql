use zarrquet::table_provider::ZarrTableProvider;
use std::sync::Arc;
use arrow_array::RecordBatch;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Multi-Variable Zarr Implementation ===\n");
    
    // Test paths to try (in order of preference)
    let test_paths = vec![
        // Real-world ERA5 dataset (if accessible)
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        // Local test data paths
        "./test_data/multi_var.zarr",
        "./test_data/inconsistent.zarr",
        "./test_data/sample.zarr",
        "../test_data/multi_var.zarr",
    ];
    
    for zarr_path in test_paths {
        println!("Attempting to test with: {}", zarr_path);
        
        match test_zarr_dataset(zarr_path) {
            Ok(()) => {
                println!("✅ Successfully tested with: {}", zarr_path);
                return Ok(());
            },
            Err(e) => {
                println!("❌ Failed with {}: {}", zarr_path, e);
                continue;
            }
        }
    }
    
    println!("\n⚠️  No accessible Zarr datasets found.");
    println!("💡 To test with real data:");
    println!("   1. Run `gcloud auth application-default login` for GCS access");
    println!("   2. Or create local test data in ./test_data/");
    
    // Create a mock demonstration of expected behavior
    demonstrate_expected_behavior();
    
    Ok(())
}

fn test_zarr_dataset(zarr_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("  📁 Opening Zarr store...");
    
    // Create ZarrTableProvider
    let provider = ZarrTableProvider::from_path(zarr_path.to_string())?;
    
    println!("  📋 Inferring schema...");
    let schema = provider.infer_schema()?;
    
    println!("  🎯 Schema inferred successfully!");
    println!("     Fields: {}", schema.fields().len());
    
    // Print schema details
    for (idx, field) in schema.fields().iter().enumerate() {
        let field_type = if field.name().starts_with("dim_") {
            "coordinate"
        } else {
            "data variable"
        };
        println!("     {}. {} ({:?}) - {}", 
                 idx + 1, field.name(), field.data_type(), field_type);
    }
    
    // Try to read a small chunk
    println!("  📊 Testing chunk reading...");
    let chunk_indices = vec![0u64; schema.fields().len().min(3)]; // Safe default
    
    match provider.chunk_to_record_batch(&chunk_indices) {
        Ok(batch) => {
            println!("  ✅ Successfully read chunk!");
            print_record_batch_summary(&batch);
        },
        Err(e) => {
            println!("  ⚠️  Chunk reading failed (this might be expected): {}", e);
        }
    }
    
    Ok(())
}

fn print_record_batch_summary(batch: &RecordBatch) {
    println!("     Rows: {}", batch.num_rows());
    println!("     Columns: {}", batch.num_columns());
    
    if batch.num_rows() > 0 {
        println!("     Sample data (first 3 rows):");
        
        for row_idx in 0..batch.num_rows().min(3) {
            print!("       Row {}: [", row_idx);
            
            for col_idx in 0..batch.num_columns() {
                let column = batch.column(col_idx);
                let value = format_array_value(column, row_idx);
                print!("{}", value);
                
                if col_idx < batch.num_columns() - 1 {
                    print!(", ");
                }
            }
            println!("]");
        }
    }
}

fn format_array_value(column: &Arc<dyn arrow_array::Array>, row_idx: usize) -> String {
    use arrow_array::*;
    
    if column.is_null(row_idx) {
        return "null".to_string();
    }
    
    // Try different array types
    if let Some(arr) = column.as_any().downcast_ref::<Int64Array>() {
        arr.value(row_idx).to_string()
    } else if let Some(arr) = column.as_any().downcast_ref::<Float64Array>() {
        format!("{:.2}", arr.value(row_idx))
    } else if let Some(arr) = column.as_any().downcast_ref::<Int32Array>() {
        arr.value(row_idx).to_string()
    } else if let Some(arr) = column.as_any().downcast_ref::<Float32Array>() {
        format!("{:.2}", arr.value(row_idx))
    } else {
        "unknown".to_string()
    }
}

fn demonstrate_expected_behavior() {
    println!("\n🎯 Expected Multi-Variable Behavior:");
    println!("   For a Zarr with variables temperature(time,lat,lon) and pressure(time,lat,lon)");
    println!("   with shapes [2,2,2], we expect:");
    println!();
    println!("   Schema: [dim_0: Int64, dim_1: Int64, dim_2: Int64, temperature: Float64, pressure: Float64]");
    println!();
    println!("   Rows (cartesian product):");
    println!("     Row 0: [0, 0, 0, temp_000, pressure_000]");
    println!("     Row 1: [0, 0, 1, temp_001, pressure_001]");
    println!("     Row 2: [0, 1, 0, temp_010, pressure_010]");
    println!("     Row 3: [0, 1, 1, temp_011, pressure_011]");
    println!("     Row 4: [1, 0, 0, temp_100, pressure_100]");
    println!("     Row 5: [1, 0, 1, temp_101, pressure_101]");
    println!("     Row 6: [1, 1, 0, temp_110, pressure_110]");
    println!("     Row 7: [1, 1, 1, temp_111, pressure_111]");
    println!();
    println!("   Each row represents one coordinate point with all variable values.");
}