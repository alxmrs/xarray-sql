use zarrquet::table_provider::ZarrTableProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Error Handling ===\n");
    
    // Test inconsistent dimensions
    println!("🧪 Testing inconsistent dimensions...");
    match test_inconsistent_dimensions() {
        Ok(()) => println!("❌ Expected error but got success!"),
        Err(e) => {
            println!("✅ Correctly caught error: {}", e);
            if e.to_string().contains("Inconsistent dimensions") {
                println!("✅ Error message correctly identifies dimension mismatch");
            }
        }
    }
    
    // Test non-existent path
    println!("\n🧪 Testing non-existent path...");
    match test_nonexistent_path() {
        Ok(()) => println!("❌ Expected error but got success!"),
        Err(e) => {
            println!("✅ Correctly caught error: {}", e);
            if e.to_string().contains("does not exist") {
                println!("✅ Error message correctly identifies missing path");
            }
        }
    }
    
    // Test successful case for comparison
    println!("\n🧪 Testing successful case...");
    match test_successful_case() {
        Ok(()) => println!("✅ Successfully processed valid dataset"),
        Err(e) => println!("❌ Unexpected error: {}", e),
    }
    
    println!("\n🎉 Error handling tests complete!");
    Ok(())
}

fn test_inconsistent_dimensions() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ZarrTableProvider::from_path("./test_data/inconsistent.zarr".to_string())?;
    let _schema = provider.infer_schema()?;
    Ok(())
}

fn test_nonexistent_path() -> Result<(), Box<dyn std::error::Error>> {
    let _provider = ZarrTableProvider::from_path("./nonexistent/path.zarr".to_string())?;
    Ok(())
}

fn test_successful_case() -> Result<(), Box<dyn std::error::Error>> {
    let provider = ZarrTableProvider::from_path("./test_data/multi_var.zarr".to_string())?;
    let schema = provider.infer_schema()?;
    
    println!("   Schema fields: {}", schema.fields().len());
    
    // Try to read a chunk
    let chunk_indices = vec![0u64, 0u64, 0u64];
    let batch = provider.chunk_to_record_batch(&chunk_indices)?;
    
    println!("   Rows: {}, Columns: {}", batch.num_rows(), batch.num_columns());
    
    Ok(())
}