use zarrquet::table_provider::ZarrTableProvider;
use datafusion::logical_expr::{Expr, col, lit, Operator, BinaryExpr};
use datafusion::scalar::ScalarValue;
use datafusion::datasource::TableProvider;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Predicate Pushdown ===\n");
    
    // Create test provider
    let provider = ZarrTableProvider::from_path("./test_data/multi_var.zarr".to_string())?;
    
    // Test coordinate filter parsing
    println!("🧪 Testing coordinate filter parsing...");
    
    // Create test filters
    let filter1 = Expr::BinaryExpr(BinaryExpr {
        left: Box::new(col("dim_0")),
        op: Operator::GtEq,
        right: Box::new(lit(ScalarValue::Int64(Some(1)))),
    });
    
    let filter2 = Expr::BinaryExpr(BinaryExpr {
        left: Box::new(col("dim_1")),
        op: Operator::Lt,
        right: Box::new(lit(ScalarValue::Int64(Some(2)))),
    });
    
    let filters = vec![&filter1, &filter2];
    
    // Test supports_filters_pushdown
    println!("📋 Testing supports_filters_pushdown...");
    let pushdown_result = provider.supports_filters_pushdown(&filters)?;
    
    println!("  Filter pushdown support:");
    for (i, result) in pushdown_result.iter().enumerate() {
        println!("    Filter {}: {:?}", i, result);
    }
    
    // Test filter parsing
    println!("\n🔍 Testing filter parsing...");
    let coordinate_filter = provider.parse_coordinate_filters(&filters)?;
    println!("  Parsed coordinate filter: {:?}", coordinate_filter);
    
    // Test filter matching
    println!("\n🎯 Testing coordinate matching...");
    let test_coordinates = vec![
        vec![0, 0, 0], // Should not match (dim_0 < 1)
        vec![1, 0, 0], // Should match
        vec![1, 1, 0], // Should match
        vec![1, 2, 0], // Should not match (dim_1 >= 2)
        vec![2, 0, 1], // Should match
    ];
    
    for coords in test_coordinates {
        let matches = coordinate_filter.matches(&coords);
        println!("  Coordinates {:?}: {}", coords, if matches { "✅ MATCH" } else { "❌ NO MATCH" });
    }
    
    // Test filtered batch creation
    println!("\n📊 Testing filtered batch creation...");
    match provider.create_filtered_batches(coordinate_filter, Some(10)) {
        Ok(batches) => {
            println!("  Successfully created {} filtered batches", batches.len());
            
            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            println!("  Total filtered rows: {}", total_rows);
            
            if !batches.is_empty() {
                println!("  Sample batch schema: {} columns", batches[0].num_columns());
                
                // Show first few rows
                let batch = &batches[0];
                if batch.num_rows() > 0 {
                    println!("  First few filtered rows:");
                    for row in 0..batch.num_rows().min(3) {
                        print!("    Row {}: [", row);
                        for col in 0..batch.num_columns() {
                            let array = batch.column(col);
                            let value = format_array_value(array, row);
                            print!("{}", value);
                            if col < batch.num_columns() - 1 {
                                print!(", ");
                            }
                        }
                        println!("]");
                    }
                }
            }
        },
        Err(e) => {
            println!("  ❌ Error creating filtered batches: {}", e);
        }
    }
    
    println!("\n✅ Predicate pushdown test complete!");
    Ok(())
}

fn format_array_value(array: &std::sync::Arc<dyn arrow_array::Array>, row: usize) -> String {
    use arrow_array::*;
    
    if array.is_null(row) {
        return "null".to_string();
    }
    
    if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        arr.value(row).to_string()
    } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        format!("{:.2}", arr.value(row))
    } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        arr.value(row).to_string()
    } else if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
        format!("{:.2}", arr.value(row))
    } else {
        "unknown".to_string()
    }
}