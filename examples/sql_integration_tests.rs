use zarrquet::table_provider::ZarrTableProvider;
use datafusion::prelude::*;
use datafusion::error::DataFusionError;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting SQL Integration Tests for Zarr Datasets\n");
    
    // Create DataFusion context
    let ctx = SessionContext::new();
    
    // Register all test datasets as tables
    println!("📋 Registering Zarr datasets as SQL tables...");
    register_zarr_tables(&ctx).await?;
    
    // Run comprehensive test suite
    println!("\n🧪 Running SQL Integration Tests...\n");
    
    // Basic query tests
    test_basic_queries(&ctx).await?;
    
    // Filtering and predicate pushdown tests
    test_filtering_queries(&ctx).await?;
    
    // Aggregation tests
    test_aggregation_queries(&ctx).await?;
    
    // Projection tests
    test_projection_queries(&ctx).await?;
    
    // Join tests - single dataset
    test_self_joins(&ctx).await?;
    
    // Join tests - multiple datasets
    test_multi_dataset_joins(&ctx).await?;
    
    // Complex query tests
    test_complex_queries(&ctx).await?;
    
    println!("🎉 All SQL integration tests completed successfully!");
    Ok(())
}

async fn register_zarr_tables(ctx: &SessionContext) -> Result<(), DataFusionError> {
    let datasets = vec![
        ("weather", "./test_data/weather.zarr"),
        ("ocean", "./test_data/ocean.zarr"),
        ("timeseries", "./test_data/timeseries.zarr"),
        ("single_dim", "./test_data/single_dim.zarr"),
        ("business", "./test_data/business.zarr"),
        ("multi_var", "./test_data/multi_var.zarr"), // From previous tests
    ];
    
    for (table_name, zarr_path) in datasets {
        match ZarrTableProvider::from_path(zarr_path.to_string()) {
            Ok(provider) => {
                ctx.register_table(table_name, Arc::new(provider))?;
                println!("  ✅ Registered table '{}' from {}", table_name, zarr_path);
            }
            Err(e) => {
                println!("  ⚠️  Skipped table '{}': {}", table_name, e);
            }
        }
    }
    Ok(())
}

async fn test_basic_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Testing Basic SQL Queries");
    
    // Test 1: Simple SELECT *
    println!("  Test 1: SELECT * FROM multi_var LIMIT 5");
    let df = ctx.sql("SELECT * FROM multi_var LIMIT 5").await?;
    let results = df.collect().await?;
    println!("    ✅ Returned {} rows, {} columns", 
             results.iter().map(|b| b.num_rows()).sum::<usize>(),
             if !results.is_empty() { results[0].num_columns() } else { 0 });
    
    // Test 2: COUNT queries
    println!("  Test 2: SELECT COUNT(*) FROM multi_var");
    let df = ctx.sql("SELECT COUNT(*) as total_rows FROM multi_var").await?;
    let results = df.collect().await?;
    print_results("COUNT", &results);
    
    // Test 3: Simple column selection
    println!("  Test 3: SELECT dim_0, dim_1 FROM multi_var LIMIT 3");
    let df = ctx.sql("SELECT dim_0, dim_1 FROM multi_var LIMIT 3").await?;
    let results = df.collect().await?;
    print_results("Column Selection", &results);
    
    Ok(())
}

async fn test_filtering_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Filtering Queries (Predicate Pushdown)");
    
    // Test 1: Simple WHERE clause
    println!("  Test 1: WHERE dim_0 >= 1");
    let df = ctx.sql("SELECT COUNT(*) as filtered_count FROM multi_var WHERE dim_0 >= 1").await?;
    let results = df.collect().await?;
    print_results("Simple Filter", &results);
    
    // Test 2: Multiple conditions
    println!("  Test 2: WHERE dim_0 >= 1 AND dim_1 < 2");
    let df = ctx.sql("SELECT COUNT(*) as count FROM multi_var WHERE dim_0 >= 1 AND dim_1 < 2").await?;
    let results = df.collect().await?;
    print_results("Multiple Conditions", &results);
    
    // Test 3: Range queries
    println!("  Test 3: WHERE dim_0 BETWEEN 0 AND 1");
    let df = ctx.sql("SELECT dim_0, COUNT(*) as count FROM multi_var WHERE dim_0 BETWEEN 0 AND 1 GROUP BY dim_0").await?;
    let results = df.collect().await?;
    print_results("Range Query", &results);
    
    // Test 4: Complex boolean logic
    println!("  Test 4: WHERE (dim_0 = 0 AND dim_1 = 0) OR (dim_0 = 2 AND dim_1 = 1)");
    let df = ctx.sql("SELECT COUNT(*) as count FROM multi_var WHERE (dim_0 = 0 AND dim_1 = 0) OR (dim_0 = 2 AND dim_1 = 1)").await?;
    let results = df.collect().await?;
    print_results("Complex Boolean", &results);
    
    Ok(())
}

async fn test_aggregation_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Aggregation Queries");
    
    // Test 1: Basic aggregations
    println!("  Test 1: Basic aggregations (MIN, MAX, AVG)");
    let df = ctx.sql("
        SELECT 
            COUNT(*) as count,
            MIN(dim_0) as min_dim0,
            MAX(dim_0) as max_dim0,
            AVG(CAST(dim_0 AS DOUBLE)) as avg_dim0
        FROM multi_var
    ").await?;
    let results = df.collect().await?;
    print_results("Basic Aggregations", &results);
    
    // Test 2: GROUP BY
    println!("  Test 2: GROUP BY dim_0");
    let df = ctx.sql("
        SELECT 
            dim_0,
            COUNT(*) as count,
            MIN(dim_1) as min_dim1,
            MAX(dim_1) as max_dim1
        FROM multi_var 
        GROUP BY dim_0 
        ORDER BY dim_0
    ").await?;
    let results = df.collect().await?;
    print_results("GROUP BY", &results);
    
    // Test 3: Multiple GROUP BY
    println!("  Test 3: GROUP BY dim_0, dim_1");
    let df = ctx.sql("
        SELECT 
            dim_0, 
            dim_1, 
            COUNT(*) as count 
        FROM multi_var 
        GROUP BY dim_0, dim_1 
        ORDER BY dim_0, dim_1
    ").await?;
    let results = df.collect().await?;
    print_results("Multiple GROUP BY", &results);
    
    // Test 4: HAVING clause
    println!("  Test 4: GROUP BY with HAVING");
    let df = ctx.sql("
        SELECT 
            dim_0, 
            COUNT(*) as count 
        FROM multi_var 
        GROUP BY dim_0 
        HAVING COUNT(*) > 2
        ORDER BY dim_0
    ").await?;
    let results = df.collect().await?;
    print_results("HAVING clause", &results);
    
    Ok(())
}

async fn test_projection_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Projection Queries");
    
    // Test 1: Column aliases
    println!("  Test 1: Column aliases");
    let df = ctx.sql("
        SELECT 
            dim_0 as time_idx,
            dim_1 as lat_idx,
            dim_2 as lon_idx
        FROM multi_var 
        LIMIT 3
    ").await?;
    let results = df.collect().await?;
    print_results("Column Aliases", &results);
    
    // Test 2: Computed columns
    println!("  Test 2: Computed columns");
    let df = ctx.sql("
        SELECT 
            dim_0,
            dim_1,
            (dim_0 * 10 + dim_1) as computed_id
        FROM multi_var 
        LIMIT 5
    ").await?;
    let results = df.collect().await?;
    print_results("Computed Columns", &results);
    
    // Test 3: DISTINCT
    println!("  Test 3: DISTINCT values");
    let df = ctx.sql("SELECT DISTINCT dim_0 FROM multi_var ORDER BY dim_0").await?;
    let results = df.collect().await?;
    print_results("DISTINCT", &results);
    
    Ok(())
}

async fn test_self_joins(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Self-Joins");
    
    // Test 1: Self join on coordinates
    println!("  Test 1: Self join on shared coordinates");
    let df = ctx.sql("
        SELECT 
            a.dim_0,
            a.dim_1,
            COUNT(*) as pair_count
        FROM multi_var a
        JOIN multi_var b ON a.dim_0 = b.dim_0 AND a.dim_1 = b.dim_1
        GROUP BY a.dim_0, a.dim_1
        LIMIT 5
    ").await?;
    let results = df.collect().await?;
    print_results("Self Join", &results);
    
    Ok(())
}

async fn test_multi_dataset_joins(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Multi-Dataset Joins");
    
    // Test 1: Cross-dataset join (if tables exist)
    println!("  Test 1: Attempting cross-dataset joins...");
    
    // Check which tables are available
    let available_tables = ctx.catalog("datafusion").unwrap()
        .schema("public").unwrap()
        .table_names();
    
    println!("    Available tables: {:?}", available_tables);
    
    // Try weather + ocean join (both have lat coordinates)
    if available_tables.contains(&"weather".to_string()) && available_tables.contains(&"ocean".to_string()) {
        println!("  Test 1a: Weather + Ocean join on lat");
        
        // First, let's see what the schemas look like
        let weather_schema = ctx.sql("SELECT * FROM weather LIMIT 1").await;
        let ocean_schema = ctx.sql("SELECT * FROM ocean LIMIT 1").await;
        
        if weather_schema.is_ok() && ocean_schema.is_ok() {
            let df = ctx.sql("
                SELECT 
                    w.dim_1 as weather_lat_idx,
                    o.dim_1 as ocean_lat_idx,
                    COUNT(*) as match_count
                FROM weather w
                JOIN ocean o ON w.dim_1 = o.dim_1
                GROUP BY w.dim_1, o.dim_1
                LIMIT 5
            ").await?;
            let results = df.collect().await?;
            print_results("Weather-Ocean Join", &results);
        }
    }
    
    // Test 2: Try different dimensional joins
    if available_tables.contains(&"timeseries".to_string()) && available_tables.contains(&"multi_var".to_string()) {
        println!("  Test 1b: Timeseries + Multi_var join");
        let df = ctx.sql("
            SELECT 
                t.dim_0 as time_idx,
                m.dim_0 as multi_time_idx,
                COUNT(*) as matches
            FROM timeseries t
            JOIN multi_var m ON t.dim_0 = m.dim_0
            GROUP BY t.dim_0, m.dim_0
            LIMIT 5
        ").await?;
        let results = df.collect().await?;
        print_results("Timeseries-MultiVar Join", &results);
    }
    
    Ok(())
}

async fn test_complex_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Complex SQL Queries");
    
    // Test 1: Subquery
    println!("  Test 1: Subquery");
    let df = ctx.sql("
        SELECT 
            dim_0,
            coord_count
        FROM (
            SELECT 
                dim_0,
                COUNT(*) as coord_count
            FROM multi_var
            GROUP BY dim_0
        ) subq
        WHERE coord_count > 3
        ORDER BY dim_0
    ").await?;
    let results = df.collect().await?;
    print_results("Subquery", &results);
    
    // Test 2: CTE (Common Table Expression)
    println!("  Test 2: Common Table Expression (CTE)");
    let df = ctx.sql("
        WITH coord_stats AS (
            SELECT 
                dim_0,
                dim_1,
                COUNT(*) as point_count
            FROM multi_var
            GROUP BY dim_0, dim_1
        )
        SELECT 
            dim_0,
            SUM(point_count) as total_points,
            AVG(CAST(point_count AS DOUBLE)) as avg_points
        FROM coord_stats
        GROUP BY dim_0
        ORDER BY dim_0
    ").await?;
    let results = df.collect().await?;
    print_results("CTE", &results);
    
    // Test 3: Window functions (if supported)
    println!("  Test 3: Window functions");
    let df = ctx.sql("
        SELECT 
            dim_0,
            dim_1,
            ROW_NUMBER() OVER (PARTITION BY dim_0 ORDER BY dim_1) as row_num
        FROM multi_var
        ORDER BY dim_0, dim_1
        LIMIT 10
    ").await?;
    let results = df.collect().await?;
    print_results("Window Functions", &results);
    
    Ok(())
}

fn print_results(test_name: &str, results: &[arrow_array::RecordBatch]) {
    println!("    📊 {} Results:", test_name);
    
    if results.is_empty() {
        println!("      (No results)");
        return;
    }
    
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    println!("      Rows: {}, Columns: {}", total_rows, results[0].num_columns());
    
    // Print first few rows
    let sample_rows = 3.min(total_rows);
    if sample_rows > 0 {
        println!("      Sample data:");
        
        // Print headers
        let schema = results[0].schema();
        print!("        ");
        for field in schema.fields() {
            print!("{:>12} ", field.name());
        }
        println!();
        
        // Print data rows
        let mut printed_rows = 0;
        for batch in results {
            for row in 0..batch.num_rows() {
                if printed_rows >= sample_rows {
                    break;
                }
                
                print!("        ");
                for col in 0..batch.num_columns() {
                    let value = format_array_value(batch.column(col), row);
                    print!("{:>12} ", value);
                }
                println!();
                printed_rows += 1;
            }
            if printed_rows >= sample_rows {
                break;
            }
        }
    }
    println!();
}

fn format_array_value(array: &std::sync::Arc<dyn arrow_array::Array>, row: usize) -> String {
    use arrow_array::*;
    
    if array.is_null(row) {
        return "NULL".to_string();
    }
    
    if let Some(arr) = array.as_any().downcast_ref::<Int64Array>() {
        arr.value(row).to_string()
    } else if let Some(arr) = array.as_any().downcast_ref::<Float64Array>() {
        format!("{:.2}", arr.value(row))
    } else if let Some(arr) = array.as_any().downcast_ref::<Int32Array>() {
        arr.value(row).to_string()
    } else if let Some(arr) = array.as_any().downcast_ref::<Float32Array>() {
        format!("{:.2}", arr.value(row))
    } else if let Some(arr) = array.as_any().downcast_ref::<arrow_array::StringArray>() {
        arr.value(row).to_string()
    } else {
        "?".to_string()
    }
}