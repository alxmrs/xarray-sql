use zarrquet::table_provider::ZarrTableProvider;
use datafusion::prelude::*;
use datafusion::error::DataFusionError;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Testing Dimensional Joins Between Zarr Datasets\n");
    
    // Create DataFusion context
    let ctx = SessionContext::new();
    
    // Register all test datasets as tables
    println!("📋 Registering Zarr datasets...");
    register_zarr_tables(&ctx).await?;
    
    // Test different dimensional joins
    test_schema_exploration(&ctx).await?;
    test_multid_to_multid_joins(&ctx).await?;
    test_2d_to_4d_joins(&ctx).await?;
    test_cross_dimensional_aggregations(&ctx).await?;
    test_complex_dimensional_queries(&ctx).await?;
    
    println!("🎉 All dimensional join tests completed!");
    Ok(())
}

async fn register_zarr_tables(ctx: &SessionContext) -> Result<(), DataFusionError> {
    let datasets = vec![
        ("weather", "./test_data/weather.zarr"),        // 4D: time×lat×lon×altitude
        ("ocean", "./test_data/ocean.zarr"),            // 3D: depth×lat×lon
        ("timeseries", "./test_data/timeseries.zarr"),  // 2D: time×station
        ("business", "./test_data/business.zarr"),      // 3D: category×region×period
        ("multi_var", "./test_data/multi_var.zarr"),    // 3D: time×lat×lon (small)
        // Skip single_dim for now due to compatibility issues
    ];
    
    for (table_name, zarr_path) in datasets {
        match ZarrTableProvider::from_path(zarr_path.to_string()) {
            Ok(provider) => {
                ctx.register_table(table_name, Arc::new(provider))?;
                println!("  ✅ {}", table_name);
            }
            Err(e) => {
                println!("  ⚠️  Skipped {}: {}", table_name, e);
            }
        }
    }
    Ok(())
}

async fn test_schema_exploration(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Exploring Dataset Schemas and Dimensions");
    
    let tables = vec!["weather", "ocean", "timeseries", "business", "multi_var"];
    
    for table in tables {
        println!("\n  📊 Table: {}", table);
        
        // Get schema info
        let df_result = ctx.sql(&format!("SELECT * FROM {} LIMIT 1", table)).await;
        let results = match df_result {
            Ok(df) => df.collect().await?,
            Err(e) => {
                println!("    ❌ Error querying table: {}", e);
                continue;
            }
        };
        
        if !results.is_empty() {
            let schema = results[0].schema();
            println!("    Columns: {}", schema.fields().len());
            
            // Print column info
            for field in schema.fields() {
                let col_type = if field.name().starts_with("dim_") { "coordinate" } else { "data" };
                println!("      {} - {:?} ({})", field.name(), field.data_type(), col_type);
            }
            
            // Count total rows
            match ctx.sql(&format!("SELECT COUNT(*) as total FROM {}", table)).await {
                Ok(count_df) => {
                    let count_results = count_df.collect().await?;
                    if !count_results.is_empty() && count_results[0].num_rows() > 0 {
                        print_simple_result("    Total rows", &count_results);
                    }
                }
                Err(e) => {
                    println!("    ⚠️  Could not count rows: {}", e);
                }
            }
        }
    }
    
    Ok(())
}

async fn test_multid_to_multid_joins(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Testing Multi-Dimensional to Multi-Dimensional Joins");
    
    // Test 1: Join 2D (timeseries) with 3D (multi_var) 
    println!("  Test 1: 2D ⋈ 3D - timeseries with multi_var");
    
    let df = ctx.sql("
        SELECT 
            t.dim_0 as time_idx,
            m.dim_0 as multi_time_idx,
            COUNT(*) as matches
        FROM timeseries t
        JOIN multi_var m ON t.dim_0 = m.dim_0
        GROUP BY t.dim_0, m.dim_0
        ORDER BY t.dim_0
        LIMIT 5
    ").await?;
    let results = df.collect().await?;
    print_results("2D-3D Join", &results);
    
    // Test 2: Join 3D (business) with 3D (multi_var) - different structures
    println!("  Test 2: 3D ⋈ 3D - business with multi_var (different coord semantics)");
    let df = ctx.sql("
        SELECT 
            b.dim_0 as business_cat,
            m.dim_0 as multi_time,
            COUNT(*) as match_count
        FROM business b
        JOIN multi_var m ON b.dim_0 = m.dim_0
        GROUP BY b.dim_0, m.dim_0
        ORDER BY b.dim_0
        LIMIT 3
    ").await?;
    let results = df.collect().await?;
    print_results("3D-3D Join", &results);
    
    Ok(())
}

async fn test_2d_to_4d_joins(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Testing 2D to 4D Joins");
    
    // Test 1: Join 2D (timeseries) with 4D (weather) on time dimension
    println!("  Test 1: 2D ⋈ 4D - timeseries with weather on time");
    let df = ctx.sql("
        SELECT 
            t.dim_0 as time_idx,
            w.dim_0 as weather_time,
            COUNT(*) as temporal_matches
        FROM timeseries t
        JOIN weather w ON t.dim_0 = w.dim_0
        GROUP BY t.dim_0, w.dim_0
        ORDER BY t.dim_0
        LIMIT 5
    ").await?;
    let results = df.collect().await?;
    print_results("2D-4D Temporal Join", &results);
    
    // Test 2: Join 3D (ocean) with 4D (weather) on spatial dimensions
    println!("  Test 2: 3D ⋈ 4D - ocean with weather on lat coordinate");
    let df = ctx.sql("
        SELECT 
            o.dim_1 as ocean_lat,
            w.dim_1 as weather_lat,
            COUNT(*) as spatial_matches
        FROM ocean o
        JOIN weather w ON o.dim_1 = w.dim_1
        GROUP BY o.dim_1, w.dim_1
        ORDER BY o.dim_1
        LIMIT 5
    ").await?;
    let results = df.collect().await?;
    print_results("3D-4D Spatial Join", &results);
    
    Ok(())
}

async fn test_cross_dimensional_aggregations(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Testing Cross-Dimensional Aggregations");
    
    // Test 1: Multi-table aggregation with different dimensions
    println!("  Test 1: Multi-dimensional aggregation summary");
    let df = ctx.sql("
        WITH dim_stats AS (
            SELECT '2D_timeseries' as dataset, COUNT(*) as total_points FROM timeseries
            UNION ALL
            SELECT '3D_multi_var' as dataset, COUNT(*) as total_points FROM multi_var
            UNION ALL
            SELECT '3D_ocean' as dataset, COUNT(*) as total_points FROM ocean
            UNION ALL
            SELECT '3D_business' as dataset, COUNT(*) as total_points FROM business
            UNION ALL
            SELECT '4D_weather' as dataset, COUNT(*) as total_points FROM weather
        )
        SELECT 
            dataset,
            total_points,
            ROUND(100.0 * total_points / SUM(total_points) OVER (), 2) as percentage
        FROM dim_stats
        ORDER BY total_points DESC
    ").await?;
    let results = df.collect().await?;
    print_results("Cross-Dimensional Stats", &results);
    
    // Test 2: Coordinate range analysis across datasets
    println!("  Test 2: Coordinate range analysis");
    let df = ctx.sql("
        SELECT 
            'timeseries_time' as coord_type,
            MIN(dim_0) as min_val,
            MAX(dim_0) as max_val,
            COUNT(DISTINCT dim_0) as unique_vals
        FROM timeseries
        UNION ALL
        SELECT 
            'multi_var_time' as coord_type,
            MIN(dim_0) as min_val,
            MAX(dim_0) as max_val,
            COUNT(DISTINCT dim_0) as unique_vals
        FROM multi_var
        UNION ALL
        SELECT 
            'weather_time' as coord_type,
            MIN(dim_0) as min_val,
            MAX(dim_0) as max_val,
            COUNT(DISTINCT dim_0) as unique_vals
        FROM weather
    ").await?;
    let results = df.collect().await?;
    print_results("Coordinate Ranges", &results);
    
    Ok(())
}

async fn test_complex_dimensional_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔗 Testing Complex Cross-Dimensional Queries");
    
    // Test 1: Multi-way join across different dimensions
    println!("  Test 1: Three-way join (2D-3D-4D)");
    let df = ctx.sql("
        SELECT 
            t.dim_0 as coord_val,
            COUNT(DISTINCT t.dim_1) as time_coords,
            COUNT(DISTINCT m.dim_1) as multi_coords,  
            COUNT(DISTINCT w.dim_1) as weather_coords
        FROM timeseries t
        LEFT JOIN multi_var m ON t.dim_0 = m.dim_0
        LEFT JOIN weather w ON t.dim_0 = w.dim_0
        GROUP BY t.dim_0
        ORDER BY t.dim_0
        LIMIT 5
    ").await?;
    let results = df.collect().await?;
    print_results("Three-way Join", &results);
    
    // Test 2: Dimensional compatibility analysis
    println!("  Test 2: Dimensional overlap analysis");
    let df = ctx.sql("
        WITH coord_overlaps AS (
            SELECT 
                'time_overlap' as analysis,
                COUNT(*) as overlap_count
            FROM (
                SELECT DISTINCT t.dim_0 FROM timeseries t
                INTERSECT 
                SELECT DISTINCT m.dim_0 FROM multi_var m
            ) overlaps
            UNION ALL
            SELECT 
                'spatial_lat_overlap' as analysis,
                COUNT(*) as overlap_count  
            FROM (
                SELECT DISTINCT o.dim_1 FROM ocean o
                INTERSECT
                SELECT DISTINCT w.dim_1 FROM weather w  
            ) overlaps
        )
        SELECT * FROM coord_overlaps
    ").await?;
    let results = df.collect().await?;
    print_results("Dimensional Overlaps", &results);
    
    // Test 3: Hierarchical coordinate queries
    println!("  Test 3: Hierarchical coordinate analysis");
    let df = ctx.sql("
        SELECT 
            w.dim_0 as time_coord,
            w.dim_1 as lat_coord,
            COUNT(*) as point_count,
            COUNT(DISTINCT w.dim_2) as lon_variations,
            COUNT(DISTINCT w.dim_3) as alt_variations
        FROM weather w
        GROUP BY w.dim_0, w.dim_1
        ORDER BY w.dim_0, w.dim_1
        LIMIT 8
    ").await?;
    let results = df.collect().await?;
    print_results("Hierarchical Coords", &results);
    
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

fn print_simple_result(label: &str, results: &[arrow_array::RecordBatch]) {
    if !results.is_empty() && results[0].num_rows() > 0 {
        let value = format_array_value(results[0].column(0), 0);
        println!("{}: {}", label, value);
    }
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