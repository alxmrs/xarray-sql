use std::sync::Arc;

use datafusion::error::DataFusionError;
use datafusion::prelude::*;
use zarrquet::table_provider::ZarrTableProvider;

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
    println!("  Test 1: SELECT * FROM weather LIMIT 5");
    let df = ctx.sql("SELECT * FROM weather LIMIT 5").await?;
    let results = df.collect().await?;
    println!(
        "    ✅ Returned {} rows, {} columns",
        results.iter().map(|b| b.num_rows()).sum::<usize>(),
        if !results.is_empty() {
            results[0].num_columns()
        } else {
            0
        }
    );

    // Test 2: COUNT queries
    println!("  Test 2: SELECT COUNT(*) FROM weather");
    let df = ctx
        .sql("SELECT COUNT(*) as total_rows FROM weather")
        .await?;
    let results = df.collect().await?;
    print_results("COUNT", &results);

    // Test 3: Simple column selection
    println!("  Test 3: SELECT time, lat FROM weather LIMIT 3");
    let df = ctx.sql("SELECT time, lat FROM weather LIMIT 3").await?;
    let results = df.collect().await?;
    print_results("Column Selection", &results);

    Ok(())
}

async fn test_filtering_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Filtering Queries (Predicate Pushdown)");

    // Test 1: Simple WHERE clause
    println!("  Test 1: WHERE time >= 1");
    let df = ctx
        .sql("SELECT COUNT(*) as filtered_count FROM weather WHERE time >= 1")
        .await?;
    let results = df.collect().await?;
    print_results("Simple Filter", &results);

    // Test 2: Multiple conditions
    println!("  Test 2: WHERE time >= 1 AND lat < 2");
    let df = ctx
        .sql("SELECT COUNT(*) as count FROM weather WHERE time >= 1 AND lat < 2")
        .await?;
    let results = df.collect().await?;
    print_results("Multiple Conditions", &results);

    // Test 3: Range queries
    println!("  Test 3: WHERE time BETWEEN 0 AND 1");
    let df = ctx
        .sql("SELECT time, COUNT(*) as count FROM weather WHERE time BETWEEN 0 AND 1 GROUP BY time")
        .await?;
    let results = df.collect().await?;
    print_results("Range Query", &results);

    // Test 4: Complex boolean logic
    println!("  Test 4: WHERE (time = 0 AND lat = 0) OR (time = 2 AND lat = 1)");
    let df = ctx.sql("SELECT COUNT(*) as count FROM weather WHERE (time = 0 AND lat = 0) OR (time = 2 AND lat = 1)").await?;
    let results = df.collect().await?;
    print_results("Complex Boolean", &results);

    Ok(())
}

async fn test_aggregation_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Aggregation Queries");

    // Test 1: Basic aggregations
    println!("  Test 1: Basic aggregations (MIN, MAX, AVG)");
    let df = ctx
        .sql(
            "
        SELECT 
            COUNT(*) as count,
            MIN(time) as min_time,
            MAX(time) as max_time,
            AVG(CAST(time AS DOUBLE)) as avg_time
        FROM weather
    ",
        )
        .await?;
    let results = df.collect().await?;
    print_results("Basic Aggregations", &results);

    // Test 2: GROUP BY
    println!("  Test 2: GROUP BY dim_0");
    let df = ctx
        .sql(
            "
        SELECT 
            time,
            COUNT(*) as count,
            MIN(lat) as min_lat,
            MAX(lat) as max_lat
        FROM weather 
        GROUP BY time 
        ORDER BY time
    ",
        )
        .await?;
    let results = df.collect().await?;
    print_results("GROUP BY", &results);

    // Test 3: Multiple GROUP BY
    println!("  Test 3: GROUP BY dim_0, dim_1");
    let df = ctx
        .sql(
            "
        SELECT 
            time, 
            lat, 
            COUNT(*) as count 
        FROM weather 
        GROUP BY time, lat 
        ORDER BY time, lat
    ",
        )
        .await?;
    let results = df.collect().await?;
    print_results("Multiple GROUP BY", &results);

    // Test 4: HAVING clause
    println!("  Test 4: GROUP BY with HAVING");
    let df = ctx
        .sql(
            "
        SELECT 
            time, 
            COUNT(*) as count 
        FROM weather 
        GROUP BY time 
        HAVING COUNT(*) > 2
        ORDER BY time
    ",
        )
        .await?;
    let results = df.collect().await?;
    print_results("HAVING clause", &results);

    Ok(())
}

async fn test_projection_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Projection Queries");

    // Test 1: Column aliases
    println!("  Test 1: Column aliases");
    let df = ctx
        .sql(
            "
        SELECT 
            time as time_coord,
            lat as lat_coord,
            lon as lon_coord
        FROM weather 
        LIMIT 3
    ",
        )
        .await?;
    let results = df.collect().await?;
    print_results("Column Aliases", &results);

    // Test 2: Computed columns
    println!("  Test 2: Computed columns");
    let df = ctx
        .sql(
            "
        SELECT 
            time,
            lat,
            (time * 10 + lat) as computed_id
        FROM weather 
        LIMIT 5
    ",
        )
        .await?;
    let results = df.collect().await?;
    print_results("Computed Columns", &results);

    // Test 3: DISTINCT
    println!("  Test 3: DISTINCT values");
    let df = ctx
        .sql("SELECT DISTINCT time FROM weather ORDER BY time")
        .await?;
    let results = df.collect().await?;
    print_results("DISTINCT", &results);

    Ok(())
}

async fn test_self_joins(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Self-Joins");

    // Test 1: Self join on coordinates
    println!("  Test 1: Self join on shared coordinates");
    let df = ctx
        .sql(
            "
        SELECT 
            a.time,
            a.lat,
            COUNT(*) as pair_count
        FROM weather a
        JOIN weather b ON a.time = b.time AND a.lat = b.lat
        GROUP BY a.time, a.lat
        LIMIT 5
    ",
        )
        .await?;
    let results = df.collect().await?;
    print_results("Self Join", &results);

    Ok(())
}

async fn test_multi_dataset_joins(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Multi-Dataset Joins");

    // Test 1: Cross-dataset join (if tables exist)
    println!("  Test 1: Attempting cross-dataset joins...");

    // Check which tables are available
    let available_tables = ctx
        .catalog("datafusion")
        .unwrap()
        .schema("public")
        .unwrap()
        .table_names();

    println!("    Available tables: {:?}", available_tables);

    // Try weather + ocean join (both have lat coordinates)
    if available_tables.contains(&"weather".to_string())
        && available_tables.contains(&"ocean".to_string())
    {
        println!("  Test 1a: Weather + Ocean join on lat");

        // First, let's see what the schemas look like
        let weather_schema = ctx.sql("SELECT * FROM weather LIMIT 1").await;
        let ocean_schema = ctx.sql("SELECT * FROM ocean LIMIT 1").await;

        if weather_schema.is_ok() && ocean_schema.is_ok() {
            let df = ctx
                .sql(
                    "
                SELECT 
                    w.lat as weather_lat,
                    o.lat as ocean_lat,
                    COUNT(*) as match_count
                FROM weather w
                JOIN ocean o ON w.lat = o.lat
                GROUP BY w.lat, o.lat
                LIMIT 5
            ",
                )
                .await?;
            let results = df.collect().await?;
            print_results("Weather-Ocean Join", &results);
        }
    }

    // Test 2: Try different dimensional joins
    if available_tables.contains(&"timeseries".to_string())
        && available_tables.contains(&"weather".to_string())
    {
        println!("  Test 1b: Timeseries + Weather join");
        let df = ctx
            .sql(
                "
            SELECT 
                t.time as timeseries_time,
                w.time as weather_time,
                COUNT(*) as matches
            FROM timeseries t
            JOIN weather w ON t.time = w.time
            GROUP BY t.time, w.time
            LIMIT 5
        ",
            )
            .await?;
        let results = df.collect().await?;
        print_results("Timeseries-Weather Join", &results);
    }

    Ok(())
}

async fn test_complex_queries(ctx: &SessionContext) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔍 Testing Complex SQL Queries");

    // Test 1: Subquery
    println!("  Test 1: Subquery");
    let df = ctx
        .sql(
            "
        SELECT 
            time,
            coord_count
        FROM (
            SELECT 
                time,
                COUNT(*) as coord_count
            FROM weather
            GROUP BY time
        ) subq
        WHERE coord_count > 3
        ORDER BY time
    ",
        )
        .await?;
    let results = df.collect().await?;
    print_results("Subquery", &results);

    // Test 2: CTE (Common Table Expression)
    println!("  Test 2: Common Table Expression (CTE)");
    let df = ctx
        .sql(
            "
        WITH coord_stats AS (
            SELECT 
                time,
                lat,
                COUNT(*) as point_count
            FROM weather
            GROUP BY time, lat
        )
        SELECT 
            time,
            SUM(point_count) as total_points,
            AVG(CAST(point_count AS DOUBLE)) as avg_points
        FROM coord_stats
        GROUP BY time
        ORDER BY time
    ",
        )
        .await?;
    let results = df.collect().await?;
    print_results("CTE", &results);

    // Test 3: Window functions (if supported)
    println!("  Test 3: Window functions");
    let df = ctx
        .sql(
            "
        SELECT 
            time,
            lat,
            ROW_NUMBER() OVER (PARTITION BY time ORDER BY lat) as row_num
        FROM weather
        ORDER BY time, lat
        LIMIT 10
    ",
        )
        .await?;
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
    println!(
        "      Rows: {}, Columns: {}",
        total_rows,
        results[0].num_columns()
    );

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
