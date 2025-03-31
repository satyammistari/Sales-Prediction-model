import os
import time
import pandas as pd
from sales_analysis import (
    load_and_prepare_data, 
    preprocess_data,
    temporal_sales_analysis,
    category_analysis,
    segment_analysis,
    operational_insights,
    build_sales_prediction_model
)
from sales_forecasting import SalesForecaster
from visualizations import create_sales_dashboard

def run_full_analysis():
    """
    Run the full sales analysis pipeline
    """
    start_time = time.time()
    
    print("="*80)
    print(" SUPERSTORE SALES ANALYSIS AND PREDICTION SYSTEM ")
    print("="*80)
    
    # Check if dataset exists
    if not os.path.exists('Sample - Superstore.csv'):
        print("Error: Dataset 'Sample - Superstore.csv' not found!")
        return
    
    print("\n1. INITIAL DATA ANALYSIS")
    print("-" * 40)
    
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    print("\n2. DETAILED SALES ANALYSIS")
    print("-" * 40)
    
    # Run temporal sales analysis
    monthly_sales = temporal_sales_analysis(processed_df)
    
    # Run category analysis
    category_performance, subcategory_performance = category_analysis(processed_df)
    
    # Run segment analysis
    segment_performance = segment_analysis(processed_df)
    
    # Generate operational insights
    category_ratio, regional_performance, shipping_performance = operational_insights(processed_df)
    
    print("\n3. SALES PREDICTION MODEL")
    print("-" * 40)
    
    # Build prediction model
    model, feature_importance, r2_score = build_sales_prediction_model(processed_df)
    
    print("\n4. SALES FORECASTING")
    print("-" * 40)
    
    # Create and run forecaster
    print("Running sales forecasting pipeline...")
    try:
        forecaster = SalesForecaster()
        forecaster.run_full_pipeline()
    except Exception as e:
        print(f"Error in forecasting pipeline: {e}")
        print("Continuing with visualization generation...")
    
    print("\n5. CREATING INTERACTIVE DASHBOARDS")
    print("-" * 40)
    
    # Create interactive visualizations
    print("Generating interactive dashboards...")
    dashboard_files = create_sales_dashboard(df)
    
    # Print summary of outputs
    print("\n" + "="*80)
    print(" ANALYSIS COMPLETE ")
    print("="*80)
    
    print("\nGenerated Files:")
    
    # Basic analysis files
    print("\nBasic Analysis Visualizations:")
    print("- monthly_sales_trends.html")
    print("- category_analysis.html")
    print("- segment_analysis.html")
    print("- feature_importance.html")
    print("- state_performance.html")
    
    # Forecasting files
    print("\nForecasting Visualizations:")
    print("- global_feature_importance.html")
    print("- sales_forecast.html")
    print("- category_forecasts.html")
    
    # Dashboard files
    print("\nInteractive Dashboards:")
    for name, file in dashboard_files.items():
        print(f"- {name}: {file}")
    
    execution_time = time.time() - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds")
    
    print("\nTo view the visualizations, open the HTML files in your web browser.")
    
    # Create a simple index.html to navigate all visualizations
    create_index_html(dashboard_files)
    
    print("\nAn index.html file has been created for easy navigation of all visualizations.")

def create_index_html(dashboard_files):
    """
    Create a simple index.html to navigate all visualizations
    """
    # Basic analysis files
    basic_analysis_files = [
        ("Monthly Sales Trends", "monthly_sales_trends.html"),
        ("Category Analysis", "category_analysis.html"),
        ("Segment Analysis", "segment_analysis.html"),
        ("Feature Importance", "feature_importance.html"),
        ("State Performance", "state_performance.html")
    ]
    
    # Forecasting files
    forecasting_files = [
        ("Global Feature Importance", "global_feature_importance.html"),
        ("Sales Forecast", "sales_forecast.html"),
        ("Category Forecasts", "category_forecasts.html")
    ]
    
    # Dashboard files
    dashboard_list = [(k.replace('_', ' ').title(), v) for k, v in dashboard_files.items()]
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Superstore Sales Analysis Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 20px;
            }
            h2 {
                color: #3498db;
                margin-top: 30px;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 10px;
            }
            ul {
                list-style-type: none;
                padding: 0;
            }
            li {
                margin-bottom: 10px;
            }
            a {
                color: #2980b9;
                text-decoration: none;
                padding: 5px 10px;
                border-radius: 4px;
                transition: background-color 0.3s;
            }
            a:hover {
                background-color: #ecf0f1;
                color: #3498db;
            }
            .section {
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                color: #7f8c8d;
                font-size: 0.9em;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Superstore Sales Analysis Dashboard</h1>
            
            <div class="section">
                <h2>Interactive Dashboards</h2>
                <ul>
    """
    
    # Add dashboard links
    for name, file in dashboard_list:
        html_content += f'                <li><a href="{file}" target="_blank">{name}</a></li>\n'
    
    html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Sales Forecasting</h2>
                <ul>
    """
    
    # Add forecasting links
    for name, file in forecasting_files:
        html_content += f'                <li><a href="{file}" target="_blank">{name}</a></li>\n'
    
    html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Basic Analysis</h2>
                <ul>
    """
    
    # Add basic analysis links
    for name, file in basic_analysis_files:
        html_content += f'                <li><a href="{file}" target="_blank">{name}</a></li>\n'
    
    html_content += """
                </ul>
            </div>
            
            <div class="footer">
                <p>Superstore Sales Analysis and Prediction System</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open("index.html", "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    run_full_analysis() 