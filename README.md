# Sales Analysis and Prediction System

This project provides comprehensive analysis of sales data from the Superstore dataset and builds a machine learning model to predict future sales.

## Key Components

### Data Preparation
- Converts date columns to datetime format for time-based insights
- Handles missing values and outliers to ensure data quality
- Extracts useful temporal features from dates

### Sales Analysis
- **Temporal Sales Trends**: Analyzes monthly sales patterns for inventory planning
- **Category Analysis**: Identifies top-performing product categories and sub-categories

### Profit Analysis
- **Profit Trends**: Identifies the most profitable products for optimizing product mix
- **Segmentation Analysis**: Evaluates profitability by customer segment

### Visualizations
- Uses Plotly for dynamic and interactive visualizations
- Generates HTML reports for key insights

### Operational Insights
- **Sales-to-Profit Ratios**: Assesses efficiency across different segments/categories
- **Customer Segment Performance**: Evaluates the profitability of different customer segments

## Setup and Usage

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Make sure your data file is named `Sample - Superstore.csv` and is in the same directory

3. Run the complete analysis pipeline:
   ```bash
   python main.py
   ```

   This will:
   - Perform data preprocessing
   - Execute sales and profit analysis
   - Build a machine learning prediction model
   - Create time series forecasts for future sales
   - Generate interactive visualizations and dashboards

4. For specific components, you can run individual scripts:
   ```bash
   python sales_analysis.py     # Run basic sales analysis
   python sales_forecasting.py  # Run sales forecasting models
   python visualizations.py     # Create interactive dashboards
   ```

5. After running the analysis, open `index.html` in your web browser to access all visualizations

## Machine Learning Model

The system uses a Random Forest Regressor to predict sales based on various features:
- Features include product categories, customer segments, and temporal attributes
- The model is evaluated using MAE, RMSE and R² metrics
- Feature importance analysis shows which factors most influence sales

## Sales Forecasting

The advanced sales forecasting model:
- Performs time series forecasting with multiple algorithms (Random Forest, Gradient Boosting, Linear Regression)
- Generates category-specific forecasts for more granular predictions
- Compares model performance to select the best forecasting approach
- Provides six-month sales projections with visualizations

## Interactive Dashboards

The system generates several interactive dashboards:
- **Sales Trends**: Monthly sales and profit trends
- **Category Analysis**: Performance by product category and subcategory
- **Segment Analysis**: Customer segment performance metrics
- **Regional Performance**: Geographic analysis with state-level metrics
- **Profit Analysis**: Sales-to-profit ratio visualization
- **Time Metrics**: Day of week and month performance

## Generated Visualizations

The analysis pipeline produces HTML visualizations:
- Basic analysis visualizations for sales, profit, and segments
- Forecasting visualizations with future projections
- Interactive dashboards with multiple charts and insights
- All visualizations are accessible through the generated `index.html` file 