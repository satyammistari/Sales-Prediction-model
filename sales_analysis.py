import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set display options for better visualization
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load the dataset
def load_and_prepare_data(file_path='Sample - Superstore.csv'):
    """
    Load and prepare the superstore dataset
    """
    print("Loading and preparing data...")
    
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            # Load data with specific encoding
            print(f"Trying to load data with {encoding} encoding...")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding.")
            
            # Display basic information
            print(f"Dataset shape: {df.shape}")
            print("\nDataset info:")
            print(df.info())
            print("\nSample data:")
            print(df.head())
            
            return df
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
    
    # If all encodings fail
    raise ValueError("Could not load the dataset with any of the tried encodings.")

def preprocess_data(df):
    """
    Preprocess the data:
    - Convert date columns to datetime format
    - Handle missing values
    - Handle outliers
    """
    print("\nPreprocessing data...")
    
    # Make a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Convert date columns to datetime
    date_columns = ['Order Date', 'Ship Date']
    for col in date_columns:
        if col in processed_df.columns:
            processed_df[col] = pd.to_datetime(processed_df[col])
            
            # Extract useful datetime features
            processed_df[f'{col[:5]}_Year'] = processed_df[col].dt.year
            processed_df[f'{col[:5]}_Month'] = processed_df[col].dt.month
            processed_df[f'{col[:5]}_Quarter'] = processed_df[col].dt.quarter
            processed_df[f'{col[:5]}_DayOfWeek'] = processed_df[col].dt.dayofweek
    
    # Check for missing values
    missing_values = processed_df.isnull().sum()
    print(f"\nMissing values:\n{missing_values[missing_values > 0]}")
    
    # Fill missing values based on column type
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    
    # Fill numeric columns with median (more robust to outliers than mean)
    for col in numeric_cols:
        processed_df[col].fillna(processed_df[col].median(), inplace=True)
    
    # Fill categorical columns with mode
    for col in categorical_cols:
        processed_df[col].fillna(processed_df[col].mode()[0], inplace=True)
    
    # Handle outliers in numeric columns using IQR method
    for col in ['Sales', 'Profit', 'Quantity', 'Discount']:
        if col in processed_df.columns:
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            processed_df[col] = np.where(
                processed_df[col] < lower_bound,
                lower_bound,
                np.where(processed_df[col] > upper_bound, upper_bound, processed_df[col])
            )
    
    return processed_df

def temporal_sales_analysis(df):
    """
    Analyze sales patterns over time
    """
    print("\nPerforming temporal sales analysis...")
    
    # Monthly sales trends
    monthly_sales = df.groupby(df['Order Date'].dt.to_period('M')).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count'
    }).rename(columns={'Order ID': 'Order Count'})
    
    monthly_sales.index = monthly_sales.index.astype(str)
    
    # Create monthly sales trend visualization
    fig = px.line(
        monthly_sales.reset_index(),
        x='Order Date',
        y=['Sales', 'Profit'],
        title='Monthly Sales and Profit Trends',
        labels={'value': 'Amount', 'variable': 'Metric'},
        template='plotly_white'
    )
    fig.write_html("monthly_sales_trends.html")
    
    # Calculate month-over-month growth
    monthly_sales['Sales_Growth_Rate'] = monthly_sales['Sales'].pct_change() * 100
    monthly_sales['Profit_Growth_Rate'] = monthly_sales['Profit'].pct_change() * 100
    
    print("\nMonth-over-month growth rates (recent months):")
    print(monthly_sales[['Sales', 'Sales_Growth_Rate', 'Profit', 'Profit_Growth_Rate']].tail())
    
    return monthly_sales

def category_analysis(df):
    """
    Analyze performance by product category and sub-category
    """
    print("\nPerforming category analysis...")
    
    # Category performance
    category_performance = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Order ID': pd.Series.nunique
    }).rename(columns={'Order ID': 'Order Count'})
    
    category_performance['Profit_Margin'] = (category_performance['Profit'] / category_performance['Sales']) * 100
    category_performance.sort_values('Sales', ascending=False, inplace=True)
    
    print("\nCategory Performance:")
    print(category_performance)
    
    # Sub-category performance
    subcategory_performance = df.groupby(['Category', 'Sub-Category']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Order ID': pd.Series.nunique
    }).rename(columns={'Order ID': 'Order Count'})
    
    subcategory_performance['Profit_Margin'] = (subcategory_performance['Profit'] / subcategory_performance['Sales']) * 100
    subcategory_performance.sort_values('Sales', ascending=False, inplace=True)
    
    print("\nTop 10 Sub-Categories by Sales:")
    print(subcategory_performance.head(10))
    
    # Create category visualization
    fig = px.treemap(
        df, 
        path=['Category', 'Sub-Category'], 
        values='Sales',
        color='Profit',
        color_continuous_scale='RdBu',
        title='Sales and Profit by Category and Sub-Category'
    )
    fig.write_html("category_analysis.html")
    
    return category_performance, subcategory_performance

def segment_analysis(df):
    """
    Analyze customer segment performance
    """
    print("\nPerforming customer segment analysis...")
    
    # Segment performance
    segment_performance = df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Customer ID': pd.Series.nunique,
        'Order ID': pd.Series.nunique
    }).rename(columns={'Customer ID': 'Customer Count', 'Order ID': 'Order Count'})
    
    segment_performance['Profit_Margin'] = (segment_performance['Profit'] / segment_performance['Sales']) * 100
    segment_performance['Sales_per_Customer'] = segment_performance['Sales'] / segment_performance['Customer Count']
    segment_performance['Profit_per_Customer'] = segment_performance['Profit'] / segment_performance['Customer Count']
    
    print("\nCustomer Segment Performance:")
    print(segment_performance)
    
    # Create segment visualization
    fig = px.bar(
        segment_performance.reset_index(),
        x='Segment',
        y=['Sales_per_Customer', 'Profit_per_Customer'],
        barmode='group',
        title='Sales and Profit per Customer by Segment',
        labels={'value': 'Amount', 'variable': 'Metric'},
        template='plotly_white'
    )
    fig.write_html("segment_analysis.html")
    
    return segment_performance

def build_sales_prediction_model(df):
    """
    Build a machine learning model to predict sales
    """
    print("\nBuilding sales prediction model...")
    
    # Create a copy for modeling
    model_df = df.copy()
    
    # Define features and target
    target = 'Sales'
    
    # Exclude non-predictive columns
    exclude_cols = ['Sales', 'Profit', 'Order ID', 'Order Date', 'Ship Date', 
                    'Customer ID', 'Customer Name', 'Product ID', 'Product Name', 
                    'Row ID']
    
    features = [col for col in model_df.columns if col not in exclude_cols]
    
    print(f"\nFeatures used for prediction: {features}")
    
    # Prepare X and y
    X = model_df[features]
    y = model_df[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify numerical and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"Root Mean Squared Error: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    if hasattr(pipeline['model'], 'feature_importances_'):
        # Get feature names after one-hot encoding
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                for col in cols:
                    for category in trans.categories_[cols.index(col)]:
                        feature_names.append(f"{col}_{category}")
        
        # Check if lengths match, if not use indexed features
        if len(feature_names) == len(pipeline['model'].feature_importances_):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': pipeline['model'].feature_importances_
            })
        else:
            feature_importance = pd.DataFrame({
                'Feature': [f'Feature_{i}' for i in range(len(pipeline['model'].feature_importances_))],
                'Importance': pipeline['model'].feature_importances_
            })
        
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        # Create feature importance visualization
        fig = px.bar(
            feature_importance.head(15),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Feature Importance for Sales Prediction',
            template='plotly_white'
        )
        fig.write_html("feature_importance.html")
    
    return pipeline, feature_importance, r2

def operational_insights(df):
    """
    Generate operational insights based on sales-to-profit ratios and other metrics
    """
    print("\nGenerating operational insights...")
    
    # Calculate sales-to-profit ratios by category
    category_ratio = df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    })
    category_ratio['Profit_Margin'] = (category_ratio['Profit'] / category_ratio['Sales']) * 100
    category_ratio['Sales_to_Profit_Ratio'] = category_ratio['Sales'] / category_ratio['Profit']
    
    print("\nSales-to-Profit Ratios by Category:")
    print(category_ratio)
    
    # Regional performance
    regional_performance = df.groupby(['Region', 'State']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique
    }).rename(columns={'Order ID': 'Order Count'})
    
    regional_performance['Profit_Margin'] = (regional_performance['Profit'] / regional_performance['Sales']) * 100
    regional_performance.sort_values('Sales', ascending=False, inplace=True)
    
    print("\nTop 10 States by Sales:")
    print(regional_performance.head(10))
    
    # Create map visualization
    state_performance = df.groupby('State').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    })
    state_performance['Profit_Margin'] = (state_performance['Profit'] / state_performance['Sales']) * 100
    
    fig = px.choropleth(
        state_performance.reset_index(),
        locations='State',
        locationmode='USA-states',
        color='Profit_Margin',
        scope='usa',
        color_continuous_scale='RdYlGn',
        title='Profit Margin by State',
        labels={'Profit_Margin': 'Profit Margin (%)'}
    )
    fig.write_html("state_performance.html")
    
    # Shipping analysis
    shipping_performance = df.groupby('Ship Mode').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique
    }).rename(columns={'Order ID': 'Order Count'})
    
    shipping_performance['Profit_Margin'] = (shipping_performance['Profit'] / shipping_performance['Sales']) * 100
    shipping_performance['Avg_Order_Value'] = shipping_performance['Sales'] / shipping_performance['Order Count']
    
    print("\nShipping Mode Performance:")
    print(shipping_performance)
    
    return category_ratio, regional_performance, shipping_performance

def main():
    """
    Main function to run the entire analysis
    """
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Temporal sales analysis
    monthly_sales = temporal_sales_analysis(processed_df)
    
    # Category analysis
    category_performance, subcategory_performance = category_analysis(processed_df)
    
    # Segment analysis
    segment_performance = segment_analysis(processed_df)
    
    # Operational insights
    category_ratio, regional_performance, shipping_performance = operational_insights(processed_df)
    
    # Build prediction model
    model, feature_importance, r2_score = build_sales_prediction_model(processed_df)
    
    print("\n=== Analysis Complete ===")
    print(f"Model accuracy (R²): {r2_score:.4f}")
    print("HTML visualizations have been generated for key insights.")

if __name__ == "__main__":
    main() 