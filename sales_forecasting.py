import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SalesForecaster:
    """
    A class for forecasting sales based on historical data
    """
    def __init__(self, data_path='Sample - Superstore.csv'):
        """
        Initialize the forecaster with the data path
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.model = None
        self.category_models = {}
        self.segment_models = {}
        self.tscv = TimeSeriesSplit(n_splits=5)
        
    def load_data(self):
        """
        Load and prepare the dataset
        """
        print("Loading data...")
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
        
        for encoding in encodings:
            try:
                # Load data with specific encoding
                print(f"Trying to load data with {encoding} encoding...")
                self.df = pd.read_csv(self.data_path, encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding.")
                print(f"Dataset shape: {self.df.shape}")
                return self.df
            except Exception as e:
                print(f"Error with {encoding} encoding: {e}")
        
        # If all encodings fail
        raise ValueError("Could not load the dataset with any of the tried encodings.")
    
    def preprocess_data(self):
        """
        Preprocess the data for forecasting
        """
        print("Preprocessing data...")
        
        # Make a copy to avoid modifying original data
        self.processed_df = self.df.copy()
        
        # Convert date columns to datetime
        date_columns = ['Order Date', 'Ship Date']
        for col in date_columns:
            if col in self.processed_df.columns:
                try:
                    self.processed_df[col] = pd.to_datetime(self.processed_df[col], errors='coerce')
                except Exception as e:
                    print(f"Error converting {col} to datetime: {e}")
                    print(f"Attempting alternative parsing method...")
                    self.processed_df[col] = pd.to_datetime(self.processed_df[col], format='%m/%d/%Y', errors='coerce')
        
        # Extract date features
        self.processed_df['Order_Year'] = self.processed_df['Order Date'].dt.year
        self.processed_df['Order_Month'] = self.processed_df['Order Date'].dt.month
        self.processed_df['Order_Quarter'] = self.processed_df['Order Date'].dt.quarter
        self.processed_df['Order_DayOfWeek'] = self.processed_df['Order Date'].dt.dayofweek
        self.processed_df['Order_DayOfMonth'] = self.processed_df['Order Date'].dt.day
        
        # Use safe isocalendar access
        try:
            self.processed_df['Order_WeekOfYear'] = self.processed_df['Order Date'].dt.isocalendar().week
        except:
            # Alternative for older pandas versions
            self.processed_df['Order_WeekOfYear'] = self.processed_df['Order Date'].dt.week
        
        # Calculate lag between order and ship date
        try:
            self.processed_df['Ship_Lag'] = (self.processed_df['Ship Date'] - self.processed_df['Order Date']).dt.days
        except:
            print("Warning: Could not calculate ship lag, setting to 0")
            self.processed_df['Ship_Lag'] = 0
        
        # Handle missing values
        numeric_cols = self.processed_df.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = self.processed_df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            self.processed_df[col].fillna(self.processed_df[col].median(), inplace=True)
        
        for col in categorical_cols:
            self.processed_df[col].fillna(self.processed_df[col].mode()[0], inplace=True)
        
        # Handle outliers in key numeric columns
        for col in ['Sales', 'Profit', 'Quantity', 'Discount']:
            if col in self.processed_df.columns:
                Q1 = self.processed_df[col].quantile(0.25)
                Q3 = self.processed_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                self.processed_df[col] = np.where(
                    self.processed_df[col] < lower_bound,
                    lower_bound,
                    np.where(self.processed_df[col] > upper_bound, upper_bound, self.processed_df[col])
                )
        
        return self.processed_df
    
    def create_time_aggregated_data(self):
        """
        Create time-aggregated data for time series forecasting
        """
        print("Creating time-aggregated data...")
        
        # Ensure Order Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.processed_df['Order Date']):
            self.processed_df['Order Date'] = pd.to_datetime(self.processed_df['Order Date'])
        
        # Create monthly aggregation using a different approach
        self.processed_df['Year'] = self.processed_df['Order Date'].dt.year
        self.processed_df['Month'] = self.processed_df['Order Date'].dt.month
        
        monthly_data = self.processed_df.groupby(['Year', 'Month']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum',
            'Discount': 'mean',
            'Order ID': 'count'
        }).reset_index()
        
        # Rename columns
        monthly_data.rename(columns={'Order ID': 'Order_Count', 'Discount': 'Avg_Discount'}, inplace=True)
        
        # Create Date column
        monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
        monthly_data.sort_values('Date', inplace=True)
        
        # Create lagged features
        for lag in [1, 2, 3, 6, 12]:  # Previous months
            if len(monthly_data) > lag:
                monthly_data[f'Sales_Lag_{lag}'] = monthly_data['Sales'].shift(lag)
                monthly_data[f'Profit_Lag_{lag}'] = monthly_data['Profit'].shift(lag)
                monthly_data[f'Quantity_Lag_{lag}'] = monthly_data['Quantity'].shift(lag)
        
        # Create rolling window features
        for window in [3, 6, 12]:  # Window sizes
            if len(monthly_data) > window:
                monthly_data[f'Sales_Rolling_Mean_{window}'] = monthly_data['Sales'].rolling(window=window).mean().shift(1)
                monthly_data[f'Sales_Rolling_Std_{window}'] = monthly_data['Sales'].rolling(window=window).std().shift(1)
        
        # Add cyclical features
        monthly_data['Month_Sin'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
        monthly_data['Month_Cos'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)
        
        # Drop rows with NaN values (due to lag creation)
        monthly_data.dropna(inplace=True)
        
        return monthly_data
    
    def create_category_aggregated_data(self):
        """
        Create category-aggregated data for category-specific forecasting
        """
        print("Creating category-aggregated data...")
        
        # Ensure Order Date is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.processed_df['Order Date']):
            self.processed_df['Order Date'] = pd.to_datetime(self.processed_df['Order Date'])
        
        # Make sure Year and Month columns exist
        if 'Year' not in self.processed_df.columns:
            self.processed_df['Year'] = self.processed_df['Order Date'].dt.year
        if 'Month' not in self.processed_df.columns:
            self.processed_df['Month'] = self.processed_df['Order Date'].dt.month
        
        # Group by category and month
        category_monthly = self.processed_df.groupby(['Category', 'Year', 'Month']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum',
            'Discount': 'mean',
            'Order ID': 'count'
        }).reset_index()
        
        # Rename columns
        category_monthly.rename(columns={'Order ID': 'Order_Count', 'Discount': 'Avg_Discount'}, inplace=True)
        
        # Create Date column
        category_monthly['Date'] = pd.to_datetime(category_monthly[['Year', 'Month']].assign(day=1))
        category_monthly.sort_values(['Category', 'Date'], inplace=True)
        
        # For each category, create lag features
        category_dfs = {}
        for category in category_monthly['Category'].unique():
            cat_df = category_monthly[category_monthly['Category'] == category].copy()
            
            # Create lagged features
            for lag in [1, 2, 3]:  # Previous months
                if len(cat_df) > lag:
                    cat_df[f'Sales_Lag_{lag}'] = cat_df['Sales'].shift(lag)
                    cat_df[f'Profit_Lag_{lag}'] = cat_df['Profit'].shift(lag)
            
            # Create rolling window features
            for window in [3, 6]:  # Window sizes
                if len(cat_df) > window:
                    cat_df[f'Sales_Rolling_Mean_{window}'] = cat_df['Sales'].rolling(window=window).mean().shift(1)
            
            # Add cyclical features
            cat_df['Month_Sin'] = np.sin(2 * np.pi * cat_df['Month'] / 12)
            cat_df['Month_Cos'] = np.cos(2 * np.pi * cat_df['Month'] / 12)
            
            # Drop rows with NaN values
            cat_df.dropna(inplace=True)
            
            category_dfs[category] = cat_df
        
        return category_dfs
    
    def build_global_forecast_model(self, monthly_data):
        """
        Build a time series forecast model for overall sales
        """
        print("Building global forecast model...")
        
        # Define features and target
        features = [col for col in monthly_data.columns if col not in [
            'Year', 'Month', 'Date', 'Sales', 'Profit', 'Quantity', 'Order_Count'
        ]]
        
        X = monthly_data[features]
        y = monthly_data['Sales']
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_model = None
        best_rmse = float('inf')
        results = {}
        
        # Perform time series cross-validation
        for name, model in models.items():
            cv_scores = []
            
            for train_idx, test_idx in self.tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                cv_scores.append(rmse)
            
            mean_rmse = np.mean(cv_scores)
            results[name] = mean_rmse
            
            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_model = model
        
        # Retrain best model on full dataset
        best_model.fit(X, y)
        self.model = best_model
        
        # Print model comparison
        print("\nModel comparison:")
        for name, rmse in results.items():
            print(f"{name}: RMSE = ${rmse:.2f}")
        
        print(f"\nSelected best model with RMSE: ${best_rmse:.2f}")
        
        # Feature importance for tree-based models
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))
            
            # Create feature importance visualization
            fig = px.bar(
                feature_importance.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Feature Importance for Global Sales Forecast',
                template='plotly_white'
            )
            fig.write_html("global_feature_importance.html")
        
        return best_model, results
    
    def build_category_forecast_models(self, category_dfs):
        """
        Build forecast models for each category
        """
        print("\nBuilding category-specific forecast models...")
        
        category_results = {}
        
        for category, cat_df in category_dfs.items():
            print(f"\nTraining model for category: {category}")
            
            # Define features and target
            features = [col for col in cat_df.columns if col not in [
                'Category', 'Year', 'Month', 'Date', 'Sales', 'Profit', 'Quantity', 'Order_Count'
            ]]
            
            X = cat_df[features]
            y = cat_df['Sales']
            
            # Use GradientBoostingRegressor for category models
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # Perform time series cross-validation
            cv_scores = []
            
            for train_idx, test_idx in self.tscv.split(X):
                # Ensure we have enough data
                if len(train_idx) < 5 or len(test_idx) < 2:
                    continue
                    
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                cv_scores.append(rmse)
            
            if cv_scores:
                mean_rmse = np.mean(cv_scores)
                category_results[category] = mean_rmse
                print(f"Category {category} RMSE: ${mean_rmse:.2f}")
                
                # Train on full dataset
                model.fit(X, y)
                self.category_models[category] = model
            else:
                print(f"Not enough data for reliable cross-validation for category {category}")
        
        return self.category_models, category_results
    
    def forecast_future_sales(self, months_ahead=3):
        """
        Forecast sales for future months
        """
        print(f"\nForecasting sales for {months_ahead} months ahead...")
        
        # Get the latest monthly data
        monthly_data = self.create_time_aggregated_data()
        last_date = monthly_data['Date'].max()
        
        # Generate future dates
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, months_ahead+1)]
        
        # Prepare dataframe for future predictions
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['Year'] = future_df['Date'].dt.year
        future_df['Month'] = future_df['Date'].dt.month
        
        # Last values of metrics
        last_values = monthly_data.iloc[-1]
        
        # Make sure all the features used in training are present in the prediction data
        # First, get all feature names from monthly_data
        feature_columns = [col for col in monthly_data.columns if col not in [
            'Year', 'Month', 'Date', 'Sales', 'Profit', 'Quantity', 'Order_Count'
        ]]
        
        # Initialize all features with defaults
        for feature in feature_columns:
            if feature not in future_df.columns:
                future_df[feature] = 0.0  # Default value
        
        # Add features for prediction to future dataframe
        for lag in [1, 2, 3, 6, 12]:  # Previous months
            # For the first few lags, use recent historical data
            if lag == 1:
                future_df[f'Sales_Lag_{lag}'] = last_values['Sales']
                future_df[f'Profit_Lag_{lag}'] = last_values['Profit']
                future_df[f'Quantity_Lag_{lag}'] = last_values['Quantity']
            else:
                idx = -lag
                if abs(idx) <= len(monthly_data):
                    future_df[f'Sales_Lag_{lag}'] = monthly_data.iloc[idx]['Sales']
                    future_df[f'Profit_Lag_{lag}'] = monthly_data.iloc[idx]['Profit']
                    future_df[f'Quantity_Lag_{lag}'] = monthly_data.iloc[idx]['Quantity']
        
        # Add rolling features
        for window in [3, 6, 12]:  # Window sizes
            future_df[f'Sales_Rolling_Mean_{window}'] = monthly_data['Sales'].rolling(window=window).mean().iloc[-1]
            future_df[f'Sales_Rolling_Std_{window}'] = monthly_data['Sales'].rolling(window=window).std().iloc[-1]
        
        # Add cyclical features
        future_df['Month_Sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
        future_df['Month_Cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
        
        # Add average discount
        future_df['Avg_Discount'] = monthly_data['Avg_Discount'].mean()
        
        # Check that all required features are available for prediction
        features_needed = list(monthly_data.columns)
        features_needed.remove('Sales')  # This is the target, not a feature
        features_needed.remove('Date')   # Not a model feature
        features_needed.remove('Year')   # Not used directly
        features_needed.remove('Month')  # Not used directly
        
        # For forecasting we don't need the historical values we're trying to predict
        if 'Profit' in features_needed: features_needed.remove('Profit')
        if 'Quantity' in features_needed: features_needed.remove('Quantity')
        if 'Order_Count' in features_needed: features_needed.remove('Order_Count')
        
        # Make sure all required features are in future_df
        missing_features = [f for f in features_needed if f not in future_df.columns]
        if missing_features:
            print(f"Warning: Missing features for prediction: {missing_features}")
            for feature in missing_features:
                # Get a reasonable default value
                if feature in monthly_data.columns:
                    future_df[feature] = monthly_data[feature].mean()
                else:
                    future_df[feature] = 0.0
        
        try:
            # Make predictions using only available features in the trained model
            print("Making sales forecast predictions...")
            forecast = self.model.predict(future_df[features_needed])
            future_df['Forecasted_Sales'] = forecast
            
            # Reset current month value for reference
            current_month_sales = last_values['Sales']
            print(f"\nCurrent month sales: ${current_month_sales:.2f}")
            print("\nSales Forecast:")
            
            for i, row in future_df.iterrows():
                date_str = row['Date'].strftime('%B %Y')
                forecasted_sales = row['Forecasted_Sales']
                print(f"{date_str}: ${forecasted_sales:.2f}")
            
            # Create forecast visualization
            # Combine historical and forecasted data
            historical = monthly_data[['Date', 'Sales']].copy()
            future = future_df[['Date', 'Forecasted_Sales']].rename(columns={'Forecasted_Sales': 'Sales'})
            future['Forecast'] = True
            historical['Forecast'] = False
            
            combined = pd.concat([historical, future])
            
            fig = px.line(
                combined, 
                x='Date', 
                y='Sales',
                color='Forecast',
                title='Historical Sales and Future Forecast',
                labels={'Sales': 'Sales Amount ($)', 'Date': 'Month'},
                template='plotly_white',
                color_discrete_map={True: 'red', False: 'blue'}
            )
            
            fig.update_layout(
                legend_title_text='',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.write_html("sales_forecast.html")
            
            return future_df
        except Exception as e:
            print(f"Error making forecast: {e}")
            print("Skipping forecast generation.")
    
    def forecast_category_sales(self, months_ahead=3):
        """
        Forecast sales by category for future months
        """
        print(f"\nForecasting category sales for {months_ahead} months ahead...")
        
        category_dfs = self.create_category_aggregated_data()
        category_forecasts = {}
        
        for category, cat_df in category_dfs.items():
            if category not in self.category_models:
                print(f"No model available for category {category}")
                continue
                
            # Get the latest data for this category
            last_date = cat_df['Date'].max()
            
            # Generate future dates
            future_dates = [last_date + timedelta(days=30*i) for i in range(1, months_ahead+1)]
            
            # Prepare dataframe for future predictions
            future_df = pd.DataFrame({'Date': future_dates})
            future_df['Year'] = future_df['Date'].dt.year
            future_df['Month'] = future_df['Date'].dt.month
            
            # Last values of metrics
            last_values = cat_df.iloc[-1]
            
            # Make sure all the features used in training are present in the prediction data
            feature_columns = [col for col in cat_df.columns if col not in [
                'Category', 'Year', 'Month', 'Date', 'Sales', 'Profit', 'Quantity', 'Order_Count'
            ]]
            
            # Initialize all features with defaults
            for feature in feature_columns:
                if feature not in future_df.columns:
                    future_df[feature] = 0.0  # Default value
            
            # Add features for prediction to future dataframe
            for lag in [1, 2, 3]:  # Previous months
                if lag == 1:
                    future_df[f'Sales_Lag_{lag}'] = last_values['Sales']
                    future_df[f'Profit_Lag_{lag}'] = last_values['Profit']
                else:
                    idx = -lag
                    if abs(idx) <= len(cat_df):
                        future_df[f'Sales_Lag_{lag}'] = cat_df.iloc[idx]['Sales']
                        future_df[f'Profit_Lag_{lag}'] = cat_df.iloc[idx]['Profit']
            
            # Add rolling features
            for window in [3, 6]:  # Window sizes
                if len(cat_df) >= window:
                    future_df[f'Sales_Rolling_Mean_{window}'] = cat_df['Sales'].rolling(window=window).mean().iloc[-1]
            
            # Add cyclical features
            future_df['Month_Sin'] = np.sin(2 * np.pi * future_df['Month'] / 12)
            future_df['Month_Cos'] = np.cos(2 * np.pi * future_df['Month'] / 12)
            
            # Add average discount
            future_df['Avg_Discount'] = cat_df['Avg_Discount'].mean()
            
            # Get the model for this category
            model = self.category_models[category]
            
            # Get features needed for prediction
            features_needed = list(cat_df.columns)
            features_needed.remove('Date')
            features_needed.remove('Year')
            features_needed.remove('Month')
            features_needed.remove('Sales')  # Target variable
            features_needed.remove('Category')  # Not needed for prediction
            
            # For forecasting we don't need the historical values we're trying to predict
            if 'Profit' in features_needed: features_needed.remove('Profit')
            if 'Quantity' in features_needed: features_needed.remove('Quantity')
            if 'Order_Count' in features_needed: features_needed.remove('Order_Count')
            
            # Make sure all required features are in future_df
            missing_features = [f for f in features_needed if f not in future_df.columns]
            if missing_features:
                print(f"Warning: Missing features for category {category} prediction: {missing_features}")
                for feature in missing_features:
                    # Get a reasonable default value
                    if feature in cat_df.columns:
                        future_df[feature] = cat_df[feature].mean()
                    else:
                        future_df[feature] = 0.0
            
            try:
                # Make predictions
                print(f"Making forecast for category: {category}")
                forecast = model.predict(future_df[features_needed])
                future_df['Forecasted_Sales'] = forecast
                category_forecasts[category] = future_df
                
                print(f"\nForecast for {category}:")
                for i, row in future_df.iterrows():
                    date_str = row['Date'].strftime('%B %Y')
                    forecasted_sales = row['Forecasted_Sales']
                    print(f"{date_str}: ${forecasted_sales:.2f}")
            except Exception as e:
                print(f"Error making forecast for category {category}: {e}")
                print(f"Skipping forecast for category {category}.")
                continue
        
        if not category_forecasts:
            print("No category forecasts were successfully generated.")
            return {}
            
        # Create category forecast visualization
        try:
            fig = go.Figure()
            
            colors = px.colors.qualitative.Plotly
            for i, (category, future_df) in enumerate(category_forecasts.items()):
                color = colors[i % len(colors)]
                
                # Get historical data
                cat_df = category_dfs[category]
                
                # Add historical line
                fig.add_trace(go.Scatter(
                    x=cat_df['Date'],
                    y=cat_df['Sales'],
                    mode='lines',
                    name=f"{category} (Historical)",
                    line=dict(color=color, width=2)
                ))
                
                # Add forecast line
                fig.add_trace(go.Scatter(
                    x=future_df['Date'],
                    y=future_df['Forecasted_Sales'],
                    mode='lines+markers',
                    line=dict(color=color, width=2, dash='dash'),
                    name=f"{category} (Forecast)",
                    marker=dict(size=8, symbol='diamond')
                ))
            
            fig.update_layout(
                title='Sales Forecast by Category',
                xaxis_title='Month',
                yaxis_title='Sales Amount ($)',
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.write_html("category_forecasts.html")
        except Exception as e:
            print(f"Error creating category forecast visualization: {e}")
        
        return category_forecasts
    
    def run_full_pipeline(self):
        """
        Run the full forecasting pipeline
        """
        # Load data
        self.load_data()
        
        # Preprocess data
        self.preprocess_data()
        
        try:
            # Create time-aggregated data
            monthly_data = self.create_time_aggregated_data()
            
            # Build global forecast model
            self.build_global_forecast_model(monthly_data)
            
            # Create category-aggregated data
            category_dfs = self.create_category_aggregated_data()
            
            # Build category forecast models
            self.build_category_forecast_models(category_dfs)
            
            try:
                # Forecast future sales
                self.forecast_future_sales(months_ahead=6)
            except Exception as e:
                print(f"Error in forecasting future sales: {e}")
                print("Skipping global sales forecast.")
            
            try:
                # Forecast category sales
                self.forecast_category_sales(months_ahead=6)
            except Exception as e:
                print(f"Error in forecasting category sales: {e}")
                print("Skipping category sales forecast.")
            
            print("\nForecasting pipeline complete!")
            print("Generated visualizations:")
            print("- global_feature_importance.html")
            print("- sales_forecast.html")
            print("- category_forecasts.html")
            
        except Exception as e:
            print(f"Error in forecasting pipeline: {e}")
            print("Some forecasting components could not be completed.")

# Main execution
if __name__ == "__main__":
    forecaster = SalesForecaster()
    forecaster.run_full_pipeline() 