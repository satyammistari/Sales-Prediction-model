import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_prepare_data(file_path='Sample - Superstore.csv'):
    """
    Load and prepare the superstore dataset for visualization
    """
    print("Loading data for visualization...")
    
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
    
    for encoding in encodings:
        try:
            # Load data with specific encoding
            print(f"Trying to load data with {encoding} encoding...")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded with {encoding} encoding.")
            
            # Convert date columns to datetime
            date_columns = ['Order Date', 'Ship Date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            return df
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
    
    # If all encodings fail
    raise ValueError("Could not load the dataset with any of the tried encodings.")

def create_sales_dashboard(df):
    """
    Create comprehensive sales dashboard with multiple visualizations
    """
    print("Creating sales dashboard...")
    
    # Ensure Order Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Order Date']):
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    
    # 1. Monthly Sales and Profit Trends
    monthly_data = df.groupby(df['Order Date'].dt.to_period('M')).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'count'
    }).reset_index()
    
    monthly_data['Order Date'] = monthly_data['Order Date'].astype(str)
    
    fig1 = px.line(
        monthly_data,
        x='Order Date',
        y=['Sales', 'Profit'],
        title='Monthly Sales and Profit Trends',
        labels={'value': 'Amount ($)', 'variable': 'Metric', 'Order Date': 'Month'},
        template='plotly_white'
    )
    
    fig1.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    
    fig1.write_html("sales_trends_dashboard.html")
    
    # 2. Product Category Analysis
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sales by Category', 
            'Profit by Category',
            'Sales by Sub-Category (Top 10)',
            'Profit Margin by Sub-Category'
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.1
    )
    
    # Category Sales - Pie Chart
    category_sales = df.groupby('Category')['Sales'].sum().reset_index()
    fig2.add_trace(
        go.Pie(
            labels=category_sales['Category'],
            values=category_sales['Sales'],
            hole=0.4,
            textinfo='label+percent',
            marker=dict(line=dict(color='white', width=2))
        ),
        row=1, col=1
    )
    
    # Category Profit - Bar Chart
    category_profit = df.groupby('Category')['Profit'].sum().reset_index()
    fig2.add_trace(
        go.Bar(
            x=category_profit['Category'],
            y=category_profit['Profit'],
            marker_color='rgb(55, 83, 109)'
        ),
        row=1, col=2
    )
    
    # Sub-Category Sales - Bar Chart
    subcategory_sales = df.groupby('Sub-Category')['Sales'].sum().reset_index()
    subcategory_sales = subcategory_sales.sort_values('Sales', ascending=False).head(10)
    fig2.add_trace(
        go.Bar(
            x=subcategory_sales['Sub-Category'],
            y=subcategory_sales['Sales'],
            marker_color='rgb(26, 118, 255)'
        ),
        row=2, col=1
    )
    
    # Sub-Category Profit Margin - Bar Chart
    subcategory_profit = df.groupby('Sub-Category').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    subcategory_profit['Profit_Margin'] = (subcategory_profit['Profit'] / subcategory_profit['Sales']) * 100
    subcategory_profit = subcategory_profit.sort_values('Profit_Margin', ascending=False).head(10)
    
    fig2.add_trace(
        go.Bar(
            x=subcategory_profit['Sub-Category'],
            y=subcategory_profit['Profit_Margin'],
            marker_color='rgb(162, 155, 254)'
        ),
        row=2, col=2
    )
    
    fig2.update_layout(
        height=800,
        title_text='Product Category Analysis',
        showlegend=False
    )
    
    fig2.update_yaxes(title_text='Sales ($)', row=2, col=1)
    fig2.update_yaxes(title_text='Profit Margin (%)', row=2, col=2)
    fig2.update_yaxes(title_text='Profit ($)', row=1, col=2)
    
    fig2.update_xaxes(tickangle=45, row=2, col=1)
    fig2.update_xaxes(tickangle=45, row=2, col=2)
    
    fig2.write_html("category_analysis_dashboard.html")
    
    # 3. Customer Segment Analysis
    fig3 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sales by Customer Segment', 
            'Profit by Customer Segment',
            'Average Order Value by Segment',
            'Profit Margin by Segment'
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.1
    )
    
    # Segment Sales - Pie Chart
    segment_sales = df.groupby('Segment')['Sales'].sum().reset_index()
    fig3.add_trace(
        go.Pie(
            labels=segment_sales['Segment'],
            values=segment_sales['Sales'],
            hole=0.4,
            textinfo='label+percent',
            marker=dict(line=dict(color='white', width=2))
        ),
        row=1, col=1
    )
    
    # Segment Profit - Bar Chart
    segment_profit = df.groupby('Segment')['Profit'].sum().reset_index()
    fig3.add_trace(
        go.Bar(
            x=segment_profit['Segment'],
            y=segment_profit['Profit'],
            marker_color='rgb(50, 171, 96)'
        ),
        row=1, col=2
    )
    
    # Average Order Value by Segment
    segment_aov = df.groupby(['Segment', 'Order ID'])['Sales'].sum().reset_index()
    segment_aov = segment_aov.groupby('Segment')['Sales'].mean().reset_index()
    segment_aov.columns = ['Segment', 'Average Order Value']
    
    fig3.add_trace(
        go.Bar(
            x=segment_aov['Segment'],
            y=segment_aov['Average Order Value'],
            marker_color='rgb(214, 39, 40)'
        ),
        row=2, col=1
    )
    
    # Profit Margin by Segment
    segment_margin = df.groupby('Segment').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    segment_margin['Profit_Margin'] = (segment_margin['Profit'] / segment_margin['Sales']) * 100
    
    fig3.add_trace(
        go.Bar(
            x=segment_margin['Segment'],
            y=segment_margin['Profit_Margin'],
            marker_color='rgb(254, 97, 0)'
        ),
        row=2, col=2
    )
    
    fig3.update_layout(
        height=800,
        title_text='Customer Segment Analysis',
        showlegend=False
    )
    
    fig3.update_yaxes(title_text='Average Order Value ($)', row=2, col=1)
    fig3.update_yaxes(title_text='Profit Margin (%)', row=2, col=2)
    fig3.update_yaxes(title_text='Profit ($)', row=1, col=2)
    
    fig3.write_html("segment_analysis_dashboard.html")
    
    # 4. Regional Performance Map
    state_performance = df.groupby('State').agg({
        'Sales': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    state_performance['Profit_Margin'] = (state_performance['Profit'] / state_performance['Sales']) * 100
    
    fig4 = px.choropleth(
        state_performance,
        locations='State',
        locationmode='USA-states',
        color='Profit_Margin',
        scope='usa',
        color_continuous_scale='RdYlGn',
        title='Profit Margin by State',
        labels={'Profit_Margin': 'Profit Margin (%)'},
        range_color=[-10, 30]
    )
    
    fig4.update_layout(
        height=600,
        geo=dict(
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        )
    )
    
    fig4.write_html("regional_performance_dashboard.html")
    
    # 5. Sales-to-Profit Ratio Analysis
    # Create a scatter plot of Sales vs Profit by Sub-Category
    subcategory_performance = df.groupby('Sub-Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    subcategory_performance['Profit_Margin'] = (subcategory_performance['Profit'] / subcategory_performance['Sales']) * 100
    subcategory_performance['Sales_to_Profit_Ratio'] = subcategory_performance['Sales'] / subcategory_performance['Profit']
    
    # Handle infinite values (when Profit is 0)
    subcategory_performance.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Create bubble chart
    fig5 = px.scatter(
        subcategory_performance,
        x='Sales',
        y='Profit',
        size='Quantity',
        color='Profit_Margin',
        hover_name='Sub-Category',
        color_continuous_scale='RdYlGn',
        title='Sub-Category Performance: Sales vs. Profit',
        labels={
            'Sales': 'Total Sales ($)',
            'Profit': 'Total Profit ($)',
            'Quantity': 'Units Sold',
            'Profit_Margin': 'Profit Margin (%)'
        },
        size_max=60
    )
    
    fig5.update_layout(
        height=600,
        coloraxis_colorbar=dict(title='Profit Margin (%)'),
        xaxis=dict(title='Total Sales ($)'),
        yaxis=dict(title='Total Profit ($)')
    )
    
    # Add a reference line for break-even (Profit = 0)
    fig5.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=subcategory_performance['Sales'].max(),
        y1=0,
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig5.add_annotation(
        x=subcategory_performance['Sales'].max() * 0.8,
        y=0,
        text="Break-even Line",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    
    fig5.write_html("profit_analysis_dashboard.html")
    
    # 6. Time-based Metrics Dashboard
    # Analyze sales and profit by day of week and month
    if not pd.api.types.is_datetime64_any_dtype(df['Order Date']):
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        
    df['DayOfWeek'] = df['Order Date'].dt.day_name()
    df['Month'] = df['Order Date'].dt.month_name()
    
    # Order days and months correctly
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    
    # Day of week analysis
    day_metrics = df.groupby('DayOfWeek').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    day_metrics['Profit_Margin'] = (day_metrics['Profit'] / day_metrics['Sales']) * 100
    day_metrics['DayOfWeek'] = pd.Categorical(day_metrics['DayOfWeek'], categories=days_order, ordered=True)
    day_metrics.sort_values('DayOfWeek', inplace=True)
    
    # Month analysis
    month_metrics = df.groupby('Month').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': pd.Series.nunique
    }).reset_index()
    
    month_metrics['Profit_Margin'] = (month_metrics['Profit'] / month_metrics['Sales']) * 100
    month_metrics['Month'] = pd.Categorical(month_metrics['Month'], categories=months_order, ordered=True)
    month_metrics.sort_values('Month', inplace=True)
    
    fig6 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sales by Day of Week', 
            'Profit Margin by Day of Week',
            'Sales by Month', 
            'Profit Margin by Month'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.1
    )
    
    # Sales by Day of Week
    fig6.add_trace(
        go.Bar(
            x=day_metrics['DayOfWeek'],
            y=day_metrics['Sales'],
            marker_color='rgb(55, 126, 184)'
        ),
        row=1, col=1
    )
    
    # Profit Margin by Day of Week
    fig6.add_trace(
        go.Bar(
            x=day_metrics['DayOfWeek'],
            y=day_metrics['Profit_Margin'],
            marker_color='rgb(77, 175, 74)'
        ),
        row=1, col=2
    )
    
    # Sales by Month
    fig6.add_trace(
        go.Bar(
            x=month_metrics['Month'],
            y=month_metrics['Sales'],
            marker_color='rgb(152, 78, 163)'
        ),
        row=2, col=1
    )
    
    # Profit Margin by Month
    fig6.add_trace(
        go.Bar(
            x=month_metrics['Month'],
            y=month_metrics['Profit_Margin'],
            marker_color='rgb(255, 127, 0)'
        ),
        row=2, col=2
    )
    
    fig6.update_layout(
        height=800,
        title_text='Time-based Performance Metrics',
        showlegend=False
    )
    
    fig6.update_yaxes(title_text='Sales ($)', row=1, col=1)
    fig6.update_yaxes(title_text='Profit Margin (%)', row=1, col=2)
    fig6.update_yaxes(title_text='Sales ($)', row=2, col=1)
    fig6.update_yaxes(title_text='Profit Margin (%)', row=2, col=2)
    
    fig6.update_xaxes(tickangle=45, row=2, col=1)
    fig6.update_xaxes(tickangle=45, row=2, col=2)
    
    fig6.write_html("time_metrics_dashboard.html")
    
    print("All visualizations created successfully.")
    return {
        "sales_trends": "sales_trends_dashboard.html",
        "category_analysis": "category_analysis_dashboard.html",
        "segment_analysis": "segment_analysis_dashboard.html",
        "regional_performance": "regional_performance_dashboard.html",
        "profit_analysis": "profit_analysis_dashboard.html",
        "time_metrics": "time_metrics_dashboard.html"
    }

if __name__ == "__main__":
    # Load the data
    df = load_and_prepare_data()
    
    # Create visualizations
    dashboard_files = create_sales_dashboard(df)
    
    print("\nDashboard files created:")
    for name, file in dashboard_files.items():
        print(f"- {name}: {file}") 