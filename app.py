import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="NIFTY 50 Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"  # This will collapse the sidebar by default
)

# Add title and description
st.title("NIFTY 50 Analysis Dashboard")
st.markdown("""
This dashboard provides comprehensive analysis of NIFTY 50 index performance including:
- Historical price trends
- Moving averages
- Trading volume analysis
- Future price predictions using Facebook Prophet
""")

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('data/NIFTY 50-16-05-2024-to-16-05-2025.csv')
    # Clean column names by stripping whitespace
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%Y')
    df.sort_values('Date', inplace=True)
    
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate daily returns
    df['Daily_Return'] = df['Close'].pct_change() * 100
    
    return df

# Load the data
df = load_data()

# Configure default theme for all plots
chart_template = dict(
    layout=dict(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#e5e7eb'),
        xaxis=dict(
            gridcolor='#374151',
            linecolor='#374151',
            zerolinecolor='#374151'
        ),
        yaxis=dict(
            gridcolor='#374151',
            linecolor='#374151',
            zerolinecolor='#374151'
        )
    )
)

# Sidebar for date range selection
st.sidebar.header('Filter Data')

# Always use a list of two dates for date_input
default_dates = [df['Date'].min(), df['Date'].max()]
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=default_dates,
    min_value=df['Date'].min(),
    max_value=df['Date'].max(),
)

# Handle both single date and date range selections
if isinstance(date_range, (list, tuple)):
    # If it's a range or list of dates
    start_date = date_range[0]
    end_date = date_range[-1]  # Use -1 index to get last element whether it's a list of 1 or 2 dates
else:
    # If it's a single date
    start_date = end_date = date_range

# Filter data based on date range
mask = (df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))
filtered_df = df.loc[mask]

# State management for expanded views
if 'expanded_card' not in st.session_state:
    st.session_state.expanded_card = None

# Set dark theme for the entire app
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
            color: #fafafa;
        }        div.card-preview {
            background-color: transparent;
            border-radius: 15px;
            padding: 1rem;
            margin: 0.5rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        div.card-preview:hover {
            background-color: rgba(55, 65, 81, 0.3);
            transform: scale(1.01);
        }        div.fullscreen {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #0e1117;
            z-index: 1000;
            padding: 2rem;
            overflow-y: auto;
            backdrop-filter: blur(10px);
        }
        button.close-button {
            position: absolute;
            top: 1rem;
            right: 1rem;
            z-index: 1001;
        }
        div.preview-chart {
            height: 200px;
            margin-top: 1rem;
        }
        .stButton button {
            background-color: #3b82f6;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #2563eb;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        }
        div[data-testid="stMetricValue"] {
            color: #60a5fa;
            font-weight: 600;
        }
        div[data-testid="stMetricDelta"] {
            color: #34d399;
        }
        div[data-testid="stHeader"] {
            background-color: transparent;
        }
        section[data-testid="stSidebar"] {
            background-color: #1f2937;
        }
        div[data-testid="stMarkdownContainer"] {
            color: #e5e7eb;
        }
    </style>
""", unsafe_allow_html=True)

# Layout for preview cards
if st.session_state.expanded_card is None:
    col1, col2 = st.columns(2)

    with col1:
        # Price Analysis Preview
        st.markdown('<div class="card-preview">', unsafe_allow_html=True)
        st.subheader("📈 Price Analysis")        
        fig = go.Figure(go.Candlestick(
            x=filtered_df['Date'].tail(30),
            open=filtered_df['Open'].tail(30),
            high=filtered_df['High'].tail(30),
            low=filtered_df['Low'].tail(30),
            close=filtered_df['Close'].tail(30),
            increasing=dict(line=dict(color='#34d399')),  # Bright green for up
            decreasing=dict(line=dict(color='#fb7185'))   # Soft red for down
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            **chart_template['layout'],
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("View Full Analysis", key="price_btn"):
            st.session_state.expanded_card = "price"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Volume Analysis Preview
        st.markdown('<div class="card-preview">', unsafe_allow_html=True)
        st.subheader("📊 Volume Analysis")        
        fig = go.Figure(go.Bar(
            x=filtered_df['Date'].tail(30),
            y=filtered_df['Shares Traded'].tail(30),
            marker_color='#60a5fa'  # Soft blue
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            **chart_template['layout'],
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("View Full Analysis", key="volume_btn"):
            st.session_state.expanded_card = "volume"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Returns Analysis Preview
        st.markdown('<div class="card-preview">', unsafe_allow_html=True)
        st.subheader("📉 Returns Analysis")        
        fig = go.Figure(go.Histogram(
            x=filtered_df['Daily_Return'].tail(30),
            nbinsx=20,
            marker_color='#818cf8'  # Soft indigo
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            **chart_template['layout'],
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("View Full Analysis", key="returns_btn"):
            st.session_state.expanded_card = "returns"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Forecasting Preview
        st.markdown('<div class="card-preview">', unsafe_allow_html=True)
        st.subheader("🔮 Price Forecasting")        
        fig = go.Figure(go.Scatter(
            x=filtered_df['Date'].tail(30),
            y=filtered_df['Close'].tail(30),
            line=dict(color='#8b5cf6')  # Purple for forecast preview
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=0, b=0),
            **chart_template['layout'],
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        if st.button("View Full Analysis", key="forecast_btn"):
            st.session_state.expanded_card = "forecast"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

else:
    # Full screen views    st.markdown('<div class="fullscreen">', unsafe_allow_html=True)
    if st.button("Close", key="close"):
        st.session_state.expanded_card = None
        st.rerun()

    if st.session_state.expanded_card == "price":
        st.header("Price Analysis", divider="rainbow")
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.03,
                           row_heights=[0.7, 0.3])

        fig.add_trace(go.Candlestick(x=filtered_df['Date'],
                                    open=filtered_df['Open'],
                                    high=filtered_df['High'],
                                    low=filtered_df['Low'],
                                    close=filtered_df['Close'],
                                    name='OHLC'),
                    row=1, col=1)

        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA20'],
                                name='20-day MA', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA50'],
                                name='50-day MA', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['MA200'],
                                name='200-day MA', line=dict(color='red')), row=1, col=1)

        fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Shares Traded'],
                            name='Volume', marker_color='rgb(158,202,225)'),
                    row=2, col=1)
        fig.update_layout(
            title_text="NIFTY 50 Price Movement with Moving Averages",
            xaxis_rangeslider_visible=False,
            height=800,
            margin=dict(l=50, r=50, t=50, b=50),
            **chart_template['layout'],
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.expanded_card == "volume":
        st.header("Volume Analysis", divider="rainbow")
        # Create subplots with titles
        fig = make_subplots(rows=2, cols=1, 
                          shared_xaxes=True,
                          subplot_titles=('Trading Volume', 'Turnover (₹ Cr)'),
                          vertical_spacing=0.15)

        # Add volume trace
        fig.add_trace(go.Bar(x=filtered_df['Date'], 
                           y=filtered_df['Shares Traded'],
                           name='Volume',
                           marker_color='#60a5fa'), 
                    row=1, col=1)

        # Add turnover trace
        fig.add_trace(go.Bar(x=filtered_df['Date'], 
                           y=filtered_df['Turnover (₹ Cr)'],
                           name='Turnover',
                           marker_color='#60a5fa'), 
                    row=2, col=1)

        # Update layout with proper styling
        fig.update_layout(
            height=800,
            showlegend=False,
            **chart_template['layout']
        )

        # Update subplot title colors
        for annotation in fig.layout.annotations:
            annotation.update(font=dict(color='#e5e7eb', size=14))

        st.plotly_chart(fig, use_container_width=True)

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg. Daily Volume", f"{filtered_df['Shares Traded'].mean():,.0f}")
        with col2:
            st.metric("Max Volume", f"{filtered_df['Shares Traded'].max():,.0f}")
        with col3:
            st.metric("Avg. Daily Turnover", f"₹{filtered_df['Turnover (₹ Cr)'].mean():,.2f} Cr")

    elif st.session_state.expanded_card == "returns":
        st.header("Returns Analysis", divider="rainbow")
        fig = go.Figure()        
        fig.add_trace(go.Histogram(
            x=filtered_df['Daily_Return'],
            name='Daily Returns',
            nbinsx=50,
            marker_color='#818cf8'  # Soft indigo
        ))
        fig.update_layout(
            title='Distribution of Daily Returns',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            height=600,
            **chart_template['layout'],
            bargap=0.1,
            title_font=dict(color='#e5e7eb'),
            title_x=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Daily Return", f"{filtered_df['Daily_Return'].mean():.2f}%")
        with col2:
            st.metric("Return Volatility", f"{filtered_df['Daily_Return'].std():.2f}%")
        with col3:
            total_return = ((filtered_df['Close'].iloc[-1] / filtered_df['Close'].iloc[0]) - 1) * 100
            st.metric("Total Period Return", f"{total_return:.2f}%")

    elif st.session_state.expanded_card == "forecast":
        st.header("Price Forecasting", divider="rainbow")
        forecast_days = st.slider("Select number of days to forecast", 7, 90, 30)
        
        prophet_df = filtered_df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        
        future_dates = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future_dates)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'],
                                name='Historical', mode='lines',
                                line=dict(color='blue')))
        
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                name='Forecast', mode='lines',
                                line=dict(color='red')))
        
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                fill=None, mode='lines', line_color='rgba(255,0,0,0.2)',
                                name='Upper Bound'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                fill='tonexty', mode='lines', line_color='rgba(255,0,0,0.2)',
                                name='Lower Bound'))
        fig.update_layout(
            title=f'NIFTY 50 Price Forecast - Next {forecast_days} Days',
            xaxis_title='Date',
            yaxis_title='Price',
            showlegend=True,
            height=600,
            **chart_template['layout'],
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1,
                font=dict(color='#e5e7eb')
            ),
            title_font=dict(color='#e5e7eb'),
            title_x=0.5
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Forecast Components")
        fig_comp = model.plot_components(forecast)
        st.pyplot(fig_comp)

    st.markdown('</div>', unsafe_allow_html=True)
