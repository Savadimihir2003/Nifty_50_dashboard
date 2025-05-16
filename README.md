# NIFTY 50 Analysis Dashboard

An interactive dashboard for analyzing NIFTY 50 index performance with forecasting capabilities.

## Features

- Historical price analysis with candlestick charts
- Multiple moving averages (20, 50, 200-day)
- Volume and turnover analysis
- Daily returns distribution
- Price forecasting using Facebook Prophet
- Interactive date range selection
- Detailed statistics and metrics

## Installation

1. Create a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```powershell
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501` in your web browser.

## Dashboard Sections

1. **Price Analysis**
   - Candlestick chart with OHLC data
   - Moving averages (20, 50, 200-day)
   - Volume subplot

2. **Volume Analysis**
   - Daily trading volume visualization
   - Turnover analysis
   - Key volume statistics

3. **Returns Analysis**
   - Daily returns distribution
   - Return statistics
   - Performance metrics

4. **Forecasting**
   - Prophet-based price predictions
   - Adjustable forecast period
   - Confidence intervals
   - Seasonality components

## Data

The dashboard uses historical NIFTY 50 data from May 2024 to May 2025, including:
- Open, High, Low, Close prices
- Trading volume
- Turnover in ₹ Crores
