import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.sidebar import render_sidebar, display_copyright
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------
# Helper Functions for Plots
# ---------------------------

def make_price_and_volatility_plot(data):
    """
    Creates a dual-axis Plotly chart for Close Price and Volatility.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Close' and 'Volatility' columns.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Close Price trace
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')),
        secondary_y=False,
    )

    # Volatility trace
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Volatility'], name='Volatility', line=dict(color='orange')),
        secondary_y=True,
    )

    # Update layout
    fig.update_layout(
        title="Nifty 50 Price and Volatility",
        xaxis_title="Date",
        legend=dict(x=0, y=1.1, orientation="h"),
        template="plotly_white",
        height=600
    )

    fig.update_yaxes(title_text="Price", secondary_y=False, color='blue')
    fig.update_yaxes(title_text="Volatility", secondary_y=True, color='orange')

    return fig


def make_price_patterns_plot(data):
    """
    Creates Plotly subplots for Close Price with Moving Averages and RSI.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Close', 'SMA50', 'SMA200', and 'RSI' columns.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Close Price with Moving Averages", "Relative Strength Index (RSI)"))

    # Close Price and Moving Averages
    fig.add_trace(
        go.Scatter(x=data.index, y=data['Close'], name='Close Price', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA50'], name='50-day SMA', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA200'], name='200-day SMA', line=dict(color='red')),
        row=1, col=1
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=2, col=1)

    # Update layout
    fig.update_layout(
        title="Nifty 50 Price Patterns",
        xaxis_title="Date",
        showlegend=True,
        template="plotly_white",
        height=800
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)

    return fig


def make_decomposition_plot(decomposition):
    """
    Creates Plotly subplots for seasonal decomposition components.

    Parameters:
        decomposition (statsmodels.tsa.seasonal.DecomposeResult): Result of seasonal decomposition.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure object.
    """
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
    )

    # Observed
    fig.add_trace(
        go.Scatter(
            x=decomposition.observed.index,
            y=decomposition.observed,
            mode='lines',
            name='Observed',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    # Trend
    fig.add_trace(
        go.Scatter(
            x=decomposition.trend.index,
            y=decomposition.trend,
            mode='lines',
            name='Trend',
            line=dict(color='green')
        ),
        row=2, col=1
    )

    # Seasonal
    fig.add_trace(
        go.Scatter(
            x=decomposition.seasonal.index,
            y=decomposition.seasonal,
            mode='lines',
            name='Seasonal',
            line=dict(color='orange')
        ),
        row=3, col=1
    )

    # Residual
    fig.add_trace(
        go.Scatter(
            x=decomposition.resid.index,
            y=decomposition.resid,
            mode='lines',
            name='Residual',
            line=dict(color='red')
        ),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        height=1200,
        width=900,
        title_text="Seasonal Decomposition of Nifty 50 Close Price",
        showlegend=False,
        template="plotly_white"
    )

    # Update y-axes titles
    fig.update_yaxes(title_text="Observed", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)

    return fig


# ---------------------------
# Data Fetching and Processing Functions
# ---------------------------

@st.cache_data
def fetch_nifty50_data(start_date, end_date):
    """
    Fetches historical data for Nifty 50 index from Yahoo Finance.

    Parameters:
        start_date (str): The start date for fetching data.
        end_date (str): The end date for fetching data.

    Returns:
        pd.DataFrame: Historical stock data.
    """
    nifty50 = yf.Ticker("^NSEI")
    data = nifty50.history(start=start_date, end=end_date)
    return data


def calculate_volatility(data, window=20):
    """
    Calculates the rolling volatility of the Nifty 50 index.

    Parameters:
        data (pd.DataFrame): DataFrame containing historical stock data.
        window (int): Rolling window size for volatility calculation.

    Returns:
        pd.DataFrame: DataFrame with added 'Returns' and 'Volatility' columns.
    """
    data = data.copy()
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=window).std() * np.sqrt(252)
    return data


def analyze_price_patterns(data):
    """
    Analyzes price patterns by calculating moving averages and RSI.

    Parameters:
        data (pd.DataFrame): DataFrame containing historical stock data.

    Returns:
        pd.DataFrame: DataFrame with added 'SMA50', 'SMA200', and 'RSI' columns.
    """
    data = data.copy()
    # Simple Moving Averages
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data


def perform_seasonal_decomposition(data):
    """
    Performs seasonal decomposition on the Close Price.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'Close' column.

    Returns:
        statsmodels.tsa.seasonal.DecomposeResult or None: Decomposition result or None if insufficient data.
    """
    # Ensure there are enough data points for decomposition
    if len(data) < 2 * 252:  # Approx. two years of trading days
        st.warning("Not enough data points for seasonal decomposition. Need at least two years of data.")
        return None

    try:
        decomposition = seasonal_decompose(data['Close'], model='multiplicative', period=252, extrapolate_trend='freq')
        return decomposition
    except Exception as e:
        st.error(f"Error during seasonal decomposition: {e}")
        return None


# ============================
# Start of Streamlit Application
# ============================

def pattern_analysis():
    # Page Title
    st.title(":material/monitoring: Nifty 50 Pattern Analysis")

    st.markdown("""
    In this page provides an analysis of the Nifty 50 index, including volatility, moving averages, RSI, and seasonal 
    decomposition. Use the sidebar to select the desired date range for the analysis.
    """)

    # ---------------------------
    # Sidebar for User Inputs
    # ---------------------------
    start_date, end_date = render_sidebar()

    end_year = end_date.year
    start_year = start_date.year
    display_copyright()

    # ---------------------------
    # Fetch Data
    # ---------------------------
    st.header("1. Fetching Nifty 50 Data")

    with st.spinner('Fetching Nifty 50 data...'):
        data = fetch_nifty50_data(start_date, end_date)

    if data.empty:
        st.error("No data fetched. Please check the date range and try again.")
        st.stop()

    st.success("Data fetched successfully!")

    # ---------------------------
    # Calculate Volatility
    # ---------------------------
    st.header("2. Calculating Volatility")

    data = calculate_volatility(data)
    if 'Volatility' in data.columns:
        st.success("Volatility calculated successfully!")
    else:
        st.error("Volatility calculation failed.")
        st.stop()

    # ---------------------------
    # Plot Price and Volatility
    # ---------------------------
    st.header("3. Nifty 50 Price and Volatility")
    price_vol_fig = make_price_and_volatility_plot(data)
    st.plotly_chart(price_vol_fig, use_container_width=True)

    with st.expander("**Interpretation for Volatality Test**"):
        st.markdown("""

    - The Nifty 50's closing price has shown a strong upward trend since 2008, with significant growth after 2015. 
    Despite short-term fluctuations, the index has moved from around 5,000 points to above 25,000 points.
    - **High Volatility During Two Phases:**
        - *2008-2009 Global Financial Crisis*
        - *2020 (COVID-19 Pandemic)*

    """)

    st.markdown("---")  # Separator

    # ---------------------------
    # Analyze Price Patterns
    # ---------------------------
    st.header("4. Analyzing Price Patterns")

    data = analyze_price_patterns(data)
    if 'SMA50' in data.columns and 'SMA200' in data.columns and 'RSI' in data.columns:
        st.success("Price patterns analyzed successfully!")
    else:
        st.error("Price pattern analysis failed.")
        st.stop()

    # ---------------------------
    # Plot Price Patterns
    # ---------------------------
    st.header("5. Price Patterns: Moving Averages and RSI")
    price_patterns_fig = make_price_patterns_plot(data)
    st.plotly_chart(price_patterns_fig, use_container_width=True)
    with st.expander("**Interpretation for MA and RSI Test**"):
        st.markdown("""
    **Moving Average:**
    - **Close Price (Blue):** Represents the actual closing price of the Nifty 50 index over time, showing a 
    strong upward trend.
    - **50-Day Simple Moving Average (SMA) (Green):** This smooths out short-term price fluctuations, offering 
    insight into the intermediate trend.
    - **200-day Simple Moving Average (SMA) (RED)** remains below the price and the 50-day SMA, indicating 
    a strong long-term uptrend.

    **RSI:**
    - **RSI (Purple):** RSI is a momentum indicator that ranges from 0 to 100, used to measure overbought or 
    oversold conditions.    
    - **Above 70 (Red Line):** Typically considered overbought, indicating the market might be due for a correction.
    - **Below 30 (Green Line):** Typically considered oversold, suggesting a potential buying opportunity.
    
    **Interpretation:**
    - *Recent Period:* The RSI hovers near the overbought level, reflecting the strong market momentum as the Nifty 50 
    reaches new highs. However, this might also signal a possible correction soon.
    - The RSI suggests that the market has experienced periods of overbought conditions recently, implying potential 
    short-term corrections or pauses in the upward movement.
    """)

    st.markdown("---")  # Separator

    # ---------------------------
    # Seasonal Decomposition
    # ---------------------------
    st.header("6. Seasonal Decomposition of Nifty 50 Close Price")
    decomposition = perform_seasonal_decomposition(data)
    if decomposition:
        decomp_fig = make_decomposition_plot(decomposition)
        st.plotly_chart(decomp_fig, use_container_width=True)
    else:
        st.warning("Seasonal decomposition could not be performed due to insufficient data or errors.")

    with st.expander("**Interpretation for Seasonality Test**"):
        st.markdown("""
    - **Pattern:** It highlights the strong upward trend of the Nifty 50, with some dips during major market corrections.
    - **Trend:** The trend is consistently upward, with an accelerating rise after 2020. The sharp rise after 2020 
    suggests strong market recovery and growth momentum.
    - **Seasonal:** The values oscillate between ~0.98 and 1.02, indicating small but consistent seasonal effects. 
    The peaks and troughs occur at regular intervals, showing a cyclical pattern likely tied to market cycles like 
    fiscal years, earnings seasons, or economic reports.
    - **Residual:** Significant residuals are observed around 2008-2009 and 2020.
    
    This Seasonal decomposition helps highlight that the long-term trend drives the Nifty 50’s performance, while 
    seasonality and irregularities play more minor roles.

    """)

    st.markdown("---")  # Separator

    # ---------------------------
    # Additional Insights (Optional)
    # ---------------------------
    st.header("7. Additional Insights")
    with st.expander("View Additional Insights"):
        # Example: Highlighting RSI Overbought/Oversold Conditions
        oversold = data[data['RSI'] < 30]
        overbought = data[data['RSI'] > 70]
        col1, col2 = st.columns(2, gap="small")
        with col1:
            st.subheader("Oversold Conditions (RSI < 30)")
            if not oversold.empty:
                st.dataframe(oversold[['Close', 'RSI']])
            else:
                st.write("No oversold conditions detected.")
        with col2:
            st.subheader("Overbought Conditions (RSI > 70)")
            if not overbought.empty:
                st.dataframe(overbought[['Close', 'RSI']])
            else:
                st.write("No overbought conditions detected.")

pattern_analysis()
