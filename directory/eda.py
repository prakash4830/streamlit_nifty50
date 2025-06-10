import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.sidebar import render_sidebar, display_copyright
import requests
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller, kpss

# ---------------------------
# Data Fetching Functions
# ---------------------------

@st.cache_data
def fetch_gdp_growth(start_year, end_year):
    """
    Fetches India's GDP Growth Rate (%) from the World Bank API for the specified year range.

    Parameters:
    - start_year (int): The starting year for data retrieval.
    - end_year (int): The ending year for data retrieval.

    Returns:
    - pd.DataFrame: A DataFrame containing the Year and GDP Growth Rate (%).
    """
    url = (
        f"https://api.worldbank.org/v2/country/IN/indicator/NY.GDP.MKTP.KD.ZG"
        f"?format=json&date={start_year}:{end_year}&per_page=100"
    )

    response = requests.get(url)
    gdp_ind = pd.DataFrame()

    if response.status_code == 200:
        try:
            data = response.json()
            if len(data) < 2:
                st.warning("No data found in the response for GDP Growth Rate.")
            else:
                gdp_data = data[1]  # The second element contains the actual data
                gdp_ind = pd.DataFrame(gdp_data)

                # Select relevant columns
                gdp_ind = gdp_ind[['date', 'value']]
                gdp_ind.columns = ['Year', 'GDP Growth Rate (%)']

                # Convert Year to integer and GDP Growth Rate to float
                gdp_ind['Year'] = pd.to_numeric(gdp_ind['Year'], errors='coerce').astype('Int64')
                gdp_ind['GDP Growth Rate (%)'] = pd.to_numeric(gdp_ind['GDP Growth Rate (%)'], errors='coerce')

                # Drop rows with missing values
                gdp_ind.dropna(inplace=True)

                # Sort the DataFrame by Year in ascending order
                gdp_ind = gdp_ind.sort_values(by='Year', ascending=True).reset_index(drop=True)

                # Filter the DataFrame for years >= start_year
                gdp_ind = gdp_ind[gdp_ind['Year'] >= start_year].reset_index(drop=True)

        except (ValueError, KeyError, IndexError) as e:
            st.error(f"Error processing the GDP data: {e}")
    else:
        st.error(f"Failed to retrieve GDP data. HTTP Status Code: {response.status_code}")

    return gdp_ind


@st.cache_data
def fetch_economic_data(start_year, end_year):
    """
    Fetches India's Inflation Rate (%) from the World Bank API for the specified year range.

    Parameters:
    - start_year (int): The starting year for data retrieval.
    - end_year (int): The ending year for data retrieval.

    Returns:
    - pd.DataFrame: A DataFrame containing the Year and Inflation Rate (%).
    """
    # Define the World Bank API endpoint
    indicator_code = 'FP.CPI.TOTL.ZG'  # Inflation rate (Consumer Prices)
    country_code = 'IND'  # India
    per_page = 100  # Number of records per page (adjust if needed)

    # Construct the API URL
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
        f"?date={start_year}:{end_year}&format=json&per_page={per_page}"
    )

    # Make the API request
    response = requests.get(url)
    economic_df = pd.DataFrame()

    if response.status_code == 200:
        try:
            data = response.json()
            if len(data) < 2:
                st.warning("No data found in the response for Inflation Rate.")
            else:
                records = data[1]  # The second element contains the data
                # Convert records to DataFrame
                economic_df = pd.DataFrame(records)
                # Select relevant columns
                economic_df = economic_df[['date', 'value']]
                economic_df.columns = ['Year', 'Inflation Rate (%)']
                # Convert Year to integer and Inflation Rate to float
                economic_df['Year'] = pd.to_numeric(economic_df['Year'], errors='coerce').astype('Int64')
                economic_df['Inflation Rate (%)'] = pd.to_numeric(economic_df['Inflation Rate (%)'], errors='coerce')
                # Drop rows with missing values
                economic_df.dropna(inplace=True)
                # Sort by Year ascending
                economic_df = economic_df.sort_values(by='Year').reset_index(drop=True)
        except (ValueError, KeyError, IndexError) as e:
            st.error(f"Error processing the Inflation Rate data: {e}")
    else:
        st.error(f"Failed to retrieve Inflation Rate data. HTTP Status Code: {response.status_code}")

    return economic_df


@st.cache_data
def fetch_exchange_rate(start_year, end_year):
    """
    Fetches INR to USD exchange rate data from Yahoo Finance.

    Parameters:
    - start_year (int): The starting year for data retrieval.
    - end_year (int): The ending year for data retrieval.

    Returns:
    - pd.DataFrame: Exchange rate data.
    """
    exchange_rate = yf.download('USDINR=X', start=start_year, end=end_year, progress=False)
    return exchange_rate


@st.cache_data
def fetch_stock_data(ticker, start_date, end_date, retries=3):
    """
    Fetches stock data using yfinance with retry logic and handles MultiIndex columns.

    Parameters:
    - ticker (str): Stock ticker symbol (e.g., '^NSEI' for NIFTY 50).
    - start_date (str or datetime): Start date in 'YYYY-MM-DD' format.
    - end_date (str or datetime): End date in 'YYYY-MM-DD' format.
    - retries (int): Number of retry attempts.

    Returns:
    - pd.DataFrame: Cleaned stock data with flattened columns.
    """
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                raise ValueError(f"No data returned for {ticker} from {start_date} to {end_date}")

            # Reset index to make 'Date' a column
            data.reset_index(inplace=True)

            # Convert 'Date' column to datetime format
            data['Date'] = pd.to_datetime(data['Date'])

            # Handle MultiIndex columns (e.g., after corporate actions or dividends)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]
                # Optional: simplify names (take only the first part)
                data.columns = [col.split('_')[0] for col in data.columns]

            # Set 'Date' back as index if needed
            data.set_index('Date', inplace=True)

            return data

        except Exception as e:
            print(f"[Attempt {attempt+1}/{retries}] Error: {e}")
            if attempt < retries - 1:
                print("Retrying in 60 seconds...")
                time.sleep(60)
            else:
                raise RuntimeError(f"Failed to fetch data for {ticker} after {retries} attempts.")

def calculate_vif(X):
    """
    Calculate Variance Inflation Factor for each feature.
    """
    vif = pd.DataFrame()
    vif["Feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif


def perform_breusch_pagan_test(model, X, y):
    """
    Perform Breusch-Pagan test for heteroscedasticity.
    """
    residuals = y - model.predict(X)
    lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(residuals, X)
    return {'LM Statistic': lm, 'LM-Test p-value': lm_pvalue, 'F-Statistic': fvalue, 'F-Test p-value': f_pvalue}

def non_stationary_plot(data):
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
        go.Scatter(x=data.index, y=data['Close'], name='Non-Stationary', line=dict(color='blue')),
        secondary_y=False,
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        legend=dict(x=0, y=1.1, orientation="h"),
        template="plotly_white",
        height=600
    )

    fig.update_yaxes(title_text="Price", secondary_y=False, color='blue')

    return fig

def stationary_plot(data):
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
        go.Scatter(x=data.index, y=data['Log_Returns'], name='stationary', line=dict(color='orange')),
        secondary_y=False,
    )
    # Update layout
    fig.update_layout(
        xaxis_title="Date",
        legend=dict(x=0, y=1.1, orientation="h"),
        template="plotly_white",
        height=600
    )

    fig.update_yaxes(title_text="Price", secondary_y=False, color='blue')

    return fig

def adf_interpret(adf, adf_critical_values):
    # Interpret ADF Test
    if adf[1] < 0.05:
        st.success(f"ADF Test: The series is stationary (reject the null hypothesis), because the "
                   f"***p-value:*** {adf[1]} < 0.05.")
    else:
        st.warning(f"ADF Test: The series is non-stationary (fail to reject the null hypothesis), because the "
                   f"***p-value:*** {adf[1]} > 0.05.")
    if adf[0] < adf_critical_values['5%']:
        st.success("ADF Test: The series is stationary (reject the null hypothesis), because the "
                   "***ADF statistic value*** is lesser than the ***Critical Values***.")
    else:
        st.warning("ADF Test: The series is non-stationary (fail to reject the null hypothesis),  because the "
                   "***ADF statistic value*** is greater than the ***Critical Values***.")

def kpss_interpret(kps, kpss_critical_values):
    # Interpret KPSS Test
    if kps[1] > 0.05:
        st.success(f"KPSS Test: The series is stationary (fail to reject the null hypothesis), because the "
                   f"***p-value:*** {kps[1]} > 0.05.")
    else:
        st.warning(f"KPSS Test: The series is non-stationary (reject the null hypothesis), because the "
                   f"***p-value:*** {kps[1]} < 0.05.")
    # KPSS Interpretation
    if kps[0] > kpss_critical_values['5%']:
        st.warning("KPSS Test: The series is non-stationary (reject the null hypothesis), because the "
                   "***KPSS statistic value*** is greater than the ***Critical Values***.")
    else:
        st.success("KPSS Test: The series is stationary (fail to reject the null hypothesis), because the "
                   "***KPSS statistic value*** is lesser than the ***Critical Values***.")

def check_stationary(data, chart_name):
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data.dropna(inplace=True)
    st.header(chart_name + " Non- Stationary Chart")
    price_vol_fig = non_stationary_plot(data)
    st.plotly_chart(price_vol_fig, use_container_width=True)

    st.header((chart_name + " Stationary Chart"))
    price_vol_fig = stationary_plot(data)
    st.plotly_chart(price_vol_fig, use_container_width=True)

    adf = adfuller(data['Log_Returns'])
    adf_statistic = adf[0]
    adf_p_value = adf[1]
    adf_critical_values = adf[4]

    st.subheader("Augmented Dickey-Fuller (ADF) Test")
    # Display ADF Test Results
    st.markdown(f"**ADF Statistic:** {adf_statistic:.4f}")
    st.markdown(f"**p-value:** {adf_p_value:.4f}")

    # Create DataFrame for ADF Critical Values
    adf_crit_df = pd.DataFrame({
        "Critical Value": [f"{v:.4f}" for v in adf_critical_values.values()]
    }, index=adf_critical_values.keys())
    st.table(adf_crit_df)
    adf_interpret(adf, adf_critical_values)

    kps = kpss(data['Log_Returns'], regression='c')
    kpss_statistic = kps[0]
    kpss_p_value = kps[1]
    kpss_critical_values = kps[3]
    st.subheader("Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test")
    # Display KPSS Test Results
    st.markdown(f"**KPSS Statistic:** {kpss_statistic:.4f}")
    st.markdown(f"**p-value:** {kpss_p_value:.4f}")

    # Create DataFrame for KPSS Critical Values
    kpss_crit_df = pd.DataFrame({
        "Critical Value": [f"{v:.4f}" for v in kpss_critical_values.values()]
    }, index=kpss_critical_values.keys())
    st.table(kpss_crit_df)
    kpss_interpret(kps, kpss_critical_values)


def run_eda():
    # App Title and Description
    st.title(":material/data_check: Data Collection, preprocessing and EDA")
    st.markdown("""
    This page fetches and analyzes economic indicators such as GDP Growth Rate, Inflation Rate, Exchange Rates,
     and Stock Market Indices (Nifty 50 and S&P 500) from Yahoo Finance and World Bank API. Then we are converting 
     the Stock Market Indices data from non-stationary to stationary by using ***log*** and perform the Exploratory 
     data analysis (EDA).
    """)

    # ---------------------------
    # Sidebar for User Inputs
    # ---------------------------
    start_date, end_date = render_sidebar()

    end_year = end_date.year
    start_year = start_date.year

    exchange_rate_final = pd.DataFrame()
    nifty_50_final = pd.DataFrame()
    sp500_final = pd.DataFrame()
    data = pd.DataFrame()
    data_lagged = pd.DataFrame()

    display_copyright()
    # ---------------------------
    # Fetch and Display GDP Data
    # ---------------------------
    st.header("1. GDP Growth Rate for India")
    gdp_ind = fetch_gdp_growth(start_year=start_year, end_year=end_year)
    if not gdp_ind.empty:
        st.dataframe(gdp_ind)
    else:
        st.warning("GDP Growth Rate data is empty.")

    # ---------------------------
    # Fetch and Display Inflation Rate Data
    # ---------------------------
    st.header("2. Inflation Rate for India")
    try:
        inflation_ind = fetch_economic_data(start_year=start_year, end_year=end_year)
        if not inflation_ind.empty:
            st.dataframe(inflation_ind)
        else:
            st.warning("Inflation Rate data is empty.")
    except Exception as e:
        st.error(f"An error occurred while fetching economic data: {e}")

    # ---------------------------
    # Fetch and Display Exchange Rate Data from Yahoo Finance
    # ---------------------------
    st.header("3. INR to USD Exchange Rate")

    exchange_rate = fetch_stock_data('USDINR=X',start_date=start_date, end_date=end_date)
    if not exchange_rate.empty:
        st.subheader("Exchange Rate Data Sample")
        st.dataframe(exchange_rate)

        check_stationary(exchange_rate, chart_name = 'Exchange Rate')

        exchange_rate_mean = exchange_rate['Close'].resample('YE').mean().dropna()

        # Convert index to Year
        exchange_rate_mean = exchange_rate_mean.reset_index()
        exchange_rate_mean['Year'] = exchange_rate_mean['Date'].dt.year

        # Select relevant columns and rename
        exchange_rate_mean = exchange_rate_mean[['Year', 'Close']]
        exchange_rate_mean.rename(columns={'Close': 'exchange_rate_Annual_Mean_Close'}, inplace=True)

        # Calculate Log Returns
        exchange_rate_mean['Log_Returns'] = np.log(
            exchange_rate_mean['exchange_rate_Annual_Mean_Close'] /
            exchange_rate_mean['exchange_rate_Annual_Mean_Close'].shift(1))
        exchange_rate_mean.dropna(inplace=True)

        # Rename Log_Returns column for clarity
        exchange_rate_mean.rename(columns={'Log_Returns': 'exchange_rate_Annual_Mean_Log_Returns'}, inplace=True)

        # Display the DataFrame
        st.subheader("Exchange_rate")
        st.dataframe(exchange_rate_mean)
    else:
        st.warning("Exchange rate data is empty.")

    # ---------------------------
    # Fetch and Display Stock Data from Yahoo Finance
    # ---------------------------
    st.header("4. Stock Market Data")

    # Nifty 50
    st.subheader("Nifty 50 (^NSEI)")
    nifty50 = fetch_stock_data('^NSEI', start_date=start_date, end_date=end_date)
    if not nifty50.empty:
        st.dataframe(nifty50)

    else:
        st.warning("Nifty 50 data is empty.")
    check_stationary(nifty50, chart_name = 'Nifty 50')

    # S&P 500
    st.subheader("S&P 500 (^GSPC)")
    sp500 = fetch_stock_data('^GSPC', start_date=start_date, end_date=end_date)
    if not sp500.empty:
        st.dataframe(sp500)
    else:
        st.warning("S&P 500 data is empty.")

    check_stationary(sp500, chart_name = 'S&P 500')

    # ---------------------------
    # Check the data was non-stationary or stationary
    # ---------------------------

    # ---------------------------
    # Calculate Annual Mean Closing Prices for Nifty 50
    # ---------------------------
    st.header("5. Annual Mean Closing Prices and Log Returns Calculation")

    if not nifty50.empty:
        # Resample to annual frequency and calculate mean closing price
        nifty50_annual_mean = nifty50['Close'].resample('YE').mean().dropna()

        # Convert index to Year
        nifty50_annual_mean = nifty50_annual_mean.reset_index()
        nifty50_annual_mean['Year'] = nifty50_annual_mean['Date'].dt.year

        # Select relevant columns and rename
        nifty50_annual_mean = nifty50_annual_mean[['Year', 'Close']]
        nifty50_annual_mean.rename(columns={'Close': 'Nifty50_Annual_Mean_Close'}, inplace=True)

        # Calculate Log Returns
        nifty50_annual_mean['Log_Returns'] = np.log(
            nifty50_annual_mean['Nifty50_Annual_Mean_Close'] / nifty50_annual_mean['Nifty50_Annual_Mean_Close'].shift(
                1))
        nifty50_annual_mean.dropna(inplace=True)

        # Rename Log_Returns column for clarity
        nifty50_annual_mean.rename(columns={'Log_Returns': 'Nifty50_Annual_Mean_Log_Returns'}, inplace=True)

        # Display the DataFrame
        st.subheader("Nifty 50")
        st.dataframe(nifty50_annual_mean)
    else:
        st.warning("Cannot calculate Nifty 50 annual mean closing prices due to empty data.")

    # ---------------------------
    # Calculate Annual Mean Closing Prices for S&P 500
    # ---------------------------
    # st.header("6. Annual Mean Closing Prices and Log Returns Calculation for S&P 500")

    if not sp500.empty:
        # Resample to annual frequency and calculate mean closing price
        sp500_annual_mean = sp500['Close'].resample('YE').mean().dropna()

        # Convert index to Year
        sp500_annual_mean = sp500_annual_mean.reset_index()
        sp500_annual_mean['Year'] = sp500_annual_mean['Date'].dt.year

        # Select relevant columns and rename
        sp500_annual_mean = sp500_annual_mean[['Year', 'Close']]
        sp500_annual_mean.rename(columns={'Close': 'SP500_Annual_Mean_Close'}, inplace=True)

        # Calculate Log Returns
        sp500_annual_mean['Log_Returns'] = np.log(
            sp500_annual_mean['SP500_Annual_Mean_Close'] / sp500_annual_mean['SP500_Annual_Mean_Close'].shift(1))
        sp500_annual_mean.dropna(inplace=True)

        # Rename Log_Returns column for clarity
        sp500_annual_mean.rename(columns={'Log_Returns': 'SP500_Annual_Mean_Log_Returns'}, inplace=True)

        # Display the DataFrame
        st.subheader("S&P 500")
        st.dataframe(sp500_annual_mean)
    else:
        st.warning("Cannot calculate S&P 500 annual mean closing prices due to empty data.")

    # ---------------------------
    # Merge All Data into a Single DataFrame
    # ---------------------------
    st.header("7. Merge All Data into a Single DataFrame")

    if not (
            nifty50_annual_mean.empty or sp500_annual_mean.empty or inflation_ind.empty or gdp_ind.empty or
            exchange_rate_mean.empty):
        # Merge Nifty 50 annual mean data
        data = nifty50_annual_mean.merge(inflation_ind, on='Year', how='inner')

        # Merge GDP Growth Rate
        data = data.merge(gdp_ind, on='Year', how='inner')

        # Merge Exchange Rate
        data = data.merge(exchange_rate_mean, on='Year', how='inner')

        # Merge S&P 500 annual mean data
        data = data.merge(sp500_annual_mean, on='Year', how='inner')

        # Display the merged DataFrame
        st.dataframe(data)
    else:
        st.warning("Merged DataFrame is empty. Please check the individual datasets.")

    # ---------------------------
    # Check if DataFrame is Empty
    # ---------------------------
    if 'data' in locals() and not data.empty:
        st.success(f"Number of observations after merging: {len(data)}")
    else:
        st.error("The merged DataFrame is empty. Check data frequencies and overlapping years.")

    # ---------------------------
    # storing DataFrame in session
    # ---------------------------
    st.session_state['org_data_eda'] = data

    # Correlation Matrix
    st.header("8. Exploratory Data Analysis (EDA)")

    if 'data' in locals() and not data.empty:
        # Compute the correlation matrix
        corr_matrix = data.iloc[:, 1:].corr()

        st.subheader("Correlation Matrix")
        st.dataframe(corr_matrix)

        # Create Heatmap using Plotly
        fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='reds', zmin=-1,
                                zmax=1, title='Correlation Matrix of Economic Indicators and Nifty 50 Log Returns')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        with st.expander("**Interpretation for Correlation Analysis**"):
            st.markdown("""
        **1.Market Sensitivity to Global Factors:** The Nifty 50 has a strong dependence on global markets, especially 
        the U.S. market (S&P 500). Both short-term returns and long-term trends in the Nifty 50 follow the movements in 
        the S&P 500 closely, indicating that macroeconomic events in the U.S. significantly affect the Indian stock market.

        **2.Role of Exchange Rates:** The Nifty 50's performance is heavily influenced by exchange rate movements. The positive 
        correlation between the exchange rate and Nifty 50 suggests that foreign exchange inflows and outflows could be key 
        drivers of market trends.

        **3.Inflation's Negative Impact:** High inflation, both current and lagged, negatively impacts the Nifty 50. This could 
        reflect tightening monetary policies and a reduction in domestic consumption, making inflation an important risk factor 
        for stock market investors in India.

        **4.Economic Growth and Stock Returns:** GDP growth has a stronger impact on Nifty 50 returns than on the absolute market 
        index value, suggesting that investors respond more to changes in growth rates in terms of stock returns rather than absolute 
        levels of the market index.

        """)
    else:
        st.warning("Cannot perform EDA on an empty DataFrame.")


run_eda()
