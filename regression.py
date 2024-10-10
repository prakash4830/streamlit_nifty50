# pages/1_NIFTY50_regression_Analysis.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.sidebar import render_sidebar, display_copyright
import requests
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------------------
# Helper Functions for Plots
# ---------------------------

def make_decomposition_plot(decomposition):
    """
    Create a Plotly figure for seasonal decomposition.

    Parameters:
    - decomposition: The result of seasonal_decompose.

    Returns:
    - Plotly Figure
    """
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))

    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name="Observed"), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name="Trend"), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name="Seasonal"), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name="Residual"), row=4, col=1)

    fig.update_layout(height=800, showlegend=False, title_text="Seasonal Decomposition of Nifty 50 Log Returns")
    return fig


def make_acf_pacf_plot(acf_vals, pacf_vals, lags=20):
    """
    Create a Plotly figure for ACF and PACF.

    Parameters:
    - acf_vals: Autocorrelation function values.
    - pacf_vals: Partial autocorrelation function values.
    - lags: Number of lags to display.

    Returns:
    - Plotly Figure
    """
    fig = make_subplots(rows=2, cols=1, subplot_titles=("ACF", "PACF"))

    fig.add_trace(go.Bar(x=list(range(len(acf_vals[:lags]))), y=acf_vals[:lags], name='ACF'), row=1, col=1)
    fig.add_trace(go.Bar(x=list(range(len(pacf_vals[:lags]))), y=pacf_vals[:lags], name='PACF'), row=2, col=1)

    fig.update_layout(height=600, showlegend=False, title_text="Autocorrelation and Partial Autocorrelation")
    return fig


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
    exchange_rate = yf.download('INRUSD=X', start=start_year, end=end_year, progress=False)
    return exchange_rate


@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches stock data from Yahoo Finance.

    Parameters:
    - ticker (str): Stock ticker symbol.
    - start_date (datetime.date): Start date for data retrieval.
    - end_date (datetime.date): End date for data retrieval.

    Returns:
    - pd.DataFrame: Stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data

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



# ---------------------------
# Main Streamlit App
# ---------------------------

def run_regression_analysis(poly_degree=None,interaction_only=None, ridge_alpha=None):
    # App Title and Description
    st.title("Economic Indicators and Nifty 50 Analysis")
    st.markdown("""
    This Streamlit app fetches and analyzes economic indicators such as GDP Growth Rate, Inflation Rate, Exchange Rates, and
    Stock Market Indices (Nifty 50 and S&P 500). It performs regression analysis to understand the relationship between 
    these indicators and the Nifty 50's annual log returns.
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

    exchange_rate = fetch_exchange_rate(start_year=start_date, end_year=end_date)
    if not exchange_rate.empty:
        st.subheader("Exchange Rate Data Sample")
        st.dataframe(exchange_rate)

        # Calculate Annual Average Exchange Rate
        exchange_rate_annual = exchange_rate['Close'].resample('Y').mean().dropna()
        exchange_rate_annual = exchange_rate_annual.to_frame(name='Exchange_Rate').reset_index()

        # Extract Year
        exchange_rate_annual['Year'] = exchange_rate_annual['Date'].dt.year
        exchange_rate_final = exchange_rate_annual[['Year', 'Exchange_Rate']]

        st.subheader("Annual Average Exchange Rate (INR per USD)")
        st.dataframe(exchange_rate_final)
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

    # S&P 500
    st.subheader("S&P 500 (^GSPC)")
    sp500 = fetch_stock_data('^GSPC', start_date=start_date, end_date=end_date)
    if not sp500.empty:
        st.dataframe(sp500)
    else:
        st.warning("S&P 500 data is empty.")

    # ---------------------------
    # Calculate Annual Log Returns for Nifty 50 and S&P 500
    # ---------------------------
    # ---------------------------
    # Calculate Annual Mean Closing Prices for Nifty 50
    # ---------------------------
    st.header("5. Annual Mean Closing Prices and Log Returns Calculation")

    if not nifty50.empty:
        # Resample to annual frequency and calculate mean closing price
        nifty50_annual_mean = nifty50['Close'].resample('Y').mean().dropna()

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
        sp500_annual_mean = sp500['Close'].resample('Y').mean().dropna()

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
            exchange_rate_final.empty):
        # Merge Nifty 50 annual mean data
        data = nifty50_annual_mean.merge(inflation_ind, on='Year', how='inner')

        # Merge GDP Growth Rate
        data = data.merge(gdp_ind, on='Year', how='inner')

        # Merge Exchange Rate
        data = data.merge(exchange_rate_final, on='Year', how='inner')

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
        **Interpretation:**
        - Helps to identify relationships between variables and quantify their strength and direction.
        - In this exchange rate lag1 and exchange rate has highest correlation of value 0.9577

        """)
    else:
        st.warning("Cannot perform EDA on an empty DataFrame.")

    # ---------------------------
    # Define Independent and Dependent Variables
    # ---------------------------
    st.header("9. Define Variables for Regression")

    if not data.empty:
        y = data['Nifty50_Annual_Mean_Log_Returns']
        X = data[['Inflation Rate (%)', 'GDP Growth Rate (%)', 'Exchange_Rate', 'SP500_Annual_Mean_Log_Returns']]

        st.subheader("Independent Variables (X)")
        st.dataframe(X)

        st.subheader("Dependent Variable (y)")
        st.dataframe(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        st.warning("Cannot define X and y due to empty DataFrame.")

    # ---------------------------
    # Check for Multicollinearity using VIF
    # ---------------------------
    st.header("10. Multicollinearity Check using VIF")

    if 'data' in locals() and not data.empty:
        # Define independent variables
        # X = data[['Inflation Rate', 'GDP Growth Rate (%)', 'Exchange_Rate', 'SP500_Annual_Mean_Log_Returns']]

        # Add constant for VIF calculation
        X_const = sm.add_constant(X)

        # Calculate VIF for each feature
        vif = pd.DataFrame()
        vif['Variable'] = X_const.columns
        vif['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

        st.table(vif)
        with st.expander("**Intrepretation for Multicollinearity**"):
        # Interpretation
            st.markdown("""
        - Table of VIF values: Indicates the degree of multicollinearity between independent variables. VIF values above 
        5-10 suggest problematic multicollinearity.
        - If all VIF values are low (close to 1), there is little multicollinearity. Higher values would 
        suggest that variables like inflation or exchange rates are highly interrelated, complicating the interpretation 
        of regression coefficients. 
        **Constant has a VIF of 35.5799:**
        - A high VIF value for the constant indicates potential multicollinearity involving the intercept term, which is 
        expected in many models. 
        **Inflation Rate has a VIF of 2.5571:**
        - VIF values below 5 generally indicate no concerning multicollinearity. A value of 2.5571 shows some 
        correlation with other variables but it’s still acceptable.
        **GDP Growth Rate (%) has a VIF of 1.0313:**
        - This value is very close to 1, indicating almost no multicollinearity. This variable is independent of others 
        in the model.
        **Exchange Rate has a VIF of 2.9045:**
        - This is another value below 5, indicating acceptable levels of multicollinearity. It shows some correlation 
        with other predictors but is not at a level that requires action.
        **SP500_Log_Returns has a VIF of 1.3098:**
        -   This VIF value is also very low, suggesting little to no multicollinearity for this variable.
        - None of the variables (apart from the constant) have a VIF exceeding 5, so multicollinearity does not appear 
        to be a problem for your independent variables. High VIF values (typically above 10) would suggest that 
        multicollinearity is a concern and that some variables might need to be dropped or transformed, but that is not 
        the case here.
        """)
    else:
        st.warning("Skipped VIF calculation due to empty DataFrame.")


    # ---------------------------
    # Regression Analysis using statsmodels
    # ---------------------------
    st.header("11. Regression Analysis")

    if 'data' in locals() and not data.empty:
        # Add a constant term for the intercept
        X = sm.add_constant(X)

        # Fit the OLS model
        try:
            model = sm.OLS(y, X).fit()
        except Exception as e:
            st.error(f"Error fitting the OLS model: {e}")
            st.stop()

        # Make Predictions
        y_pred = model.predict(X)

        # Visualize the Results
        st.subheader("Actual vs Predicted Annual Log Returns (Nifty 50)")
        fig_scatter = px.scatter(x=y, y=y_pred,
                                 labels={'x': 'Actual Annual Log Returns', 'y': 'Predicted Annual Log Returns'},
                                 title='Actual vs Predicted Annual Log Returns (Nifty 50)')
        fig_scatter.add_shape(
            type="line",
            x0=y.min(), y0=y.min(),
            x1=y.max(), y1=y.max(),
            line=dict(color='Black', dash='dash')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        with st.expander("**Interpretation for Actual Vs predicted values**"):
            st.markdown("""
        - A linear relationship between actual and predicted is shown, line would indicate that the model's predictions 
        are close to the actual values. The points scattered around this line suggest the accuracy and variance of 
        predictions. 
        - If the points closely align with the diagonal line, the model is performing well in predicting Nifty 
        50 returns. The scatter and spread reflect the deviation or errors in predictions. 
        - Here most of the points are around the line only few is away from straight line, it shows the model is good 
        fit, there is no that much difference between actual and predicted value.
        """)
        # Print the Regression Summary
        st.subheader("Regression Summary")
        st.text(model.summary())

        # Evaluate the Model
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        st.markdown(f"**Mean Squared Error (MSE):** {mse:.6f}")
        st.markdown(f"**R² Score:** {r2:.6f}")

        with st.expander("**Interpretation for regression Analysis**"):
            st.markdown("""
        **Regression Summary:**
        - The regression summary provides detailed information about the relationship between your independent variables 
        (Inflation Rate, GDP Growth Rate, Exchange Rate, SP500 Log Returns) and the dependent variable (Nifty50 Log 
        Returns). Here’s an interpretation of the results:

        **Model Fit:**
        - **R-squared: 0.695:** This means that approximately 69.5% of the variance in the Nifty50 Log Returns is 
        explained by the independent variables in the model. This indicates a moderately strong model.
        - **Adjusted R-squared: 0.584:** This adjusts for the number of predictors in the model, and suggests that about 
        58.4% of the variance is explained when accounting for the number of predictors.
        - **F-statistic: 6.262 (Prob: 0.00705):** The F-test checks whether the overall regression model is a good fit 
        for the data. Since the p-value (0.00705) is below 0.05, the overall model is statistically significant.

        **Coefficients:**
        - **constant (0.2033):** The constant or intercept is 0.2033, meaning that when all the independent variables 
        are zero, the Nifty50 Log Returns would be approximately 0.2033.
        - **The p-value (0.466)** indicates that the constant is not statistically significant.

        - **Inflation Rate (0.0337):** For each 1-unit increase in Inflation Rate, the Nifty50 Log Returns increase by 
        0.0337, holding all other variables constant.
        - **The p-value (0.249)** is higher than 0.05, indicating that the Inflation Rate is not statistically 
        significant in this model.

        - **GDP Growth Rate (0.0132):** For each 1% increase in GDP Growth Rate, the Nifty50 Log Returns increase by 
        0.0132, holding other variables constant.
        - **The p-value (0.343)** indicates that GDP Growth Rate is not statistically significant.

        - **Exchange Rate (-30.3980):** A 1-unit increase in Exchange Rate results in a decrease in Nifty50 Log Returns 
        by 30.3980, holding other variables constant.
        - **The p-value (0.204)** indicates that this variable is not statistically significant either.

        - **SP500 Log Returns (0.9252):** A 1-unit increase in SP500 Log Returns leads to a 0.9252 increase in Nifty50 
        Log Returns.
        - **The p-value (0.006)** is less than 0.05, indicating that this variable is *statistically significant* in 
        predicting Nifty50 Log Returns.

        """)
        # ---------------------------
        # Test for Seasonality
        # ---------------------------
        st.subheader("Seasonality Test (Seasonal Decomposition)")
        try:
            decomposition = seasonal_decompose(data['Nifty50_Annual_Mean_Log_Returns'], model='additive', period=1)
            fig_decomp = make_decomposition_plot(decomposition)
            st.plotly_chart(fig_decomp, use_container_width=True)
        except Exception as e:
            st.error(f"Error in seasonal decomposition: {e}")

        with st.expander("**Interpretation for Seasonality Test**"):
            st.markdown("""
        **Interpretation:**
        - Displays seasonal, trend, and residual components of Nifty 50 log returns. The trend component shows long-term
         movement, the seasonal component captures repeating patterns over a time frame, and residuals reflect the noise. 
        - Here the seasonal component exhibits consistent cycles, Nifty 50 log returns have a seasonal trend. 
        Fluctuations in the residual component show unexplained variance, here there is no fluctuation in residual. 
        Trend was initially peak then reduced and maintained constant high. There is no too much fluctuation in trend, 
        seasonal & residual remains constant over time.
        """)

        # ---------------------------
        # Test for Autocorrelation
        # ---------------------------
        st.subheader("Autocorrelation (ACF) and Partial Autocorrelation (PACF)")
        try:
            acf_values = acf(data['Nifty50_Annual_Mean_Log_Returns'])
            pacf_values = pacf(data['Nifty50_Annual_Mean_Log_Returns'])

            fig_acf_pacf = make_acf_pacf_plot(acf_values, pacf_values)
            st.plotly_chart(fig_acf_pacf, use_container_width=True)
        except Exception as e:
            st.error(f"Error in ACF/PACF calculation: {e}")

        with st.expander("**Interpretation for Auto-correlation**"):
            st.markdown("""
        **ACF Interpretation:**
        - The ACF plot measures the correlation between the time series and its lagged values.
           The blue bars represent the autocorrelations at different lag values, with the red line representing the 
           confidence interval (typically 95%).
        
        - At lag 1, the autocorrelation is positive and strong, as shown by a bar above 0.5. This indicates a 
        significant correlation between the series and its value one step before. Several other lags 
        (e.g., lag 2, lag 5) also have smaller significant correlations, as their values are beyond the confidence 
        interval.

        **PACF Interpretation:**
        - The PACF plot shows the partial autocorrelations, filtering out indirect correlations through intermediate 
        lags. It helps identify the direct effect of a lag on the series. Like the ACF plot, the confidence interval is 
        shown with the red line.
        - At lag 1, the PACF shows a significant positive value. This suggests that the most substantial direct 
        autocorrelation occurs at lag 1.
        - At lag 4 and lag 6, the PACF plot shows smaller significant spikes, indicating potential additional 
        correlations at these lags.
        - Autocorrelation is present, especially at lag 1, as indicated by both ACF and PACF plots. The ACF indicates 
        some serial dependence, while the PACF highlights the direct influence of lag 1 and a few smaller correlations 
        at later lags (lag 4 and lag 6). This suggests that the time series may benefit from an autoregressive (AR) 
        model to capture the relationship with its past values.

        """)

        # ---------------------------
        # Durbin-Watson Test for Autocorrelation
        # ---------------------------
        st.subheader("Durbin-Watson Test for Autocorrelation")
        dw_stat = sm.stats.stattools.durbin_watson(model.resid)
        st.markdown(f"**Durbin-Watson Statistic:** {dw_stat:.4f}")

        with st.expander("**Interpretation for Durbin-Watson Test**"):
            st.markdown("""
        - Durbin-Watson Test for Autocorrelation Durbin-Watson Statistic: 2.8282
        - It is used to test Auto correlation in residual
        - DW < 1.5 (Autocorrelation)
        - DW>2.5 (Negative)
        - DW=0 (Strong Positive)
        - DW>4 (Strongly Negative)
        - Here DW= 2.8282 which is greater than 2.5 so negative, there is no Autocorrelation.
        """)

        # ---------------------------
        # Test for Heteroskedasticity
        # ---------------------------
        st.subheader("Breusch-Pagan Test for Heteroskedasticity")
        bp_test = het_breuschpagan(model.resid, model.model.exog)
        labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
        bp_results = pd.DataFrame({
            'Test': labels,
            'Value': bp_test
        })
        st.table(bp_results)

        with st.expander("**Interpretation for Heteroskedasticity**"):
            st.markdown("""
        - The Breusch-Pagan test is used to detect heteroskedasticity (non-constant variance of the residuals) in a 
        regression model. Here’s an interpretation of the results:
        - **LM Statistic: 5.4592:**
        - This is the value of the Lagrange Multiplier (LM) test statistic. It helps assess the presence of 
        heteroskedasticity, but its p-value is more informative for interpretation.
        - **LM-Test p-value: 0.2433:**
        - The p-value corresponding to the LM Statistic is 0.2433, which is greater than the typical significance level 
        of 0.05. This means *we fail to reject the null hypothesis*. The null hypothesis for the Breusch-Pagan test is 
        that homoskedasticity is present (constant variance of residuals).
        - Since the p-value is greater than 0.05, there is no significant evidence of heteroskedasticity.

        - **F-Statistic: 1.4243:**
        - The F-statistic also tests for heteroskedasticity. However, its interpretation is similar to the LM test. We 
        primarily focus on the p-value associated with this test.
        - **F-Test p-value: 0.2897:**
        - Similar to the LM-Test p-value, the F-Test p-value (0.2897) is greater than 0.05. Therefore, we fail to reject 
        the null hypothesis and conclude that there is no significant evidence of heteroskedasticity in the model.

        - **Conclusion:**
        - Both the LM-Test and F-Test p-values are above the 0.05 threshold, meaning there is *no significant 
        heteroskedasticity in your regression model.
        - You can proceed without concern about heteroskedasticity, as the assumption of constant variance in the 
        residuals appears to hold.
        """)

        # ---------------------------
        # Polynomial Regression with Ridge Regularization
        # ---------------------------
        st.subheader("12. Polynomial Regression with Ridge Regularization")
        with st.expander("View Polynomial Ridge Regression Details"):
            # Define independent variables
            X_poly = data[['Inflation Rate (%)', 'GDP Growth Rate (%)', 'Exchange_Rate', 'SP500_Annual_Mean_Log_Returns']]

            # Create polynomial features
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly_transformed = poly.fit_transform(X_poly)

            # Define target variable
            y_poly = data['Nifty50_Annual_Mean_Log_Returns']

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_poly_transformed, y_poly, test_size=0.2,
                                                                random_state=42)

            # Fit Ridge Regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)

            # Predictions
            y_pred_poly = ridge.predict(X_test)

            # Evaluate
            mse_poly = mean_squared_error(y_test, y_pred_poly)
            r2_poly = r2_score(y_test, y_pred_poly)
            st.markdown(f"**Polynomial Ridge Regression Mean Squared Error:** {mse_poly:.6f}")
            st.markdown(f"**Polynomial Ridge Regression R² Score:** {r2_poly:.6f}")

            # Plot Actual vs Predicted
            fig_ridge = px.scatter(x=y_test, y=y_pred_poly, labels={'x': 'Actual', 'y': 'Predicted'},
                                   title='Polynomial Ridge Regression: Actual vs Predicted')
            fig_ridge.add_shape(
                type="line",
                x0=y_test.min(), y0=y_test.min(),
                x1=y_test.max(), y1=y_test.max(),
                line=dict(color='Red', dash='dash')
            )
            st.plotly_chart(fig_ridge, use_container_width=True)

        # ---------------------------
        # Final Regression Model with Lagged Variables
        # ---------------------------
        st.subheader("13. Final Regression Model with Lagged Variables")
        with st.expander("View Lagged Regression Model Details"):
            # Example of incorporating lagged variables (lagged by 1 year)
            data['Inflation Rate (%)_Lag1'] = data['Inflation Rate (%)'].shift(1)
            data['GDP Growth Rate (%)_Lag1'] = data['GDP Growth Rate (%)'].shift(1)
            data['Exchange_Rate_Lag1'] = data['Exchange_Rate'].shift(1)
            data['SP500_Log_Returns_Lag1'] = data['SP500_Annual_Mean_Log_Returns'].shift(1)

            # Drop the first row with NaN values
            data_lagged = data.dropna().reset_index(drop=True)
            # st.write(data_lagged)
            if not data_lagged.empty:
                # Define independent variables with lagged terms
                X_lagged = data_lagged[
                    ['Inflation Rate (%)_Lag1', 'GDP Growth Rate (%)_Lag1', 'Exchange_Rate_Lag1', 'SP500_Log_Returns_Lag1']]
                y_lagged = data_lagged['Nifty50_Annual_Mean_Log_Returns']

                # Add constant
                X_lagged = sm.add_constant(X_lagged)

                # Fit the OLS model
                model_lagged = sm.OLS(y_lagged, X_lagged).fit()

                # Predictions
                y_pred_lagged = model_lagged.predict(X_lagged)

                # Print the Regression Summary
                st.markdown("**Lagged Regression Summary**")
                st.text(model_lagged.summary())

                # Evaluate
                mse_lagged = mean_squared_error(y_lagged, y_pred_lagged)
                r2_lagged = r2_score(y_lagged, y_pred_lagged)
                st.markdown(f"**Lagged Mean Squared Error:** {mse_lagged:.6f}")
                st.markdown(f"**Lagged R² Score:** {r2_lagged:.6f}")

                # Visualize Lagged Model Results
                st.markdown("**Actual vs Predicted Annual Log Returns (Lagged Model)**")
                fig_lagged = px.scatter(x=y_lagged, y=y_pred_lagged,
                                        labels={'x': 'Actual Annual Log Returns', 'y': 'Predicted Annual Log Returns'},
                                        title='Actual vs Predicted Annual Log Returns (Nifty 50) - Lagged Model')
                fig_lagged.add_shape(
                    type="line",
                    x0=y_lagged.min(), y0=y_lagged.min(),
                    x1=y_lagged.max(), y1=y_lagged.max(),
                    line=dict(color='Black', dash='dash')
                )
                st.plotly_chart(fig_lagged, use_container_width=True)
            else:
                st.warning("Lagged data is empty after shifting. Cannot perform lagged regression.")

    # ---------------------------
    # Correlation Analysis
    # ---------------------------
    st.header("14. Correlation Analysis")

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
        **Interpretation:**
        - Helps to identify relationships between variables and quantify their strength and direction.
        - In this exchange rate lag1 and exchange rate has highest correlation of value 0.9577

        """)
    else:
        st.warning("Skipped Correlation Matrix and Heatmap due to empty DataFrame.")



    # ---------------------------
    # End of Application
    # ---------------------------

run_regression_analysis()
