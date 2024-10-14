import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils.sidebar import render_sidebar, display_copyright
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


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

def run_regression_analysis():
    # App Title and Description
    st.title(":material/search_insights: Nifty 50 Regression Analysis")
    st.markdown("""
    In this page we performs regression analysis for Nifty 50 index to understand the relationship between these 
    independent variables indicators and the Nifty 50's annual log returns.
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
    # fetching DataFrame in session
    # ---------------------------
    if 'org_data_eda' in st.session_state:
        data = st.session_state['org_data_eda']
        st.success("DataFrame successfully fetched from previous page!")
    else:
        st.error("No DataFrame found. Please check the date range and try again🙂.")
    # ---------------------------
    # Define Independent and Dependent Variables
    # ---------------------------
    st.header("1. Define Variables for Regression")

    if not data.empty:
        y = data['Nifty50_Annual_Mean_Log_Returns']
        X = data[['Inflation Rate (%)', 'GDP Growth Rate (%)', 'exchange_rate_Annual_Mean_Log_Returns', 'SP500_Annual_Mean_Log_Returns']]

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
    st.header("2. Multicollinearity Check using VIF")

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
        
        **Constant has a VIF of 13.8922:**
        A high VIF value for the constant indicates potential multicollinearity involving the intercept term, which is 
        expected in many models. 
        
        **Inflation Rate has a VIF of 1.0869:**
        VIF values below 5 generally indicate no concerning multicollinearity. A value of 2.5571 shows some 
        correlation with other variables but it’s still acceptable.
        
        **GDP Growth Rate (%) has a VIF of 1.0527:**
        This value is very close to 1, indicating almost no multicollinearity. This variable is independent of others 
        in the model.
        
        **Exchange Rate has a VIF of 1.5849:**
        This is another value below 5, indicating acceptable levels of multicollinearity. It shows some correlation 
        with other predictors but is not at a level that requires action.
        
        **SP500_Log_Returns has a VIF of 1.5183:**
        This VIF value is also very low, suggesting little to no multicollinearity for this variable.
        
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
    st.header("3. Regression Analysis")

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
        # st.subheader("Actual vs Predicted Annual Log Returns (Nifty 50)")
        fig_scatter = px.scatter(x=y, y=y_pred,
                                 labels={'x': 'Actual Annual Log Returns', 'y': 'Predicted Annual Log Returns'},
                                 title='Actual vs Predicted Annual Log Returns (Nifty 50)')
        fig_scatter.add_shape(
            type="line",
            x0=y.min(), y0=y.min(),
            x1=y.max(), y1=y.max(),
            line=dict(color='Black', dash='dash')
        )
        X = data[['GDP Growth Rate (%)', 'SP500_Annual_Mean_Log_Returns']]
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        # st.text(ridge.score(X, y))
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
        st.markdown(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.markdown(f"**R² Score:** {r2:.4f}")

        with st.expander("**Interpretation for regression Analysis**"):
            st.markdown(f"""
        **Model Fit:**
        - **R-squared({r2:.4f}):** This means that approximately {r2*100:.2f} of the variance in the Nifty50 Log Returns 
        is explained by the independent variables in the model. This indicates a strong model.
        - **Adjusted R-squared(0.759):** This adjusts for the number of predictors in the model, and suggests that about 
        75.9% of the variance is explained when accounting for the number of predictors.
        - **F-statistic Prob(0.000403):** The F-test checks whether the overall regression model is a good fit 
        for the data. Since the p-value (0.000403) is below 0.05, the overall model is statistically significant.

        **Coefficients:**
        - **constant (-0.0309):** The constant or intercept is -0.0309, meaning that when all the independent variables 
        are zero, the Nifty50 Log Returns would be approximately -0.0309.
        ***The p-value (0.654)*** indicates that the constant is not statistically significant.

        - **Inflation Rate (0.0007):** For each 1-unit increase in Inflation Rate, the Nifty50 Log Returns increase by 
        0.0007, holding all other variables constant.
        ***The p-value (0.924)*** is higher than 0.05, indicating that the Inflation Rate is not statistically 
        significant in this model.

        - **GDP Growth Rate (0.0149):** For each 1% increase in GDP Growth Rate, the Nifty50 Log Returns increase by 
        0.0149, holding other variables constant.
         ***The p-value (0.018)*** is lower than 0.05, indicates that GDP Growth Rate is statistically significant.

        - **Exchange Rate (-0.6677):** A 1-unit increase in Exchange Rate results in a decrease in Nifty50 Log Returns 
        by -0.6677, holding other variables constant.
        ***The p-value (0.185)*** is higher than 0.05, indicates that this variable is not statistically significant 
        either.

        - **SP500 Log Returns (0.6652):** A 1-unit increase in SP500 Log Returns leads to a 0.6652 increase in Nifty50 
        Log Returns.
         ***The p-value (0.002)*** is less than 0.05, indicating that this variable is **statistically significant** in 
        predicting Nifty50 Log Returns.
        
        **Mean Squared Error (MSE)= {mse:.4f} :**
        - An MSE of {mse:.4f} indicates that, on average, the squared difference between the predicted and actual values is
         quite small, suggesting that the model's predictions are relatively accurate.

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
            st.markdown(f"""
        - It is used to test Auto correlation in residual
        - DW < 1.5 (Autocorrelation)
        - DW>2.5 (Negative)
        - DW=0 (Strong Positive)
        - DW>4 (Strongly Negative)
        - Here DW= {dw_stat:.4f} which is greater than 1.5, so there is no Autocorrelation.
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
            st.markdown(f"""
        - The Breusch-Pagan test is used to detect heteroskedasticity (non-constant variance of the residuals) in a 
        regression model.
        - **LM Statistic= {bp_test[0]:.4f}:** 
        This is the value of the Lagrange Multiplier (LM) test statistic. It helps assess the presence of 
        heteroskedasticity, but its p-value is more informative for interpretation.
        - **LM-Test p-value= {bp_test[1]:.4f}:**
        The p-value corresponding to the LM Statistic is {bp_test[0]:.4f}, which is greater than the typical 
        significance level of 0.05. This means ***we fail to reject the null hypothesis***. The null hypothesis for the 
        Breusch-Pagan test is that homoskedasticity is present (constant variance of residuals). Since the p-value is 
        greater than 0.05, there is no significant evidence of heteroskedasticity.

        - **F-Statistic= {bp_test[2]:.4f}:**
        The F-statistic also tests for heteroskedasticity. However, its interpretation is similar to the LM test. We 
        primarily focus on the p-value associated with this test.
        - **F-Test p-value= {bp_test[3]:.4f}:**
        Similar to the LM-Test p-value, the F-Test p-value ({bp_test[3]:.4f}) is greater than 0.05. Therefore, we fail 
        to reject the null hypothesis and conclude that there is no significant evidence of heteroskedasticity in the 
        model.

        - **Conclusion:**
        - Both the LM-Test and F-Test p-values are above the 0.05 threshold, meaning there is ***no significant 
        heteroskedasticity in your regression model***.
        We can proceed without concern about heteroskedasticity, as the assumption of constant variance in the 
        residuals appears to hold.
        """)

        # ---------------------------
        # Polynomial Regression with Ridge Regularization
        # ---------------------------
        if False:
            '''
        st.subheader("4. Polynomial Regression with Ridge Regularization")
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
        '''
        pass

        # ---------------------------
        # Final Regression Model with Lagged Variables
        # ---------------------------
        st.subheader("4. Final Regression Model with Lagged Variables")
        with st.expander("View Lagged Regression Model Details"):
            # Example of incorporating lagged variables (lagged by 1 year)
            data['Inflation Rate (%)_Lag1'] = data['Inflation Rate (%)'].shift(1)
            data['GDP Growth Rate (%)_Lag1'] = data['GDP Growth Rate (%)'].shift(1)
            data['Exchange_Rate_Lag1'] = data['exchange_rate_Annual_Mean_Log_Returns'].shift(1)
            data['SP500_Log_Returns_Lag1'] = data['SP500_Annual_Mean_Log_Returns'].shift(1)

            # Drop the first row with NaN values
            data_lagged = data.dropna().reset_index(drop=True)
            # st.write(data_lagged)
            if not data_lagged.empty:
                # Define independent variables with lagged terms
                X_lagged = data_lagged[
                    ['Inflation Rate (%)_Lag1', 'GDP Growth Rate (%)_Lag1', 'Exchange_Rate_Lag1',
                     'SP500_Log_Returns_Lag1']]
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
    st.header("6. Correlation Analysis")

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
        - **1.Nifty50 and Exchange Rates**
            - Nifty50 Annual Mean Close has a strong positive correlation (0.92) with the Exchange Rate Annual Mean Close, 
            suggesting that changes in exchange rates significantly impact the performance of Nifty 50. When the value of 
            the rupee weakens, Nifty 50 tends to show an upward movement, possibly due to better export performance or foreign 
            investments. The Nifty50 Log Returns show a moderate positive correlation (0.33) with the Exchange Rate Mean Close, 
            implying that short-term returns are also influenced by exchange rate fluctuations, but to a lesser extent than the 
            annual closing values.

        - **2.Inflation Rate and Nifty 50:**
            - The Inflation Rate has a negative correlation (-0.58) with the Nifty50 Annual Mean Close, indicating that higher 
            inflation tends to depress stock prices. This could be attributed to the fact that inflation erodes purchasing power 
            and leads to tighter monetary policy, which can negatively affect stock market performance. The Inflation Rate (Lagged 
            by 1 Year) also has a negative correlation (-0.55) with the Nifty50 Annual Mean Close, confirming that inflation impacts 
            market performance over a longer horizon as well.

        - **3.GDP Growth Rate:**
            - The GDP Growth Rate shows a moderate positive correlation (0.48) with Nifty50 Log Returns, indicating that higher economic 
            growth generally leads to better stock market returns. This highlights the importance of economic growth in driving investor 
            confidence and stock market performance. However, the correlation between the GDP Growth Rate and Nifty50 Annual Mean Close 
            is weaker, indicating that the overall stock market index might not be as sensitive to short-term changes in GDP growth as 
            the returns are.

        - **4.S&P 500 and Nifty 50:**
            - The S&P500 Annual Mean Close shows a very high positive correlation (0.98) with the Nifty50 Annual Mean Close, signifying 
            that the Nifty 50 follows global market trends, particularly the U.S. stock market. This is a strong indication of the 
            interconnectedness of the Indian and U.S. economies, where changes in the U.S. market tend to drive corresponding changes in 
            the Indian market. Similarly, S&P500 Log Returns also show a strong positive correlation (0.79) with Nifty50 Log Returns, 
            reinforcing the link between short-term market movements in the U.S. and India.

        - **5. Exchange Rate Lag and Inflation Lag:**
            - The Exchange Rate Lag shows a positive correlation (0.29) with the Nifty50 Log Returns, which implies that the previous 
            year's exchange rate has a mild influence on this year's Nifty 50 returns. The Inflation Rate Lag also shows a negative 
            correlation (-0.55) with the Nifty 50, highlighting the lasting impact of inflation over time.

        """)
    else:
        st.warning("Skipped Correlation Matrix and Heatmap due to empty DataFrame.")


    # ---------------------------
    # End of Application
    # ---------------------------


run_regression_analysis()
