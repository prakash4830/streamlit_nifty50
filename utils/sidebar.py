# utils/sidebar.py

import streamlit as st
import pandas as pd
from datetime import date

def display_copyright():
    """Displays the copyright information."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <footer style='text-align: center; margin-top: 20px;'>
            <p>&copy; 2024 JP Pvt Ltd. All rights reserved.</p>
        </footer>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    """
    Renders a shared sidebar for selecting start and end dates.
    Stores the selected years in Streamlit's session state.

    Returns:
        tuple: (start_year, end_year)
    """
    st.sidebar.header("Select Date Range")

    # Initialize session state variables if they don't exist
    if 'start_date' not in st.session_state:
        st.session_state.start_date = '2007-09-18'
    if 'end_date' not in st.session_state:
        st.session_state.end_date = date.today()

    # Select Start Date
    start_date = st.sidebar.date_input(
        "Start Date",
        pd.to_datetime(st.session_state.start_date),
        min_value=pd.to_datetime('2007-09-18'),
        max_value=pd.to_datetime('today')
    )

    # Select End Date
    end_date = st.sidebar.date_input(
        "End Date",
        pd.to_datetime(st.session_state.end_date),
        min_value=start_date,
        max_value=pd.to_datetime('today')
    )

    # Ensure that start_date is not greater than end_date
    if start_date > end_date:
        st.sidebar.error("Start Date must be earlier than or equal to End Date.")
    else:
        # Update session state
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date


# Display selected years
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Selected Start Date:** {start_date}")
    st.sidebar.write(f"**Selected End Date:** {end_date}")

# Store the years in session state for accessibility across pages
    st.session_state.start_year = start_date
    st.session_state.end_year = end_date

    return start_date, end_date
