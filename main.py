import streamlit as st

st.set_page_config(layout="wide")
# Define pages with their titles and icons
def create_page(file_path, title, icon, default):
    return st.Page(file_path, title=title, icon=icon, default=default)

def main():
    # Create pages
    home_page = create_page("directory/eda.py", "Data Collection", icon=":material/data_check:", default=True)
    page_1 = create_page("directory/regression.py", "Regression Analysis", icon=":material/search_insights:", default=False) 
    page_2 = create_page("directory/pattern.py", "Pattern Analysis", icon=":material/monitoring:", default=False) 
    about_page = create_page("directory/about.py", "About", icon=":material/account_circle:", default=False)  

    # Set up navigation
    pages = {
        "About": [about_page],
        "Analysis": [home_page, page_1, page_2],
    }

    # Run the navigation
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()
