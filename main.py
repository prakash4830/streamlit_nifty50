import streamlit as st

st.set_page_config(layout="wide")
# Define pages with their titles and icons
def create_page(file_path, title, icon):
    return st.Page(file_path, title=title, icon=icon)

def main():
    # Create pages
    home_page = create_page("directory/eda.py", "Data Collection", icon=":material/data_check:")
    page_1 = create_page("directory/regression.py", "Regression Analysis", icon=":material/search_insights:")  # Chart emoji
    page_2 = create_page("directory/pattern.py", "Pattern Analysis", icon=":material/monitoring:")  # Bar chart emoji
    about_page = create_page("directory/about.py", "About", icon=":material/account_circle:")  # Information emoji

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
