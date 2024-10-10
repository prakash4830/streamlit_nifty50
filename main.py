import streamlit as st

# Define pages with their titles and icons
def create_page(file_path, title, icon):
    return st.Page(file_path, title=title, icon=icon)

def main():
    # Create pages
    home_page = create_page("directory/regression.py", "Regression", "📈")  # Chart emoji
    page_1 = create_page("directory/pattern.py", "Pattern Analysis", "📊")  # Bar chart emoji
    about_page = create_page("directory/about.py", "About", "ℹ️")  # Information emoji

    # Set up navigation
    pages = {
        "About": [about_page],
        "Analysis": [home_page, page_1],
    }

    # Run the navigation
    pg = st.navigation(pages)
    pg.run()

if __name__ == "__main__":
    main()
