import streamlit as st

def setup_sidebar():
    """
    Function to set up a consistent sidebar across all pages.
    """
    # Apply custom CSS
    st.markdown("""
        <style>
            /* Sidebar Background */
            [data-testid="stSidebar"] {
                background-color: #2b4c7e !important;
            }

            /* Sidebar Text Styling */
            [data-testid="stSidebar"] * {
                color: #f5f7fa !important;
            }

            /* Sidebar Title Styling */
            [data-testid="stSidebar"] h1 {
                color: #ffffff !important;
                font-size: 20px !important;
                font-weight: bold !important;
                margin-top: 20px !important;
            }

            /* Logo Alignment */
            [data-testid="stSidebar"] img {
                display: block;
                margin: 0 auto 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar Logo and Title
    st.sidebar.image("logo.png", use_container_width=True)
    st.sidebar.title("🔍 CO₂ Emission Analysis")
    st.sidebar.markdown("---")

   
