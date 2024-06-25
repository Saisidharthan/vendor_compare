import streamlit as st
from streamlit_option_menu import option_menu
from product_page import display_product_page
from frontend import display_chat_page
# Main application logic
def main():
    st.set_page_config(layout="wide")
    with st.sidebar:
        selected = option_menu(
        menu_title=None,
        options=["Products", "Chat"],
        icons=["bag", "chat"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
    )
    # Session state to manage current page
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Products'
    if selected.lower() == "products":
        st.session_state.current_page = 'Products'
    elif selected.lower() == "chat":
        st.session_state.current_page = 'Chat'
    # Display content based on selected page
    if st.session_state.current_page == 'Products':
        display_product_page()
    elif st.session_state.current_page == 'Chat':
        display_chat_page()
# Run the main function to start the app
if __name__ == "__main__":
    main()