import streamlit as st
import pandas as pd

def display_product_page():
    #st.set_page_config(layout="wide")
    st.markdown('<h1 style="color: black; font-weight: bold;">Vendor Products</h1>', unsafe_allow_html=True)
    # Load the CSV file
    file_path = r'C:\Users\sasee\OneDrive\Desktop\folder\sap.csv'  # Replace with the correct path to your CSV file
    df = pd.read_csv(file_path).iloc[:50,:]
    
    # Custom CSS for box layout and styling
    st.markdown("""
    <style>
    .stApp {
        background-color:lightblue;
    }
    .stButton>button {
        width: 100%;
        background-color:black;
        color:white;
        margin-top: 10px;
        border: 1px solid #333;
        border-radius: 5px;
    }
    .st-emotion-cache-1v0mbdj {
        width: 100%;
        padding: 0 1rem;
    }
    .full-width-container {
        width: 100%;
        max-width: none;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a 2-column layout with more space between columns
    for i in range(0, len(df), 2):
        col1, space, col2 = st.columns([1, 0.1, 1])
        
        with col1:
            if i < len(df):
                display_product(df.iloc[i])
        
        with col2:
            if i + 1 < len(df):
                display_product(df.iloc[i + 1])
        
        st.markdown("<br>", unsafe_allow_html=True)  # Add extra space between rows

def display_product(product):
    with st.container():
        st.markdown('<div class="product-box">', unsafe_allow_html=True)
        
        st.markdown(f'<h2 style="color:black;">{product["title"]}</h2>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: black;">{product["Vendor"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: black;"><strong>Price</strong>: {product["price"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: black;"><strong>Weight</strong>: {product["weight"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: black;"><strong>Dimensions</strong>: {product["dimension"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color: black;"><strong>Rating</strong>: {product["rating"]} out of 5 ({product["review"]} ratings reviews)</div>', unsafe_allow_html=True)
        if st.button("Add to Cart", key=f"add_{product.name}"):
            st.write(f"{product['title']} added to cart!")
        st.markdown('</div>', unsafe_allow_html=True)