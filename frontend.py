import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
import torch
import re
import textwrap
from the_final_sap_vscode_1 import query_system
# Set page configuration and background image

def display_chat_page():
    #st.set_page_config(page_title="Chat To Know About The Vendors Of Bags", page_icon=":pouch:")
    st.markdown(f"""<style>.main {{
            background-image: url("https://static.vecteezy.com/system/resources/previews/010/803/399/non_2x/background-with-colorful-shopping-bags-illustration-sale-and-discount-concept-vector.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;}}
        .stHeader > h2 {{color: black !important;}}
        .stTextInput > label {{color: black !important;}}
        </style>""", unsafe_allow_html=True)
    
    st.header("Chat to know about vendors :handbag:")
    user_query = st.text_input("Ask questions to get to know about the best vendor")
    answer_text = st.empty()  # Create an empty container for the answer

    if user_query:
        answer, _ = query_system(user_query)  # Call backend function to get answer
        answer_text.text_area("Answer:", value=answer, height=100)  # Display the answer in a text area

    # Sidebar with options
    st.sidebar.markdown(f"""<style>
        [data-testid="stSidebar"] > div:first-child {{
            background-image: url("https://images.pexels.com/photos/2905238/pexels-photo-2905238.jpeg");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;}}</style>""", unsafe_allow_html=True)

    options = ["👜 Handbags", "🧳 Travel Bags", "💼 Sleeve", "👜 Tote bag", "🎒 Trekking bags", "💻 Laptop bags"]
    st.sidebar.selectbox("What kind of bags would you like to search?", options)