import streamlit as st
import pandas as pd
import numpy as np

# Add title & header
with st.container():
    st.title('Cupid\'s Therapist :cupid:')
    st.header("Show their texts to find out if they're a red flag :triangular_flag_on_post:", divider="red")

# Chat Message Container
chat_container = st.container(height=400, border=True)

# Initialize Message History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if type(message) == "text":   
                st.markdown(message["content"]) # texts
            elif type(message) == "image":
                st.image(message["content"])    # images

# ---- INPUTS ----

# Retrieve user input (data is a dictionary)
data = st.chat_input(
    "Enter Texts (text/screenshot)", 
    max_chars=500, 
    max_upload_size=5, 
    accept_file="multiple", 
    width=1000
)

# Display user input in chat message container
if data:
    if data.text:
        # HANDLE TEXT INPUT
        with chat_container:
            with st.chat_message("user"):
                st.markdown(data.text)  
        # Add user input to chat history (text)
        st.session_state.messages.append({"role": "user", "content": data.text})
    if data["files"]:
        # HANDLE IMAGE INPUT
        with chat_container:
            with st.chat_message("user"):
                st.image(data["files"]) 
        # Add user input to chat history (image)
        st.session_state.messages.append({"role": "user", "content": data["files"]})
    
    # Generate a response
    response = f"Analyzing message: {data.text}" # function to be implemented for response
    
    # Display assistant response in chat message container
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(response)
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})