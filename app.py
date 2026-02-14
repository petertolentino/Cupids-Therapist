import streamlit as st
import pandas as pd
import numpy as np
from utils.text_extractor import TextExtractor

# Add title & header
with st.container():
    st.title('Cupid\'s Therapist 💘')
    st.header("Show their texts to find out if they're a red flag 🚩🚩🚩", divider="red")

# Chat Message Container
chat_container = st.container(height=300, border=True)

# Initialize Message History 
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "text":   
                st.markdown(message["content"]) # texts
            elif message.get("type") == "image":
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
    # HANDLE TEXT INPUT
    if data.text:
        # Display Text in Chat Container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(data.text)  
        # Add Text to History
        st.session_state.messages.append({"role": "user", "type": "text", "content": data.text})

    # HANDLE IMAGE INPUT
    if data["files"]:
        # Display Image in Chat Container
        with chat_container:
            with st.chat_message("user"):
                for image in data["files"]:
                    st.markdown(image) 
        # Add Image to History
        st.session_state.messages.append({"role": "user", "type": "image", "content": data["files"]})
    
    # Generate a response
    response = f"Analyzing message: {data.text}" # function to be implemented for response
    
    # Display assistant response in chat message container
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(response)
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "type": "text", "content": response})