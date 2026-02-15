import streamlit as st
import pandas as pd
import numpy as np
from utils.text_extractor import TextExtractor as te
from utils.detector import RedFlagDetector as re

# Set Page Settings
st.set_page_config(
    page_title = "Cupid's Therapist",
    page_icon = "💘",
    layout = "wide",
    initial_sidebar_state= "expanded"
)

# Inject Custom CSS Styling
st.markdown(
    """
<style>
    .stProgress > div > div > div > div { background-color: #ff4b4b; }
    .red-flag {
        color: #ff4b4b; font-size: 40px; text-align: center; padding: 20px;
        border-radius: 10px; border: 2px solid #ff4b4b;
        background-color: rgba(255, 75, 75, 0.1);
    }
    .safe-flag {
        color: #00cc66; font-size: 40px; text-align: center; padding: 20px;
        border-radius: 10px; border: 2px solid #00cc66;
        background-color: rgba(0, 204, 102, 0.1);
    }
    .metric-card {
        background-color: #f0f2f6; border-radius: 10px; padding: 15px;
        text-align: center; margin: 5px;
    }
</style>
""",
    unsafe_allow_html = True
)

# Add title, header & threshold
st.title('Cupid\'s Therapist 💘')
st.header("AI-Powered Dating App Red Flag Detector", divider="red")

# Initialize Chat Message Container
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
                st.image(message["content"]) # images

# Get threshold through slider 
threshold = st.sidebar.slider(
    label = "Threshold",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.3
)

# ---- INPUTS ----
# Retrieve user input (prompt is a dictionary)
prompt = st.chat_input(
    "Enter Texts (text/screenshot)", 
    max_chars=500, 
    max_upload_size=5, 
    accept_file="multiple", 
    file_type=["jpg", "jpeg", "png"],
    width="stretch"
)

combined_text = ""
# Display user input in chat message container
if prompt:
    # HANDLE TEXT INPUT
    if prompt.text:
        # Display Text in Chat Container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt.text)
                combined_text += prompt.text  
        # Add Text to History
        st.session_state.messages.append({"role": "user", "type": "text", "content": prompt.text})

    # HANDLE IMAGE INPUT
    if prompt.files:
        # Display Image in Chat Container
        with chat_container:
            with st.chat_message("user"):
                for image in prompt.files:
                    st.image(image) 

                    # Extract text & append it to the prompt
                    text = te.extract_text_from_image(image)
                    combined_text += f" {text}"   

        # Add Image to History
        st.session_state.messages.append({"role": "user", "type": "image", "content": prompt["files"]})

    # Get average red flag score and results
    print(f"prompt {combined_text}")
    results_df, is_red_flag = re.get_results(combined_text, threshold)
    results_df= pd.DataFrame(results_df, columns=["Flag", "Scores"])


    if is_red_flag:
        response = "RED FLAGGGGGG 🚩🚩🚩🚩🚩🚩"   # Generate response to be implemented
    else:
        response = "Average score was under the threshold! ✅"

    # Display assistant response in chat message container
    with chat_container:
        with st.chat_message("assistant"):
            st.markdown(response)

    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "type": "text", "content": response})

    # Display Results
    with st.sidebar:
        st.divider()
        st.header("Results")
        results_container = st.container() 
        results_table = st.table(results_df)