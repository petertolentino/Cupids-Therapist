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

# How it works header
with st.sidebar:
    st.subheader("How it works!\n1. Pick a sensitivity\n2. Copy and paste texts or upload a screenshot(s) 📲\n3. Wait for the result!")
    
# Get threshold through slider 
threshold = st.sidebar.slider(
    label = "Overall Sensitivity (Threshold )",
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
        verdict = "🚩 Red Flag" if is_red_flag else "✅ Looks Safe" 
        st.markdown(f"**Verdict:** {verdict}")

        st.subheader("Confidence 🎯")
        confidence = float(results_df["Scores"].max()) if not results_df.empty else 0.0
        st.metric("Highest flag score", f"{confidence:.2f}")
        
        st.header("Results")
        results_container = st.container()
        results_table = st.table(results_df)
        # --- Nice feature: Top-3 flags with progress bars ---
        st.subheader("Top Flags 🔥")

        # If results_df exists and has rows
        if not results_df.empty:
            top3 = results_df.sort_values("Scores", ascending=False).head(3)

            for _, row in top3.iterrows():
                flag = row["Flag"]
                score = float(row["Scores"])

                st.markdown(f"**{flag}** — `{score:.2f}`")
                
        else:
            st.caption("No flags detected yet.")
