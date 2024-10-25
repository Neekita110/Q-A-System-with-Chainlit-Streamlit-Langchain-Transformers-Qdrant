import streamlit as st
import requests

# Streamlit UI
st.title("Document Analysis with Ollama")

# Upload document
uploaded_file = st.file_uploader("Upload Document", type=['txt', 'pdf'])

if uploaded_file is not None:
    # Display uploaded document
    st.write("Uploaded Document:")
    st.write(uploaded_file.getvalue())

    # Query vector database and display response
    if st.button("Analyze Document"):
        # Send document to Ollama API for analysis
        files = {'file': uploaded_file.getvalue()}
        ollama_url = 'http://localhost:8000/analyze'  # Replace with your Ollama API URL
        response = requests.post(ollama_url, files=files)

        if response.status_code == 200:
            analysis_result = response.json()
            st.write("Analysis Result:")
            st.write(analysis_result)
        else:
            st.error("Failed to analyze document. Please try again.")
