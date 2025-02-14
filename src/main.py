import os
import streamlit as st
from agents.media_processing_agent import MediaProcessingAgent
import asyncio
from typing import Dict
import time

def initialize_session_state():
    """Initialize session state variables"""
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}
    if 'batch_files' not in st.session_state:
        st.session_state.batch_files = []
    if 'results' not in st.session_state:
        st.session_state.results = {}

def display_progress(progress_dict: Dict):
    """Display progress bars and status messages"""
    for file_id, status in progress_dict.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(status.progress)
        with col2:
            st.write(f"{status.status} - {status.message}")

def display_results(results: Dict, file_type: str):
    """Display processed results based on file type"""
    for file_id, result in results.items():
        if isinstance(result, Exception):
            st.error(f"Error processing file {file_id}: {str(result)}")
            continue

        if file_type == "youtube":
            st.subheader(f"Stock Mentions - {file_id}")
            for company in result:
                with st.expander(f"{company.name} ({company.bullish_or_bearish})"):
                    st.write(f"**Why:** {company.why}")
        
        elif file_type == "podcast":
            st.subheader(f"Predictions - {file_id}")
            for pred in result:
                with st.expander(f"Prediction ({pred.timeframe})"):
                    st.write(pred.prediction)
        
        elif file_type == "pdf":
            st.subheader(f"Themes - {file_id}")
            for theme in result:
                st.write(f"- {theme.name}")

def main():
    st.set_page_config(page_title="Media Content Analyzer", layout="wide")
    initialize_session_state()

    # Sidebar
    with st.sidebar:
        st.title("Settings")
        processing_mode = st.radio("Processing Mode", ["Single File", "Batch Processing"])
        
        # Cache controls
        st.subheader("Cache Controls")
        if st.button("Clear Cache"):
            agent = MediaProcessingAgent()
            agent.cache.clear()
            st.success("Cache cleared!")

    # Main content
    st.title("Media Content Analyzer")
    
    # Initialize the agent
    agent = MediaProcessingAgent()

    # File upload section
    st.header("Upload Media")
    
    file_type = st.selectbox(
        "Select content type",
        ["youtube", "podcast", "pdf"],
        format_func=lambda x: {
            "youtube": "YouTube Video",
            "podcast": "Podcast Audio",
            "pdf": "PDF Newsletter"
        }[x]
    )

    if processing_mode == "Single File":
        uploaded_file = st.file_uploader(
            f"Upload your {file_type} file",
            type={
                "youtube": ["mp4", "mkv"],
                "podcast": ["mp3", "wav"],
                "pdf": ["pdf"]
            }[file_type]
        )

        if uploaded_file and st.button("Process"):
            st.session_state.processing_status = {}
            try:
                results = asyncio.run(agent.process_file(
                    uploaded_file, 
                    file_type,
                    agent.get_file_hash(uploaded_file),
                    st.session_state.processing_status
                ))
                st.session_state.results = {agent.get_file_hash(uploaded_file): results}
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    else:  # Batch Processing
        uploaded_files = st.file_uploader(
            f"Upload your {file_type} files",
            type={
                "youtube": ["mp4", "mkv"],
                "podcast": ["mp3", "wav"],
                "pdf": ["pdf"]
            }[file_type],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.session_state.batch_files = [
                {"file": f, "type": file_type} for f in uploaded_files
            ]

            if st.button("Process Batch"):
                st.session_state.processing_status = {}
                try:
                    results = asyncio.run(agent.process_batch(
                        st.session_state.batch_files,
                        st.session_state.processing_status
                    ))
                    st.session_state.results = results
                except Exception as e:
                    st.error(f"Error processing batch: {str(e)}")

    # Display progress
    if st.session_state.processing_status:
        st.header("Processing Progress")
        display_progress(st.session_state.processing_status)

    # Display results
    if st.session_state.results:
        st.header("Results")
        display_results(st.session_state.results, file_type)

if __name__ == "__main__":
    main()