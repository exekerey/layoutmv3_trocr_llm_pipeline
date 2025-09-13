import json
import os
import sys
import tempfile

import streamlit as st
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pipeline import DocumentPipeline


# Initialize the pipeline
@st.cache_resource
def load_pipeline():
    return DocumentPipeline(
        lang='ru',  # or 'kz' for Kazakh
        llm_api_key=os.environ.get("OPENAI_API_KEY")
    )


st.set_page_config(
    page_title="Banking Document OCR",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking Document OCR")
st.markdown("""
This application uses advanced OCR technology to extract information from banking documents.
Upload a scan of a receipt, contract, or bank statement to extract structured data.
""")

# Sidebar
st.sidebar.header("Settings")
lang_option = st.sidebar.selectbox(
    "Language",
    options=["Russian", "Kazakh"],
    index=0
)

lang_code = "ru" if lang_option == "Russian" else "kz"

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a banking document image",
                                     type=["jpg", "jpeg", "png", "pdf"])

    if uploaded_file is not None:
        # Save the file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # Display the image
        if tmp_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            st.image(tmp_path, caption="Uploaded Document", use_column_width=True)
        else:
            st.info("PDF file uploaded. Processing first page.")

        # Process button
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                try:
                    # Initialize pipeline with the selected language
                    pipeline = load_pipeline()

                    # Process the document
                    result = pipeline.process(tmp_path)

                    # Store results in session state
                    st.session_state.result = result
                    st.session_state.processing_complete = True

                    # Show success
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                finally:
                    # Clean up the temp file
                    os.unlink(tmp_path)

with col2:
    st.header("Results")
    if st.session_state.get("processing_complete", False):
        result = st.session_state.result

        # Display document type
        st.subheader(f"Document Type: {result['document_type'].capitalize()}")

        # Display confidence
        st.metric("Confidence", f"{result['confidence']:.2%}")

        # Display extracted data
        st.subheader("Extracted Data")
        st.json(result["extracted_data"])

        # Display processing times
        st.subheader("Processing Times")
        times = result["processing_times"]
        st.write(f"OCR: {times['ocr']:.2f}s")
        st.write(f"Vision Transformer: {times['vision_transformer']:.2f}s")
        st.write(f"LLM Processing: {times['llm']:.2f}s")
        st.write(f"Total: {times['total']:.2f}s")

        # Allow download of JSON
        st.download_button(
            label="Download JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False),
            file_name="extracted_data.json",
            mime="application/json"
        )
    else:
        st.info("Upload and process a document to see results here.")

# Display metrics
st.header("Performance Metrics")
st.markdown("""
The system is evaluated based on:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Field-level Accuracy
- JSON Structure Validity
""")

# Footer
st.markdown("---")
st.markdown("Banking Document OCR System | Built with PaddleOCR + LayoutLMv3 + LLM")
