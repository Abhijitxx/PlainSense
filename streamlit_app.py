"""
Streamlit App for OCR Document Processing Pipeline
Clean, professional UI
"""

import streamlit as st
from pathlib import Path
import time
from pipeline import CompletePipeline

# Page configuration
st.set_page_config(
    page_title="OCR Document Processing",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI
st.markdown("""
    <style>
    /* Better tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 8px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 32px;
        background-color: transparent;
        border-radius: 6px;
        color: #1f2937;
        font-size: 16px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e5e7eb;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff;
        color: #0066cc;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* Better metrics */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 500;
        color: #6b7280;
    }
    /* Better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Better buttons */
    .stButton > button {
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results_history' not in st.session_state:
    st.session_state.results_history = []
if 'document_type' not in st.session_state:
    st.session_state.document_type = 'medical'

# Header
st.title("OCR Document Processing Pipeline")
st.markdown("Transform documents into clean, structured text")
st.markdown("")

# Document Type Toggle Section
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("#### General Documents")
    st.markdown("Contracts • Agreements • General Text")
    st.caption("ML correction enabled")
    
with col2:
    st.markdown("#### Processing Mode")
    document_type = st.toggle(
        "Switch to Medical",
        value=(st.session_state.document_type == 'medical'),
        help="Toggle between general and medical processing"
    )
    
    new_type = 'medical' if document_type else 'general'
    if new_type != st.session_state.document_type or st.session_state.pipeline is None:
        st.session_state.document_type = new_type
        use_ml = (new_type == 'general')
        st.session_state.pipeline = CompletePipeline(
            output_dir='output',
            use_ml_correction=use_ml
        )
        st.rerun()

with col3:
    st.markdown("#### Medical Lab Reports")
    st.markdown("Lab Tests • Medical Results • Forms")
    st.caption("Medical corrections enabled")

# Mode indicator
if st.session_state.document_type == 'medical':
    st.info("**MEDICAL MODE** • Optimized for lab reports • Medical corrections • ML disabled")
else:
    st.success("**GENERAL MODE** • Optimized for documents • Standard corrections • ML enabled")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    if st.session_state.document_type == 'medical':
        st.markdown("**Mode:** Medical")
        st.markdown("**ML Correction:** OFF")
        st.markdown("**Corrections:** 25+ medical terms")
        st.markdown("**Accuracy:** 87-92%")
    else:
        st.markdown("**Mode:** General")
        st.markdown("**ML Correction:** ON")
        st.markdown("**Dictionary:** 82,834 words")
        st.markdown("**Accuracy:** 85-95%")
    
    st.markdown("---")
    st.header("Speed")
    st.markdown("**Text:** < 1s")
    st.markdown("**DOCX:** 1-2s")
    st.markdown("**PDF:** 2-10s")
    st.markdown("**Images:** 8-15s")
    
    st.markdown("---")
    st.header("Statistics")
    total = len(st.session_state.results_history)
    st.metric("Processed", total)
    if st.session_state.results_history:
        words = sum(r['word_count'] for r in st.session_state.results_history)
        st.metric("Total Words", f"{words:,}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Text Input", "File Upload", "History", "System Info"])

# Tab 1: Text Input
with tab1:
    st.header("Direct Text Input")
    st.markdown("Paste text for instant processing")
    st.markdown("")
    
    text_input = st.text_area(
        "text",
        height=250,
        placeholder="Paste document text here...",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        name = st.text_input("n", "text_document", placeholder="Document name", label_visibility="collapsed")
    with col2:
        btn1 = st.button("Process", use_container_width=True, type="primary")
    
    if btn1:
        if text_input.strip():
            with st.spinner("Processing..."):
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                temp_path = temp_dir / f"{name}.txt"
                temp_path.write_text(text_input, encoding='utf-8')
                
                result = st.session_state.pipeline.process_document(str(temp_path))
                
                if result['success']:
                    st.success("Completed")
                    
                    result['document_name'] = name
                    result['document_type'] = st.session_state.document_type
                    st.session_state.results_history.append(result)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mode", "Medical" if st.session_state.document_type == 'medical' else "General")
                    c2.metric("Words", result['word_count'])
                    c3.metric("Time", f"{result.get('processing_time', 0):.2f}s")
                    c4.metric("Status", "Success")
                    
                    txt = result.get('text', '')
                    st.markdown("---")
                    st.subheader("Output Preview")
                    st.text_area("preview", txt[:1500], height=300, label_visibility="collapsed")
                    
                    st.download_button(
                        "Download",
                        txt,
                        f"{name}_processed.txt",
                        type="primary"
                    )
                else:
                    st.error(f"Error: {result.get('error', 'Unknown')}")
        else:
            st.warning("Enter text first")

# Tab 2: File Upload
with tab2:
    st.header("File Upload")
    st.markdown("Supports: DOCX • PDF • TXT • JPG • PNG")
    st.markdown("")
    
    file = st.file_uploader(
        "file",
        type=['docx', 'pdf', 'txt', 'jpg', 'jpeg', 'png'],
        label_visibility="collapsed"
    )
    
    if file:
        c1, c2, c3 = st.columns(3)
        c1.metric("Name", file.name)
        c2.metric("Size", f"{file.size / 1024:.1f} KB")
        c3.metric("Type", file.name.split('.')[-1].upper())
        
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_path = temp_dir / file.name
        
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        
        st.success(f"Uploaded: {file.name}")
        
        btn2 = st.button("Process File", use_container_width=True, type="primary")
        
        if btn2:
            with st.spinner("Processing..."):
                result = st.session_state.pipeline.process_document(str(temp_path))
                
                if result['success']:
                    st.success("Completed")
                    
                    result['document_name'] = Path(file.name).stem
                    result['document_type'] = st.session_state.document_type
                    st.session_state.results_history.append(result)
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mode", "Medical" if st.session_state.document_type == 'medical' else "General")
                    c2.metric("Method", result.get('extraction_method', 'N/A').replace('_', ' ').title())
                    c3.metric("Words", result['word_count'])
                    c4.metric("Time", f"{result.get('processing_time', 0):.2f}s")
                    
                    txt = result.get('text', '')
                    st.markdown("---")
                    
                    # Quality check for medical
                    if st.session_state.document_type == 'medical':
                        st.subheader("Quality Check")
                        wc = len(txt.split())
                        nums = any(char.isdigit() for char in txt)
                        letters = any(char.isalpha() for char in txt)
                        
                        qc1, qc2, qc3 = st.columns(3)
                        qc1.metric("Words", wc)
                        qc2.metric("Numbers", "Yes" if nums else "No")
                        qc3.metric("Text", "Yes" if letters else "No")
                        
                        if wc > 50 and nums and letters:
                            st.success("Document OK")
                        elif wc < 20:
                            st.warning("Low word count")
                        elif not nums:
                            st.warning("No numbers found")
                        
                        st.markdown("---")
                    
                    st.subheader("Output Preview")
                    st.text_area("preview", txt[:1500], height=300, label_visibility="collapsed")
                    
                    st.download_button(
                        "Download",
                        txt,
                        f"{Path(file.name).stem}_processed.txt",
                        type="primary"
                    )
                else:
                    st.error(f"Error: {result.get('error', 'Unknown')}")

# Tab 3: History
with tab3:
    st.header("Processing History")
    
    if st.session_state.results_history:
        c1, c2, c3, c4 = st.columns(4)
        
        total = len(st.session_state.results_history)
        words = sum(r['word_count'] for r in st.session_state.results_history)
        
        types = {}
        for r in st.session_state.results_history:
            t = r.get('document_type', 'unknown')
            types[t] = types.get(t, 0) + 1
        
        c1.metric("Total", total)
        c2.metric("Words", f"{words:,}")
        c3.metric("Medical", types.get('medical', 0))
        c4.metric("General", types.get('general', 0))
        
        st.markdown("---")
        st.subheader("All Documents")
        
        data = []
        for i, r in enumerate(st.session_state.results_history, 1):
            dtype = r.get('document_type', 'unknown')
            data.append({
                "#": i,
                "Name": r.get('document_name', 'N/A'),
                "Type": "Medical" if dtype == 'medical' else "General",
                "Method": r.get('extraction_method', 'N/A'),
                "Words": r['word_count'],
                "Time": f"{r.get('processing_time', 0):.1f}s",
                "Status": "Success" if r['success'] else "Failed"
            })
        
        st.dataframe(data, use_container_width=True, hide_index=True)
        
        if st.button("Clear History"):
            st.session_state.results_history = []
            st.rerun()
    else:
        st.info("No documents processed yet")

# Tab 4: Info
with tab4:
    st.header("System Information")
    
    st.subheader("Configuration")
    st.markdown("**Tesseract:** C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    st.markdown("**Poppler:** C:\\poppler\\poppler-24.08.0\\Library\\bin")
    st.markdown("**OCR Mode:** PSM 6 (uniform block)")
    
    st.markdown("---")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Medical Mode")
        st.markdown("""
        **For:** Lab reports, medical forms
        
        **Features:**
        - 25+ medical term corrections
        - Pattern recognition (g/dl, cells/cumm)
        - ML correction disabled
        - 87-92% accuracy
        
        **Sample Corrections:**
        - "whale boo" → "whole blood"
        - "god" → "g/dl"
        - "cells cumin" → "cells/cumm"
        """)
    
    with c2:
        st.subheader("General Mode")
        st.markdown("""
        **For:** Contracts, agreements
        
        **Features:**
        - ML correction (SymSpell)
        - 82,834 word dictionary
        - Legal term standardization
        - 85-95% accuracy
        
        **Sample Corrections:**
        - "die" → "the"
        - "tlie" → "the"
        - "witli" → "with"
        """)
    
    st.markdown("---")
    st.subheader("File Types")
    st.markdown("**DOCX:** Direct extraction (1-2s)")
    st.markdown("**PDF:** Auto-detect typed/scanned (2-10s)")
    st.markdown("**Images:** OCR processing (8-15s)")
    st.markdown("**Text:** Direct processing (< 1s)")

# Footer
st.markdown("---")
st.caption("OCR Document Processing Pipeline v2.0")
