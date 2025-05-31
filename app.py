import streamlit as st
import json
from document_loader import DocumentLoader
from langchain_agent import LangChainAgent
from shared_memory import SharedMemory
import uuid
import mimetypes

# Initialize components
document_loader = DocumentLoader()
langchain_agent = LangChainAgent()
shared_memory = SharedMemory()

st.set_page_config(page_title="Multi-Agent AI System", layout="wide")

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .upload-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #dee2e6;
        margin-bottom: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #667eea;
        background: #f0f2ff;
    }
    
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-container h3 {
        margin: 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    .metric-container h2 {
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .history-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .history-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.4);
    }
    
    .json-container {
        background: #2d3748;
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #4a5568;
        font-family: 'Courier New', monospace;
    }
    
    .success-message {
        background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(90deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Beautiful header
st.markdown("""
<div class="main-header">
    <h1>🤖 Multi-Agent AI System</h1>
    <p style="font-size: 1.2rem; margin: 0; opacity: 0.9;">Upload a document (PDF, JSON, or Email) for processing</p>
</div>
""", unsafe_allow_html=True)

# File upload section with beautiful styling
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
st.markdown("### 📁 Choose Your Document")
uploaded_file = st.file_uploader("Select a document to process", type=['pdf', 'json', 'txt'], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # File info display
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 📋 File Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📄 Filename</h3>
            <h2 style="font-size: 1rem;">{uploaded_file.name}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>📊 Size</h3>
            <h2>{uploaded_file.size} bytes</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🏷️ Type</h3>
            <h2 style="font-size: 1rem;">{uploaded_file.type}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Read file content
    content = uploaded_file.read()
    
    # Get file type from extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Map MIME types to our supported types
    mime_type = uploaded_file.type
    if mime_type == 'application/pdf':
        file_type = 'pdf'
    elif mime_type == 'application/json':
        file_type = 'json'
    elif mime_type == 'text/plain':
        file_type = 'txt'
    else:
        file_type = file_extension
    
    # Process button
    if st.button("🚀 Process Document", use_container_width=True):
        with st.spinner("🔄 Processing your document..."):
            try:
                # Load document using LangChain
                docs = document_loader.load_document(content, file_type, uploaded_file.name)
                
                if not docs:
                    st.markdown("""
                    <div class="error-message">
                        ❌ No content could be extracted from the document.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Process the first document
                    doc = docs[0]
                    
                    # Process using LangChain agent
                    result = langchain_agent.process_document(doc)
                    
                    # Success message
                    st.markdown("""
                    <div class="success-message">
                        ✅ Processing completed successfully!
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display results with beautiful cards
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("### 🎯 Document Analysis Results")
                    
                    # Display format and intent with beautiful metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>📋 Format</h3>
                            <h2>{result["format"]}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h3>🎯 Intent</h3>
                            <h2>{result["intent"]}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Display analysis with beautiful formatting
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f"### 📊 {result['format']} Analysis")
                    
                    # Beautiful JSON display
                    analysis_json = json.dumps(result["analysis"], indent=2)
                    st.markdown(f"""
                    <div class="json-container">
                        <pre>{analysis_json}</pre>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Store results in shared memory
                    shared_memory.store_processing_result(
                        task_id=str(uuid.uuid4()),
                        source=uploaded_file.name,
                        format_type=result["format"],
                        intent=result["intent"],
                        extracted_data=result["analysis"]
                    )
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-message">
                    ❌ Error processing document: {str(e)}
                </div>
                """, unsafe_allow_html=True)

# Display processing history with beautiful styling
st.markdown("---")
st.markdown("### 📚 Processing History")

history = shared_memory.get_recent_history(5)
if history:
    for item in history:
        # Create expandable content
        with st.expander(f"📄 {item['source']} - {item['timestamp']}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container" style="margin: 0.5rem 0;">
                    <h3>📋 Format</h3>
                    <h2 style="font-size: 1.5rem;">{item['format_type']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container" style="margin: 0.5rem 0;">
                    <h3>🎯 Intent</h3>
                    <h2 style="font-size: 1.5rem;">{item['intent']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**📊 Analysis:**")
            analysis_json = json.dumps(item['extracted_data'], indent=2)
            st.markdown(f"""
            <div class="json-container">
                <pre>{analysis_json}</pre>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #6c757d; background: #f8f9fa; border-radius: 15px;">
        <h3>📭 No processing history available</h3>
        <p style="font-size: 1.1rem;">Upload and process your first document to see results here!</p>
    </div>
    """, unsafe_allow_html=True)
