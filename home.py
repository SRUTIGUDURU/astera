import streamlit as st
import platform
from datetime import datetime

st.set_page_config(
    page_title="GPU Cluster Management Suite",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header
st.title("🚀 GPU Cluster Management Suite")
st.markdown("### Comprehensive GPU cluster monitoring, control, and telemetry collection")

# Overview cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🖥️ GPU Node Agent
    Monitor and manage individual GPU nodes in your cluster.
    
    **Features:**
    - Real-time GPU telemetry
    - Automatic node registration
    - Background monitoring
    - Integration with control plane
    
    👉 **Go to GPU Node Agent page**
    """)

with col2:
    st.markdown("""
    ### 🎛️ Control Plane
    Central management hub for your GPU cluster.
    
    **Features:**
    - Cluster-wide monitoring
    - Job scheduling and management
    - Resource allocation
    - Performance analytics
    
    👉 **Go to Control Plane page**
    """)

with col3:
    st.markdown("""
    ### 📊 Telemetry Collection
    Hardware detection and performance monitoring.
    
    **Features:**
    - Multi-platform support
    - Real-time metrics
    - Historical data tracking
    - Export capabilities
    
    👉 **Go to Telemetry Collection page**
    """)

# System information
st.markdown("---")
st.subheader("🔍 System Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Operating System", platform.system())

with col2:
    st.metric("Python Version", platform.python_version())

with col3:
    st.metric("Architecture", platform.machine())

with col4:
    st.metric("Current Time", datetime.now().strftime("%H:%M:%S"))

# Getting started guide
st.markdown("---")
st.subheader("🚀 Getting Started")

st.markdown("""
1. **Start with Telemetry Collection** to detect your GPU hardware
2. **Use GPU Node Agent** to monitor individual nodes
3. **Access Control Plane** for cluster-wide management

**Navigation:** Use the sidebar to switch between different components of the suite.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>GPU Cluster Management Suite - Built with Streamlit</p>
    <p>Supports NVIDIA, AMD, Intel, and Apple GPUs across Windows, Linux, and macOS</p>
</div>
""", unsafe_allow_html=True)
