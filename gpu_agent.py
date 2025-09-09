import streamlit as st
import asyncio
import json
import logging
import socket
import time
import threading
from datetime import datetime
import aiohttp
import platform
from typing import Dict, List, Optional
import pandas as pd

# Import your telemetry module
from gpu_telemetry import create_telemetry_collector, GPUStats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitGPUAgent:
    def __init__(self, control_plane_url: str, node_id: str = None):
        self.control_plane_url = control_plane_url.rstrip('/')
        self.node_id = node_id or f"{platform.node()}-{socket.gethostname()}"
        self.telemetry = create_telemetry_collector()
        self.registered_gpus = {}
        self.running = False
        self.last_metrics = {}
        self.last_update = None
        self.background_thread = None
        
    def start_background_tasks(self):
        """Start background tasks in a separate thread"""
        if not self.running:
            self.running = True
            self.background_thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.background_thread.start()
    
    def stop_background_tasks(self):
        """Stop background tasks"""
        self.running = False
        if self.background_thread:
            self.background_thread.join(timeout=1)
    
    def _run_async_loop(self):
        """Run async event loop in background thread"""
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._background_tasks())
        except Exception as e:
            logger.error(f"Background task error: {e}")
        finally:
            try:
                loop.close()
            except:
                pass
    
    async def _background_tasks(self):
        """Run the background monitoring tasks"""
        # Register GPUs first
        await self.register_gpus()
        
        # Start monitoring loops
        await asyncio.gather(
            self.telemetry_loop(),
            self.heartbeat_loop(),
            return_exceptions=True
        )
    
    async def register_gpus(self):
        """Register all detected GPUs with control plane"""
        try:
            async with aiohttp.ClientSession() as session:
                for i, gpu_info in enumerate(self.telemetry.gpus):
                    gpu_node_id = f"{self.node_id}-gpu{i}"
                    
                    data = {
                        "node_id": gpu_node_id,
                        "hostname": platform.node(),
                        "gpu_name": gpu_info.get('name', 'Unknown GPU'),
                        "gpu_uuid": gpu_info.get('uuid', ''),
                        "vendor": gpu_info.get('vendor', 'Unknown'),
                        "memory_total_mb": gpu_info.get('memory_total', 0),
                        "driver_version": gpu_info.get('driver_version', ''),
                        "tags": ["telemetry-enabled"],
                        "location": platform.node()
                    }
                    
                    # Add capabilities if available
                    if gpu_info.get('vendor') == 'NVIDIA':
                        data["capabilities"] = {
                            "compute_capability": gpu_info.get('compute_capability', ''),
                            "supports_nvlink": False,
                            "supports_mig": False
                        }
                    
                    async with session.post(
                        f"{self.control_plane_url}/register",
                        json=data
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            self.registered_gpus[i] = {
                                'gpu_id': result['gpu_id'],
                                'node_id': gpu_node_id
                            }
                            logger.info(f"Registered GPU {i} as {gpu_node_id}")
                        else:
                            logger.error(f"Failed to register GPU {i}: {await resp.text()}")
        except Exception as e:
            logger.error(f"Error in register_gpus: {e}")
    
    async def telemetry_loop(self):
        """Send GPU metrics to control plane"""
        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    # Collect metrics for all GPUs
                    gpu_stats = self.telemetry.get_all_gpu_stats()
                    metrics_update = {}
                    
                    for stats in gpu_stats:
                        if stats.gpu_id in self.registered_gpus:
                            node_id = self.registered_gpus[stats.gpu_id]['node_id']
                            
                            # Prepare metrics update
                            metrics = {
                                "temperature_celsius": stats.temperature,
                                "utilization_percent": stats.utilization,
                                "memory_used_mb": stats.memory_used,
                                "power_watts": stats.power_usage,
                                "fan_speed_percent": stats.fan_speed,
                                "clock_speed_mhz": stats.clock_core,
                                "memory_clock_mhz": stats.clock_memory
                            }
                            
                            # Remove None values
                            metrics = {k: v for k, v in metrics.items() if v is not None and v > 0}
                            metrics_update[node_id] = metrics
                            
                            # Send update to control plane
                            try:
                                async with session.patch(
                                    f"{self.control_plane_url}/gpu/{node_id}/metrics",
                                    json=metrics
                                ) as resp:
                                    if resp.status != 200:
                                        logger.warning(f"Failed to update metrics for {node_id}")
                            except Exception as e:
                                logger.error(f"Error updating metrics for {node_id}: {e}")
                    
                    # Update last metrics for UI display
                    self.last_metrics = metrics_update
                    self.last_update = datetime.now()
                
                except Exception as e:
                    logger.error(f"Telemetry loop error: {e}")
                
                await asyncio.sleep(10)
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats"""
        while self.running:
            try:
                # Heartbeat logic if needed
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Get current GPU metrics for display"""
        if not self.last_metrics:
            # Fallback to direct telemetry if no background data
            try:
                gpu_stats = self.telemetry.get_all_gpu_stats()
                metrics = {}
                for stats in gpu_stats:
                    gpu_name = f"GPU {stats.gpu_id}"
                    metrics[gpu_name] = {
                        "temperature_celsius": stats.temperature or 0,
                        "utilization_percent": stats.utilization or 0,
                        "memory_used_mb": stats.memory_used or 0,
                        "power_watts": stats.power_usage or 0,
                        "fan_speed_percent": stats.fan_speed or 0,
                    }
                return metrics
            except Exception as e:
                logger.error(f"Error getting fallback metrics: {e}")
                return {}
        
        return self.last_metrics

def main():
    st.set_page_config(
        page_title="GPU Node Agent",
        page_icon="🖥️",
        layout="wide"
    )
    
    st.title("🖥️ GPU Node Agent Dashboard")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
        st.session_state.agent_running = False
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        control_plane_url = st.text_input(
            "Control Plane URL",
            value="http://localhost:8000",
            help="URL of the control plane server"
        )
        
        node_id = st.text_input(
            "Node ID (optional)",
            value="",
            help="Custom node ID, leave empty for auto-generation"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Start Agent", type="primary"):
                if not st.session_state.agent_running:
                    try:
                        st.session_state.agent = StreamlitGPUAgent(
                            control_plane_url=control_plane_url,
                            node_id=node_id if node_id else None
                        )
                        st.session_state.agent.start_background_tasks()
                        st.session_state.agent_running = True
                        st.success("Agent started successfully!")
                        time.sleep(1)  # Give it a moment to initialize
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to start agent: {e}")
                else:
                    st.warning("Agent is already running")
        
        with col2:
            if st.button("Stop Agent"):
                if st.session_state.agent_running and st.session_state.agent:
                    st.session_state.agent.stop_background_tasks()
                    st.session_state.agent_running = False
                    st.session_state.agent = None
                    st.success("Agent stopped")
                    st.rerun()
    
    # Main dashboard
    if st.session_state.agent_running and st.session_state.agent:
        agent = st.session_state.agent
        
        # Status indicator
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Status", "🟢 Running")
        with col2:
            st.metric("Node ID", agent.node_id)
        with col3:
            if agent.last_update:
                st.metric("Last Update", agent.last_update.strftime("%H:%M:%S"))
        
        st.divider()
        
        # Auto-refresh
        if st.checkbox("Auto-refresh (5s)", value=True):
            time.sleep(5)
            st.rerun()
        
        # Get current metrics
        metrics = agent.get_current_metrics()
        
        if metrics:
            st.subheader("GPU Metrics")
            
            # Create metrics display
            for gpu_name, gpu_metrics in metrics.items():
                with st.expander(f"📊 {gpu_name}", expanded=True):
                    cols = st.columns(5)
                    
                    with cols[0]:
                        temp = gpu_metrics.get('temperature_celsius', 0)
                        st.metric("Temperature", f"{temp}°C")
                    
                    with cols[1]:
                        util = gpu_metrics.get('utilization_percent', 0)
                        st.metric("Utilization", f"{util}%")
                    
                    with cols[2]:
                        memory = gpu_metrics.get('memory_used_mb', 0)
                        st.metric("Memory Used", f"{memory} MB")
                    
                    with cols[3]:
                        power = gpu_metrics.get('power_watts', 0)
                        st.metric("Power", f"{power} W")
                    
                    with cols[4]:
                        fan = gpu_metrics.get('fan_speed_percent', 0)
                        st.metric("Fan Speed", f"{fan}%")
            
            # Convert to DataFrame for charting
            if len(metrics) > 0:
                df_data = []
                for gpu_name, gpu_metrics in metrics.items():
                    row = {"GPU": gpu_name}
                    row.update(gpu_metrics)
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                
                st.subheader("Charts")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'utilization_percent' in df.columns:
                        st.bar_chart(df.set_index('GPU')[['utilization_percent']])
                        st.caption("GPU Utilization %")
                
                with col2:
                    if 'temperature_celsius' in df.columns:
                        st.bar_chart(df.set_index('GPU')[['temperature_celsius']])
                        st.caption("Temperature (°C)")
        
        else:
            st.warning("No GPU metrics available. Make sure GPUs are detected and agent is running.")
    
    else:
        st.info("👆 Use the sidebar to configure and start the GPU agent")
        st.markdown("""
        ### Features:
        - 🔍 Auto-discover and register GPUs
        - 📊 Real-time metrics monitoring
        - 🔄 Background telemetry collection
        - 📡 Integration with control plane
        - 📈 Live dashboard with charts
        """)

if __name__ == "__main__":
    main()
