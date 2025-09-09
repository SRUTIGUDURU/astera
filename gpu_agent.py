import streamlit as st
import sqlite3
import json
import uuid
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from contextlib import contextmanager
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# Configuration
DB_FILE = "gpu_cluster.db"
HEARTBEAT_TIMEOUT_SECONDS = 60
CLEANUP_INTERVAL_SECONDS = 30
METRICS_RETENTION_DAYS = 7

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# Enums and Constants
# ==============================================================================

class GPUState(str, Enum):
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class JobState(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class SchedulingStrategy(str, Enum):
    BEST_FIT = "best_fit"
    FIRST_FIT = "first_fit"
    LOAD_BALANCED = "load_balanced"
    POWER_EFFICIENT = "power_efficient"
    AFFINITY = "affinity"

# ==============================================================================
# Database Management
# ==============================================================================

class DatabaseManager:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        with self.get_connection() as conn:
            # GPUs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gpus (
                    gpu_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT UNIQUE NOT NULL,
                    hostname TEXT NOT NULL,
                    gpu_name TEXT NOT NULL,
                    gpu_uuid TEXT,
                    vendor TEXT DEFAULT 'NVIDIA',
                    state TEXT DEFAULT 'available',
                    memory_total_mb INTEGER NOT NULL,
                    memory_allocated_mb INTEGER DEFAULT 0,
                    memory_used_mb INTEGER DEFAULT 0,
                    driver_version TEXT,
                    
                    temperature_celsius REAL DEFAULT 0,
                    utilization_percent REAL DEFAULT 0,
                    power_watts REAL DEFAULT 0,
                    fan_speed_percent INTEGER DEFAULT 0,
                    clock_speed_mhz INTEGER DEFAULT 0,
                    memory_clock_mhz INTEGER DEFAULT 0,
                    
                    compute_capability TEXT,
                    cuda_cores INTEGER DEFAULT 0,
                    tensor_cores INTEGER DEFAULT 0,
                    memory_bandwidth_gb REAL DEFAULT 0,
                    supports_nvlink BOOLEAN DEFAULT 0,
                    supports_mig BOOLEAN DEFAULT 0,
                    
                    location TEXT,
                    tags TEXT DEFAULT '[]',
                    registered_at TEXT NOT NULL,
                    last_heartbeat TEXT NOT NULL,
                    total_runtime_hours REAL DEFAULT 0,
                    total_jobs_completed INTEGER DEFAULT 0,
                    
                    CHECK (state IN ('available', 'busy', 'offline', 'maintenance', 'error'))
                )
            """)
            
            # Jobs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    priority INTEGER DEFAULT 5,
                    requested_gpus INTEGER DEFAULT 1,
                    memory_per_gpu_mb INTEGER NOT NULL,
                    expected_duration_minutes INTEGER,
                    scheduling_strategy TEXT DEFAULT 'best_fit',
                    gpu_type_preference TEXT,
                    require_nvlink BOOLEAN DEFAULT 0,
                    tags TEXT DEFAULT '[]',
                    metadata TEXT DEFAULT '{}',
                    
                    state TEXT DEFAULT 'pending',
                    scheduled_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    failed_at TEXT,
                    error_message TEXT,
                    
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    
                    CHECK (state IN ('pending', 'scheduled', 'running', 'completed', 'failed', 'cancelled'))
                )
            """)
            
            # Job assignments
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_assignments (
                    assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    gpu_id INTEGER NOT NULL,
                    assigned_at TEXT NOT NULL,
                    released_at TEXT,
                    
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id),
                    FOREIGN KEY (gpu_id) REFERENCES gpus(gpu_id),
                    UNIQUE(job_id, gpu_id)
                )
            """)
            
            # Metrics history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gpu_metrics_history (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gpu_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    temperature_celsius REAL,
                    utilization_percent REAL,
                    memory_used_mb INTEGER,
                    power_watts REAL,
                    
                    FOREIGN KEY (gpu_id) REFERENCES gpus(gpu_id)
                )
            """)
            
            # Cluster events
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cluster_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT DEFAULT 'info',
                    source TEXT,
                    message TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Create indices
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gpus_state ON gpus(state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON gpu_metrics_history(timestamp)")
            
            conn.commit()

# ==============================================================================
# GPU Scheduler
# ==============================================================================

class GPUScheduler:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def schedule_job(self, job_request: dict) -> Optional[List[int]]:
        with self.db.get_connection() as conn:
            available_gpus = self._get_available_gpus(conn, job_request)
            
            if len(available_gpus) < job_request['requested_gpus']:
                return None
            
            # Simple first-fit scheduling for Streamlit version
            selected = [g['gpu_id'] for g in available_gpus[:job_request['requested_gpus']]]
            return selected
    
    def _get_available_gpus(self, conn, job_request: dict) -> List[Dict]:
        query = """
            SELECT * FROM gpus 
            WHERE state = 'available' 
            AND (memory_total_mb - memory_allocated_mb) >= ?
        """
        params = [job_request['memory_per_gpu_mb']]
        
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

# ==============================================================================
# Streamlit App
# ==============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager(DB_FILE)
    if 'scheduler' not in st.session_state:
        st.session_state.scheduler = GPUScheduler(st.session_state.db_manager)

def get_cluster_metrics():
    """Get current cluster metrics"""
    db = st.session_state.db_manager
    
    with db.get_connection() as conn:
        # GPU statistics
        gpu_stats = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN state = 'available' THEN 1 ELSE 0 END) as available,
                SUM(CASE WHEN state = 'busy' THEN 1 ELSE 0 END) as busy,
                SUM(CASE WHEN state = 'offline' THEN 1 ELSE 0 END) as offline,
                SUM(memory_total_mb) as total_memory,
                SUM(memory_allocated_mb) as allocated_memory,
                AVG(CASE WHEN state != 'offline' THEN utilization_percent ELSE NULL END) as avg_util,
                AVG(CASE WHEN state != 'offline' THEN temperature_celsius ELSE NULL END) as avg_temp,
                SUM(CASE WHEN state != 'offline' THEN power_watts ELSE 0 END) as total_power
            FROM gpus
        """).fetchone()
        
        # Job statistics
        job_stats = conn.execute("""
            SELECT 
                SUM(CASE WHEN state = 'running' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN state = 'pending' THEN 1 ELSE 0 END) as pending
            FROM jobs
        """).fetchone()
        
        return {
            'total_gpus': gpu_stats['total'] or 0,
            'available_gpus': gpu_stats['available'] or 0,
            'busy_gpus': gpu_stats['busy'] or 0,
            'offline_gpus': gpu_stats['offline'] or 0,
            'total_memory_mb': gpu_stats['total_memory'] or 0,
            'allocated_memory_mb': gpu_stats['allocated_memory'] or 0,
            'average_utilization': gpu_stats['avg_util'] or 0.0,
            'average_temperature': gpu_stats['avg_temp'] or 0.0,
            'total_power_watts': gpu_stats['total_power'] or 0.0,
            'active_jobs': job_stats['active'] or 0,
            'pending_jobs': job_stats['pending'] or 0
        }

def render_overview_tab():
    """Render the overview dashboard"""
    st.header("🎛️ Cluster Overview")
    
    # Auto-refresh checkbox
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True, key="overview_refresh")
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Get metrics
    metrics = get_cluster_metrics()
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total GPUs", 
            metrics['total_gpus'],
            help="Total number of registered GPUs"
        )
    
    with col2:
        available_pct = (metrics['available_gpus'] / max(metrics['total_gpus'], 1)) * 100
        st.metric(
            "Available GPUs", 
            f"{metrics['available_gpus']}/{metrics['total_gpus']}",
            delta=f"{available_pct:.1f}% available",
            help="GPUs ready for job assignment"
        )
    
    with col3:
        st.metric(
            "Active Jobs", 
            metrics['active_jobs'],
            delta=f"{metrics['pending_jobs']} queued",
            help="Currently running and pending jobs"
        )
    
    with col4:
        memory_used_pct = (metrics['allocated_memory_mb'] / max(metrics['total_memory_mb'], 1)) * 100
        st.metric(
            "Memory Usage", 
            f"{memory_used_pct:.1f}%",
            delta=f"{metrics['allocated_memory_mb']:,} MB used",
            help="GPU memory allocation across cluster"
        )
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Avg Utilization",
            f"{metrics['average_utilization']:.1f}%",
            help="Average GPU utilization across active GPUs"
        )
    
    with col2:
        st.metric(
            "Avg Temperature",
            f"{metrics['average_temperature']:.1f}°C",
            help="Average temperature across active GPUs"
        )
    
    with col3:
        st.metric(
            "Total Power",
            f"{metrics['total_power_watts']:.0f}W",
            help="Total power consumption"
        )

def render_gpus_tab():
    """Render GPU management tab"""
    st.header("🖥️ GPU Management")
    
    db = st.session_state.db_manager
    
    # GPU registration form
    with st.expander("Register New GPU", expanded=False):
        with st.form("register_gpu_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                node_id = st.text_input("Node ID*", help="Unique identifier for this GPU node")
                hostname = st.text_input("Hostname*", help="Server hostname")
                gpu_name = st.text_input("GPU Name*", placeholder="e.g., RTX 4090", help="GPU model name")
                memory_mb = st.number_input("Memory (MB)*", min_value=1024, value=24576, help="Total GPU memory in MB")
            
            with col2:
                vendor = st.selectbox("Vendor", ["NVIDIA", "AMD", "Intel"], help="GPU manufacturer")
                driver_version = st.text_input("Driver Version", placeholder="e.g., 535.129.03", help="GPU driver version")
                location = st.text_input("Location", placeholder="e.g., rack-1", help="Physical location")
                tags = st.text_input("Tags (comma-separated)", placeholder="e.g., compute, training", help="GPU tags for organization")
            
            submitted = st.form_submit_button("Register GPU", type="primary")
            
            if submitted:
                if not all([node_id, hostname, gpu_name]):
                    st.error("Please fill in all required fields (marked with *)")
                else:
                    try:
                        with db.get_connection() as conn:
                            # Check if already exists
                            existing = conn.execute(
                                "SELECT gpu_id FROM gpus WHERE node_id = ?", 
                                (node_id,)
                            ).fetchone()
                            
                            if existing:
                                st.error(f"GPU with node ID '{node_id}' already exists")
                            else:
                                now = datetime.utcnow().isoformat()
                                tag_list = [t.strip() for t in tags.split(',') if t.strip()]
                                
                                cursor = conn.execute("""
                                    INSERT INTO gpus (
                                        node_id, hostname, gpu_name, vendor,
                                        memory_total_mb, driver_version, location, tags,
                                        registered_at, last_heartbeat
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    node_id, hostname, gpu_name, vendor,
                                    memory_mb, driver_version, location, json.dumps(tag_list),
                                    now, now
                                ))
                                conn.commit()
                                
                                st.success(f"Successfully registered GPU '{gpu_name}' on node '{node_id}'")
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error registering GPU: {e}")

    # GPU list and management
    with db.get_connection() as conn:
        gpus = conn.execute("""
            SELECT * FROM gpus 
            ORDER BY gpu_id
        """).fetchall()
    
    if gpus:
        st.subheader("GPU Cluster Status")
        
        # Create DataFrame for display
        gpu_data = []
        for gpu in gpus:
            gpu_dict = dict(gpu)
            gpu_dict['tags'] = ', '.join(json.loads(gpu_dict.get('tags', '[]')))
            
            # Memory usage percentage
            if gpu_dict['memory_total_mb'] > 0:
                gpu_dict['memory_usage_pct'] = (gpu_dict['memory_allocated_mb'] / gpu_dict['memory_total_mb']) * 100
            else:
                gpu_dict['memory_usage_pct'] = 0
            
            gpu_data.append(gpu_dict)
        
        df = pd.DataFrame(gpu_data)
        
        # State filter
        col1, col2 = st.columns([1, 3])
        with col1:
            state_filter = st.selectbox(
                "Filter by State",
                options=["All"] + [state.value for state in GPUState],
                key="gpu_state_filter"
            )
        
        if state_filter != "All":
            df = df[df['state'] == state_filter]
        
        # Display GPU cards
        for idx, gpu in df.iterrows():
            with st.container():
                # State indicator
                state_colors = {
                    'available': '🟢',
                    'busy': '🟡',
                    'offline': '🔴',
                    'maintenance': '🔧',
                    'error': '⚠️'
                }
                
                state_icon = state_colors.get(gpu['state'], '❓')
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{state_icon} GPU {gpu['gpu_id']}: {gpu['gpu_name']}**")
                    st.caption(f"Node: {gpu['node_id']} | Host: {gpu['hostname']}")
                
                with col2:
                    st.metric(
                        "Utilization", 
                        f"{gpu['utilization_percent']:.1f}%",
                        help="GPU utilization percentage"
                    )
                
                with col3:
                    st.metric(
                        "Memory", 
                        f"{gpu['memory_usage_pct']:.1f}%",
                        delta=f"{gpu['memory_allocated_mb']}/{gpu['memory_total_mb']} MB",
                        help="Memory allocation"
                    )
                
                with col4:
                    if st.button(f"Manage", key=f"manage_gpu_{gpu['gpu_id']}"):
                        st.session_state.selected_gpu = gpu['gpu_id']
                        st.rerun()
                
                st.divider()
        
        # GPU management modal
        if 'selected_gpu' in st.session_state:
            gpu_id = st.session_state.selected_gpu
            selected_gpu = df[df['gpu_id'] == gpu_id].iloc[0]
            
            st.subheader(f"Manage GPU {gpu_id}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_state = st.selectbox(
                    "Change State",
                    options=[state.value for state in GPUState],
                    index=[state.value for state in GPUState].index(selected_gpu['state']),
                    key=f"state_select_{gpu_id}"
                )
                
                if st.button("Update State", key=f"update_state_{gpu_id}"):
                    with db.get_connection() as conn:
                        conn.execute(
                            "UPDATE gpus SET state = ? WHERE gpu_id = ?",
                            (new_state, gpu_id)
                        )
                        conn.commit()
                    st.success(f"Updated GPU {gpu_id} state to {new_state}")
                    del st.session_state.selected_gpu
                    st.rerun()
            
            with col2:
                if st.button("Remove GPU", key=f"remove_gpu_{gpu_id}", type="secondary"):
                    if selected_gpu['state'] == 'busy':
                        st.error("Cannot remove GPU with active jobs")
                    else:
                        with db.get_connection() as conn:
                            conn.execute("DELETE FROM gpus WHERE gpu_id = ?", (gpu_id,))
                            conn.commit()
                        st.success(f"Removed GPU {gpu_id}")
                        del st.session_state.selected_gpu
                        st.rerun()
            
            if st.button("Close", key=f"close_manage_{gpu_id}"):
                del st.session_state.selected_gpu
                st.rerun()
    
    else:
        st.info("No GPUs registered yet. Use the form above to register your first GPU.")
        
        # Demo data button
        if st.button("Create Demo GPUs", type="primary"):
            create_demo_gpus()
            st.rerun()

def render_jobs_tab():
    """Render job management tab"""
    st.header("🔄 Job Management")
    
    db = st.session_state.db_manager
    scheduler = st.session_state.scheduler
    
    # Job submission form
    with st.expander("Submit New Job", expanded=False):
        with st.form("submit_job_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                job_name = st.text_input("Job Name*", help="Descriptive name for this job")
                user_id = st.text_input("User ID*", help="User submitting the job")
                requested_gpus = st.number_input("Requested GPUs", min_value=1, max_value=8, value=1, help="Number of GPUs needed")
                memory_per_gpu = st.number_input("Memory per GPU (MB)", min_value=1024, value=8192, help="Memory required per GPU")
            
            with col2:
                priority = st.slider("Priority", min_value=1, max_value=10, value=5, help="Job priority (10 = highest)")
                duration = st.number_input("Expected Duration (minutes)", min_value=1, value=60, help="Expected job runtime")
                strategy = st.selectbox("Scheduling Strategy", options=[s.value for s in SchedulingStrategy], help="GPU allocation strategy")
                gpu_type = st.text_input("GPU Type Preference", placeholder="e.g., RTX 4090", help="Preferred GPU model (optional)")
            
            submitted = st.form_submit_button("Submit Job", type="primary")
            
            if submitted:
                if not all([job_name, user_id]):
                    st.error("Please fill in all required fields")
                else:
                    try:
                        job_id = str(uuid.uuid4())
                        now = datetime.utcnow().isoformat()
                        
                        job_request = {
                            'requested_gpus': requested_gpus,
                            'memory_per_gpu_mb': memory_per_gpu,
                            'gpu_type_preference': gpu_type if gpu_type else None
                        }
                        
                        # Try to schedule immediately
                        assigned_gpus = scheduler.schedule_job(job_request)
                        
                        with db.get_connection() as conn:
                            # Create job record
                            if assigned_gpus:
                                state = JobState.SCHEDULED.value
                                scheduled_at = now
                            else:
                                state = JobState.PENDING.value
                                scheduled_at = None
                            
                            conn.execute("""
                                INSERT INTO jobs (
                                    job_id, job_name, user_id, priority, requested_gpus,
                                    memory_per_gpu_mb, expected_duration_minutes,
                                    scheduling_strategy, gpu_type_preference,
                                    state, scheduled_at, created_at, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                job_id, job_name, user_id, priority, requested_gpus,
                                memory_per_gpu, duration, strategy, gpu_type,
                                state, scheduled_at, now, now
                            ))
                            
                            if assigned_gpus:
                                # Create assignments and update GPU states
                                for gpu_id in assigned_gpus:
                                    conn.execute("""
                                        INSERT INTO job_assignments (job_id, gpu_id, assigned_at)
                                        VALUES (?, ?, ?)
                                    """, (job_id, gpu_id, now))
                                    
                                    conn.execute("""
                                        UPDATE gpus 
                                        SET state = 'busy', memory_allocated_mb = memory_allocated_mb + ?
                                        WHERE gpu_id = ?
                                    """, (memory_per_gpu, gpu_id))
                            
                            conn.commit()
                            
                            if assigned_gpus:
                                st.success(f"Job '{job_name}' scheduled on GPUs: {assigned_gpus}")
                            else:
                                st.info(f"Job '{job_name}' queued - waiting for available resources")
                            
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error submitting job: {e}")

    # Job list and management
    with db.get_connection() as conn:
        jobs = conn.execute("""
            SELECT j.*, 
                   GROUP_CONCAT(ja.gpu_id) as assigned_gpu_ids
            FROM jobs j
            LEFT JOIN job_assignments ja ON j.job_id = ja.job_id AND ja.released_at IS NULL
            GROUP BY j.job_id
            ORDER BY j.created_at DESC
            LIMIT 50
        """).fetchall()
    
    if jobs:
        st.subheader("Job Queue")
        
        # State filter
        col1, col2 = st.columns([1, 3])
        with col1:
            job_state_filter = st.selectbox(
                "Filter by State",
                options=["All"] + [state.value for state in JobState],
                key="job_state_filter"
            )
        
        # Create jobs DataFrame
        job_data = []
        for job in jobs:
            job_dict = dict(job)
            if job_dict['assigned_gpu_ids']:
                job_dict['assigned_gpus'] = job_dict['assigned_gpu_ids'].split(',')
            else:
                job_dict['assigned_gpus'] = []
            job_data.append(job_dict)
        
        df_jobs = pd.DataFrame(job_data)
        
        if job_state_filter != "All":
            df_jobs = df_jobs[df_jobs['state'] == job_state_filter]
        
        # Display job cards
        for idx, job in df_jobs.iterrows():
            with st.container():
                state_colors = {
                    'pending': '🟡',
                    'scheduled': '🔵',
                    'running': '🟢',
                    'completed': '✅',
                    'failed': '❌',
                    'cancelled': '⏹️'
                }
                
                state_icon = state_colors.get(job['state'], '❓')
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"**{state_icon} {job['job_name']}**")
                    st.caption(f"ID: {job['job_id'][:8]}... | User: {job['user_id']}")
                
                with col2:
                    st.write(f"**State:** {job['state']}")
                    st.write(f"**Priority:** {job['priority']}")
                
                with col3:
                    st.write(f"**GPUs:** {job['requested_gpus']}")
                    if job['assigned_gpus']:
                        st.write(f"**Assigned:** {', '.join(job['assigned_gpus'])}")
                
                with col4:
                    # Job action buttons
                    if job['state'] in ['pending', 'scheduled', 'running']:
                        if st.button("Cancel", key=f"cancel_job_{job['job_id']}", type="secondary"):
                            cancel_job(job['job_id'])
                            st.rerun()
                    
                    if job['state'] == 'scheduled':
                        if st.button("Start", key=f"start_job_{job['job_id']}", type="primary"):
                            update_job_state(job['job_id'], JobState.RUNNING.value)
                            st.rerun()
                
                st.divider()
    
    else:
        st.info("No jobs in the system yet. Submit a job using the form above.")

def render_monitoring_tab():
    """Render monitoring and analytics tab"""
    st.header("📊 Monitoring & Analytics")
    
    db = st.session_state.db_manager
    
    # Time range selector
    col1, col2 = st.columns([1, 3])
    with col1:
        hours = st.selectbox("Time Range", options=[1, 6, 12, 24, 48, 168], index=3, format_func=lambda x: f"Last {x} hours")
    
    # Get metrics data
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    with db.get_connection() as conn:
        # GPU metrics history
        metrics_data = conn.execute("""
            SELECT gpu_id, timestamp, temperature_celsius, utilization_percent, 
                   memory_used_mb, power_watts
            FROM gpu_metrics_history
            WHERE timestamp > ?
            ORDER BY timestamp
        """, (cutoff_time.isoformat(),)).fetchall()
        
        # GPU info for labels
        gpu_info = conn.execute("""
            SELECT gpu_id, node_id, gpu_name FROM gpus
        """).fetchall()
        
        gpu_names = {gpu['gpu_id']: f"GPU {gpu['gpu_id']} ({gpu['gpu_name']})" for gpu in gpu_info}
    
    if metrics_data:
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'])
        df_metrics['gpu_label'] = df_metrics['gpu_id'].map(gpu_names)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("GPU Utilization Over Time")
            if not df_metrics.empty:
                fig_util = px.line(                    df_metrics, 
                    x='timestamp', 
                    y='utilization_percent', 
                    color='gpu_label',
                    title="GPU Utilization (%)",
                    labels={'utilization_percent': 'Utilization (%)', 'timestamp': 'Time'}
                )
                fig_util.update_layout(height=400)
                st.plotly_chart(fig_util, use_container_width=True)
        
        with col2:
            st.subheader("Temperature Monitoring")
            if not df_metrics.empty:
                fig_temp = px.line(
                    df_metrics, 
                    x='timestamp', 
                    y='temperature_celsius', 
                    color='gpu_label',
                    title="GPU Temperature (°C)",
                    labels={'temperature_celsius': 'Temperature (°C)', 'timestamp': 'Time'}
                )
                fig_temp.update_layout(height=400)
                st.plotly_chart(fig_temp, use_container_width=True)
        
        # Power consumption chart
        st.subheader("Power Consumption")
        if not df_metrics.empty and 'power_watts' in df_metrics.columns:
            fig_power = px.area(
                df_metrics, 
                x='timestamp', 
                y='power_watts', 
                color='gpu_label',
                title="Power Consumption Over Time (Watts)",
                labels={'power_watts': 'Power (W)', 'timestamp': 'Time'}
            )
            fig_power.update_layout(height=400)
            st.plotly_chart(fig_power, use_container_width=True)
        
        # Current status heatmap
        st.subheader("Current GPU Status Heatmap")
        with db.get_connection() as conn:
            current_gpus = conn.execute("""
                SELECT gpu_id, node_id, gpu_name, utilization_percent, 
                       temperature_celsius, state
                FROM gpus
                ORDER BY gpu_id
            """).fetchall()
        
        if current_gpus:
            df_current = pd.DataFrame(current_gpus)
            
            # Create heatmap data
            heatmap_data = df_current.pivot_table(
                index='gpu_name', 
                columns='node_id', 
                values='utilization_percent', 
                aggfunc='first'
            ).fillna(0)
            
            if not heatmap_data.empty:
                fig_heatmap = px.imshow(
                    heatmap_data,
                    title="GPU Utilization Heatmap",
                    labels={'color': 'Utilization (%)'},
                    color_continuous_scale='RdYlGn_r'
                )
                fig_heatmap.update_layout(height=300)
                st.plotly_chart(fig_heatmap, use_container_width=True)
    
    else:
        st.info("No metrics data available for the selected time range.")
        st.write("Metrics are collected when GPUs are active and reporting telemetry.")
    
    # Real-time cluster stats
    st.subheader("Real-time Cluster Statistics")
    
    metrics = get_cluster_metrics()
    
    # Create gauge charts
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_util_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = metrics['average_utilization'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Avg Utilization (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_util_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_util_gauge, use_container_width=True)
    
    with col2:
        fig_temp_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = metrics['average_temperature'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Avg Temperature (°C)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 75], 'color': "yellow"},
                    {'range': [75, 85], 'color': "orange"},
                    {'range': [85, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig_temp_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_temp_gauge, use_container_width=True)
    
    with col3:
        memory_usage_pct = (metrics['allocated_memory_mb'] / max(metrics['total_memory_mb'], 1)) * 100
        fig_memory_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = memory_usage_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Memory Usage (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "purple"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 90], 'color': "orange"},
                    {'range': [90, 100], 'color': "red"}
                ]
            }
        ))
        fig_memory_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_memory_gauge, use_container_width=True)

def render_events_tab():
    """Render events and logs tab"""
    st.header("📝 Events & Logs")
    
    db = st.session_state.db_manager
    
    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        hours = st.selectbox("Time Range", options=[1, 6, 12, 24, 48, 168], index=2, format_func=lambda x: f"Last {x} hours")
    with col2:
        severity_filter = st.selectbox("Severity", options=["All", "info", "warning", "error"])
    with col3:
        if st.button("Refresh Logs"):
            st.rerun()
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    with db.get_connection() as conn:
        query = "SELECT * FROM cluster_events WHERE timestamp > ?"
        params = [cutoff_time.isoformat()]
        
        if severity_filter != "All":
            query += " AND severity = ?"
            params.append(severity_filter)
        
        query += " ORDER BY timestamp DESC LIMIT 1000"
        
        events = conn.execute(query, params).fetchall()
    
    if events:
        # Event summary
        df_events = pd.DataFrame(events)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Events", len(events))
        with col2:
            error_count = len(df_events[df_events['severity'] == 'error'])
            st.metric("Errors", error_count)
        with col3:
            warning_count = len(df_events[df_events['severity'] == 'warning'])
            st.metric("Warnings", warning_count)
        
        # Event timeline
        if len(df_events) > 0:
            df_events['timestamp'] = pd.to_datetime(df_events['timestamp'])
            events_by_hour = df_events.set_index('timestamp').resample('1H').size().reset_index()
            events_by_hour.columns = ['hour', 'count']
            
            fig_timeline = px.bar(
                events_by_hour,
                x='hour',
                y='count',
                title="Events Over Time",
                labels={'count': 'Number of Events', 'hour': 'Hour'}
            )
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Event list
        st.subheader("Event Log")
        
        for event in events[:100]:  # Show last 100 events
            severity_colors = {
                'info': '🔵',
                'warning': '🟡', 
                'error': '🔴'
            }
            
            severity_icon = severity_colors.get(event['severity'], '⚪')
            timestamp = datetime.fromisoformat(event['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.write(f"{severity_icon} **{event['severity'].upper()}**")
                    st.caption(timestamp)
                
                with col2:
                    st.write(f"**{event['event_type']}** - {event['message']}")
                    if event['source']:
                        st.caption(f"Source: {event['source']}")
                
                st.divider()
    
    else:
        st.info("No events found for the selected criteria.")

def cancel_job(job_id: str):
    """Cancel a job and release its resources"""
    db = st.session_state.db_manager
    now = datetime.utcnow().isoformat()
    
    with db.get_connection() as conn:
        # Get job details
        job = conn.execute(
            "SELECT memory_per_gpu_mb FROM jobs WHERE job_id = ?", 
            (job_id,)
        ).fetchone()
        
        if not job:
            st.error("Job not found")
            return
        
        # Update job state
        conn.execute("""
            UPDATE jobs 
            SET state = 'cancelled', updated_at = ?
            WHERE job_id = ?
        """, (now, job_id))
        
        # Release GPU assignments
        gpu_assignments = conn.execute("""
            SELECT gpu_id FROM job_assignments
            WHERE job_id = ? AND released_at IS NULL
        """, (job_id,)).fetchall()
        
        for assignment in gpu_assignments:
            gpu_id = assignment['gpu_id']
            
            # Release assignment
            conn.execute("""
                UPDATE job_assignments 
                SET released_at = ? 
                WHERE job_id = ? AND gpu_id = ?
            """, (now, job_id, gpu_id))
            
            # Update GPU state
            conn.execute("""
                UPDATE gpus 
                SET state = 'available',
                    memory_allocated_mb = memory_allocated_mb - ?
                WHERE gpu_id = ?
            """, (job['memory_per_gpu_mb'], gpu_id))
        
        conn.commit()
        st.success(f"Job {job_id[:8]}... cancelled successfully")

def update_job_state(job_id: str, new_state: str):
    """Update job state"""
    db = st.session_state.db_manager
    now = datetime.utcnow().isoformat()
    
    with db.get_connection() as conn:
        update_fields = ["state = ?", "updated_at = ?"]
        update_values = [new_state, now]
        
        if new_state == JobState.RUNNING.value:
            update_fields.append("started_at = ?")
            update_values.append(now)
        elif new_state == JobState.COMPLETED.value:
            update_fields.append("completed_at = ?")
            update_values.append(now)
        
        update_values.append(job_id)
        conn.execute(
            f"UPDATE jobs SET {', '.join(update_fields)} WHERE job_id = ?",
            update_values
        )
        conn.commit()
        
        st.success(f"Job state updated to {new_state}")

def create_demo_gpus():
    """Create demo GPUs for testing"""
    db = st.session_state.db_manager
    
    demo_gpus = [
        {
            "node_id": f"demo-node-{i:02d}",
            "hostname": f"gpu-server-{i:02d}.cluster.local",
            "gpu_name": random.choice(["RTX 4090", "RTX 4080", "A100", "H100"]),
            "memory_total_mb": random.choice([24576, 16384, 40960, 80896]),
            "driver_version": "535.129.03",
            "location": f"rack-{i//4 + 1}",
            "tags": json.dumps(["demo", f"zone-{(i % 3) + 1}"])
        }
        for i in range(8)
    ]
    
    with db.get_connection() as conn:
        now = datetime.utcnow().isoformat()
        
        for gpu_data in demo_gpus:
            try:
                # Simulate some metrics
                gpu_data.update({
                    'utilization_percent': random.uniform(0, 100),
                    'temperature_celsius': random.uniform(30, 80),
                    'power_watts': random.uniform(50, 300),
                    'registered_at': now,
                    'last_heartbeat': now
                })
                
                conn.execute("""
                    INSERT OR IGNORE INTO gpus (
                        node_id, hostname, gpu_name, memory_total_mb, 
                        driver_version, location, tags,
                        utilization_percent, temperature_celsius, power_watts,
                        registered_at, last_heartbeat
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    gpu_data['node_id'], gpu_data['hostname'], gpu_data['gpu_name'],
                    gpu_data['memory_total_mb'], gpu_data['driver_version'], 
                    gpu_data['location'], gpu_data['tags'],
                    gpu_data['utilization_percent'], gpu_data['temperature_celsius'],
                    gpu_data['power_watts'], gpu_data['registered_at'], gpu_data['last_heartbeat']
                ))
            except Exception as e:
                continue  # Skip if already exists
        
        conn.commit()
    
    st.success("Demo GPUs created successfully!")

def create_demo_metrics():
    """Create demo metrics history for charts"""
    db = st.session_state.db_manager
    
    with db.get_connection() as conn:
        # Get all GPU IDs
        gpus = conn.execute("SELECT gpu_id FROM gpus").fetchall()
        
        if not gpus:
            return
        
        # Generate metrics for last 24 hours
        now = datetime.utcnow()
        
        for hours_back in range(24):
            timestamp = (now - timedelta(hours=hours_back)).isoformat()
            
            for gpu in gpus:
                gpu_id = gpu['gpu_id']
                
                # Generate realistic-looking metrics
                base_util = random.uniform(20, 80)
                util_variation = random.uniform(-10, 10)
                utilization = max(0, min(100, base_util + util_variation))
                
                temperature = random.uniform(45, 75)
                power = random.uniform(100, 250)
                memory_used = random.randint(2000, 20000)
                
                conn.execute("""
                    INSERT OR IGNORE INTO gpu_metrics_history 
                    (gpu_id, timestamp, temperature_celsius, utilization_percent, 
                     memory_used_mb, power_watts)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (gpu_id, timestamp, temperature, utilization, memory_used, power))
        
        conn.commit()

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="GPU Cluster Control Plane",
        page_icon="🖥️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize
    init_session_state()
    
    # Header
    st.title("🖥️ GPU Cluster Control Plane")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        
        tab_selection = st.radio(
            "Select View",
            options=["Overview", "GPU Management", "Job Management", "Monitoring", "Events & Logs"],
            index=0
        )
        
        st.markdown("---")
        
        # Quick actions
        st.header("Quick Actions")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("📊 Generate Demo Metrics", use_container_width=True):
            create_demo_metrics()
            st.success("Demo metrics generated!")
            st.rerun()
        
        if st.button("🧹 Clear All Data", use_container_width=True, type="secondary"):
            if st.session_state.get('confirm_clear'):
                # Clear database
                db = st.session_state.db_manager
                with db.get_connection() as conn:
                    conn.execute("DELETE FROM job_assignments")
                    conn.execute("DELETE FROM jobs")
                    conn.execute("DELETE FROM gpu_metrics_history")
                    conn.execute("DELETE FROM cluster_events")
                    conn.execute("DELETE FROM gpus")
                    conn.commit()
                st.success("All data cleared!")
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm deletion")
        
        # System info
        st.markdown("---")
        st.header("System Info")
        
        metrics = get_cluster_metrics()
        st.metric("Database", DB_FILE)
        st.metric("Total GPUs", metrics['total_gpus'])
        st.metric("Active Jobs", metrics['active_jobs'])
        
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    # Main content based on tab selection
    if tab_selection == "Overview":
        render_overview_tab()
    elif tab_selection == "GPU Management":
        render_gpus_tab()
    elif tab_selection == "Job Management":
        render_jobs_tab()
    elif tab_selection == "Monitoring":
        render_monitoring_tab()
    elif tab_selection == "Events & Logs":
        render_events_tab()

if __name__ == "__main__":
    main()
