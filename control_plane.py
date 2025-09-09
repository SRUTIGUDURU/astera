#!/usr/bin/env python3

import json
import sqlite3
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from contextlib import contextmanager, asynccontextmanager
import time
import random
from collections import defaultdict

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status, Query, Path as PathParam
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Configuration
DB_FILE = "gpu_cluster.db"
HEARTBEAT_TIMEOUT_SECONDS = 60
CLEANUP_INTERVAL_SECONDS = 30
METRICS_RETENTION_DAYS = 7

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    BEST_FIT = "best_fit"          # Minimize wasted resources
    FIRST_FIT = "first_fit"        # Fastest allocation
    LOAD_BALANCED = "load_balanced" # Distribute evenly
    POWER_EFFICIENT = "power_efficient" # Minimize power usage
    AFFINITY = "affinity"          # Keep jobs from same user together

# ==============================================================================
# Data Models
# ==============================================================================

class GPUCapabilities(BaseModel):
    compute_capability: Optional[str] = ""
    cuda_cores: Optional[int] = 0
    tensor_cores: Optional[int] = 0
    memory_bandwidth_gb: Optional[float] = 0.0
    pcie_gen: Optional[int] = 3
    supports_nvlink: Optional[bool] = False
    supports_mig: Optional[bool] = False  # Multi-Instance GPU

class RegisterGPURequest(BaseModel):
    node_id: str = Field(..., min_length=1, max_length=64)
    hostname: str
    gpu_name: str
    gpu_uuid: Optional[str] = ""
    vendor: str = "NVIDIA"
    memory_total_mb: int = Field(..., ge=1024)
    driver_version: str
    capabilities: Optional[GPUCapabilities] = None
    tags: List[str] = []
    location: Optional[str] = ""  # rack/datacenter location
    
    @validator('node_id')
    def validate_node_id(cls, v):
        if not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError('node_id must be alphanumeric with hyphens/underscores only')
        return v

class UpdateGPUMetricsRequest(BaseModel):
    temperature_celsius: Optional[float] = None
    utilization_percent: Optional[float] = None
    memory_used_mb: Optional[int] = None
    power_watts: Optional[float] = None
    fan_speed_percent: Optional[int] = None
    clock_speed_mhz: Optional[int] = None
    memory_clock_mhz: Optional[int] = None
    pcie_throughput_mb: Optional[float] = None
    ecc_errors: Optional[int] = None

class JobRequest(BaseModel):
    job_name: str
    user_id: str
    priority: int = Field(default=5, ge=1, le=10)
    requested_gpus: int = Field(default=1, ge=1)
    memory_per_gpu_mb: int = Field(..., ge=1)
    expected_duration_minutes: Optional[int] = None
    scheduling_strategy: SchedulingStrategy = SchedulingStrategy.BEST_FIT
    gpu_type_preference: Optional[str] = None
    require_nvlink: bool = False
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class JobResponse(BaseModel):
    job_id: str
    state: JobState
    assigned_gpus: List[int]
    scheduled_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: str = ""

class ClusterMetrics(BaseModel):
    timestamp: str
    total_gpus: int
    available_gpus: int
    busy_gpus: int
    offline_gpus: int
    total_memory_mb: int
    used_memory_mb: int
    average_utilization: float
    average_temperature: float
    total_power_watts: float
    active_jobs: int
    pending_jobs: int
    jobs_per_hour: float

# ==============================================================================
# Database Management
# ==============================================================================

class DatabaseManager:
    """Handles all database operations with connection pooling"""
    
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup"""
        conn = sqlite3.connect(self.db_file, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize all database tables"""
        with self.get_connection() as conn:
            # GPUs table with comprehensive tracking
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
                    
                    -- Real-time metrics
                    temperature_celsius REAL DEFAULT 0,
                    utilization_percent REAL DEFAULT 0,
                    power_watts REAL DEFAULT 0,
                    fan_speed_percent INTEGER DEFAULT 0,
                    clock_speed_mhz INTEGER DEFAULT 0,
                    memory_clock_mhz INTEGER DEFAULT 0,
                    pcie_throughput_mb REAL DEFAULT 0,
                    ecc_errors INTEGER DEFAULT 0,
                    
                    -- Capabilities
                    compute_capability TEXT,
                    cuda_cores INTEGER DEFAULT 0,
                    tensor_cores INTEGER DEFAULT 0,
                    memory_bandwidth_gb REAL DEFAULT 0,
                    pcie_gen INTEGER DEFAULT 3,
                    supports_nvlink BOOLEAN DEFAULT 0,
                    supports_mig BOOLEAN DEFAULT 0,
                    
                    -- Metadata
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
            
            # Job-GPU assignments
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
            
            # Cluster events log
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
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gpus_state ON gpus(state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gpus_heartbeat ON gpus(last_heartbeat)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assignments_job ON job_assignments(job_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assignments_gpu ON job_assignments(gpu_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON gpu_metrics_history(timestamp)")
            
            conn.commit()
            logger.info("Database initialized successfully")

# ==============================================================================
# GPU Scheduler
# ==============================================================================

class GPUScheduler:
    """Advanced GPU scheduling with multiple strategies"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def schedule_job(self, job_request: JobRequest) -> Optional[List[int]]:
        """Schedule a job using the specified strategy"""
        with self.db.get_connection() as conn:
            # Get available GPUs
            available_gpus = self._get_available_gpus(conn, job_request)
            
            if len(available_gpus) < job_request.requested_gpus:
                return None
            
            # Apply scheduling strategy
            if job_request.scheduling_strategy == SchedulingStrategy.BEST_FIT:
                selected = self._best_fit_scheduling(available_gpus, job_request)
            elif job_request.scheduling_strategy == SchedulingStrategy.FIRST_FIT:
                selected = self._first_fit_scheduling(available_gpus, job_request)
            elif job_request.scheduling_strategy == SchedulingStrategy.LOAD_BALANCED:
                selected = self._load_balanced_scheduling(available_gpus, job_request)
            elif job_request.scheduling_strategy == SchedulingStrategy.POWER_EFFICIENT:
                selected = self._power_efficient_scheduling(available_gpus, job_request)
            else:  # AFFINITY
                selected = self._affinity_scheduling(available_gpus, job_request, conn)
            
            return selected
    
    def _get_available_gpus(self, conn, job_request: JobRequest) -> List[Dict]:
        """Get GPUs that meet job requirements"""
        query = """
            SELECT * FROM gpus 
            WHERE state = 'available' 
            AND (memory_total_mb - memory_allocated_mb) >= ?
            AND last_heartbeat > datetime('now', '-60 seconds')
        """
        params = [job_request.memory_per_gpu_mb]
        
        if job_request.gpu_type_preference:
            query += " AND gpu_name LIKE ?"
            params.append(f"%{job_request.gpu_type_preference}%")
        
        if job_request.require_nvlink:
            query += " AND supports_nvlink = 1"
        
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def _best_fit_scheduling(self, gpus: List[Dict], job: JobRequest) -> List[int]:
        """Select GPUs that minimize wasted resources"""
        # Sort by closest fit to requested memory
        sorted_gpus = sorted(
            gpus,
            key=lambda g: (g['memory_total_mb'] - g['memory_allocated_mb']) - job.memory_per_gpu_mb
        )
        return [g['gpu_id'] for g in sorted_gpus[:job.requested_gpus]]
    
    def _first_fit_scheduling(self, gpus: List[Dict], job: JobRequest) -> List[int]:
        """Select first available GPUs (fastest allocation)"""
        return [g['gpu_id'] for g in gpus[:job.requested_gpus]]
    
    def _load_balanced_scheduling(self, gpus: List[Dict], job: JobRequest) -> List[int]:
        """Distribute load evenly across GPUs"""
        # Sort by utilization and jobs completed
        sorted_gpus = sorted(
            gpus,
            key=lambda g: (g['utilization_percent'], g['total_jobs_completed'])
        )
        return [g['gpu_id'] for g in sorted_gpus[:job.requested_gpus]]
    
    def _power_efficient_scheduling(self, gpus: List[Dict], job: JobRequest) -> List[int]:
        """Select GPUs to minimize power consumption"""
        # Prefer GPUs already running (avoid cold start) and lower power models
        sorted_gpus = sorted(
            gpus,
            key=lambda g: (
                0 if g['utilization_percent'] > 0 else 1,  # Prefer already active
                g['power_watts'],  # Then lower power
                -g['memory_bandwidth_gb']  # But higher performance
            )
        )
        return [g['gpu_id'] for g in sorted_gpus[:job.requested_gpus]]
    
    def _affinity_scheduling(self, gpus: List[Dict], job: JobRequest, conn) -> List[int]:
        """Try to schedule jobs from same user on nearby GPUs"""
        # Get GPUs with existing jobs from same user
        cursor = conn.execute("""
            SELECT DISTINCT g.gpu_id, g.node_id, g.location
            FROM gpus g
            JOIN job_assignments ja ON g.gpu_id = ja.gpu_id
            JOIN jobs j ON ja.job_id = j.job_id
            WHERE j.user_id = ? AND j.state = 'running'
        """, (job.user_id,))
        
        user_nodes = {row['node_id'] for row in cursor.fetchall()}
        
        if user_nodes:
            # Prefer GPUs on same nodes
            gpus_sorted = sorted(
                gpus,
                key=lambda g: (0 if g['node_id'] in user_nodes else 1, g['utilization_percent'])
            )
            return [g['gpu_id'] for g in gpus_sorted[:job.requested_gpus]]
        
        # Fallback to load balanced
        return self._load_balanced_scheduling(gpus, job)

# ==============================================================================
# Cluster Monitor
# ==============================================================================

class ClusterMonitor:
    """Monitors cluster health and handles failures"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.websocket_clients: Set[WebSocket] = set()
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._cleanup_old_data())
    
    async def _heartbeat_monitor(self):
        """Check GPU heartbeats and mark offline nodes"""
        while True:
            try:
                with self.db.get_connection() as conn:
                    # Find GPUs with stale heartbeats
                    timeout_threshold = datetime.utcnow() - timedelta(seconds=HEARTBEAT_TIMEOUT_SECONDS)
                    
                    # SQLite doesn't support RETURNING in UPDATE, so we need to do this in two steps
                    cursor = conn.execute("""
                        SELECT gpu_id, node_id FROM gpus 
                        WHERE state != 'offline' 
                        AND last_heartbeat < ?
                    """, (timeout_threshold.isoformat(),))
                    
                    offline_gpus = cursor.fetchall()
                    
                    if offline_gpus:
                        # Update state to offline
                        gpu_ids = [gpu['gpu_id'] for gpu in offline_gpus]
                        placeholders = ','.join('?' * len(gpu_ids))
                        conn.execute(f"""
                            UPDATE gpus 
                            SET state = 'offline' 
                            WHERE gpu_id IN ({placeholders})
                        """, gpu_ids)
                        
                        for gpu in offline_gpus:
                            await self._log_event(
                                "gpu_offline",
                                "warning",
                                f"gpu_{gpu['gpu_id']}",
                                f"GPU {gpu['gpu_id']} on node {gpu['node_id']} went offline"
                            )
                            # TODO: Reschedule affected jobs
                    
                    conn.commit()
            except Exception as e:
                logger.error(f"Heartbeat monitor error: {e}")
            
            await asyncio.sleep(10)
    
    async def _metrics_collector(self):
        """Collect and store cluster-wide metrics"""
        while True:
            try:
                metrics = self.get_cluster_metrics()
                await self._broadcast_metrics(metrics)
                
                # Store metrics history
                with self.db.get_connection() as conn:
                    # Sample GPU metrics
                    conn.execute("""
                        INSERT INTO gpu_metrics_history 
                        (gpu_id, timestamp, temperature_celsius, utilization_percent, 
                         memory_used_mb, power_watts)
                        SELECT gpu_id, ?, temperature_celsius, utilization_percent,
                               memory_used_mb, power_watts
                        FROM gpus
                        WHERE state = 'busy'
                    """, (metrics.timestamp,))
                    conn.commit()
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
            
            await asyncio.sleep(30)
    
    async def _cleanup_old_data(self):
        """Clean up old metrics and completed jobs"""
        while True:
            try:
                with self.db.get_connection() as conn:
                    cutoff_date = datetime.utcnow() - timedelta(days=METRICS_RETENTION_DAYS)
                    
                    # Clean old metrics
                    conn.execute(
                        "DELETE FROM gpu_metrics_history WHERE timestamp < ?",
                        (cutoff_date.isoformat(),)
                    )
                    
                    # Archive old jobs
                    conn.execute(
                        "DELETE FROM jobs WHERE state IN ('completed', 'failed', 'cancelled') AND updated_at < ?",
                        (cutoff_date.isoformat(),)
                    )
                    
                    conn.commit()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            
            await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
    
    def get_cluster_metrics(self) -> ClusterMetrics:
        """Get current cluster-wide metrics"""
        with self.db.get_connection() as conn:
            # GPU statistics
            gpu_stats = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN state = 'available' THEN 1 ELSE 0 END) as available,
                    SUM(CASE WHEN state = 'busy' THEN 1 ELSE 0 END) as busy,
                    SUM(CASE WHEN state = 'offline' THEN 1 ELSE 0 END) as offline,
                    SUM(memory_total_mb) as total_memory,
                    SUM(memory_allocated_mb) as allocated_memory,
                    AVG(CASE WHEN state = 'busy' THEN utilization_percent ELSE NULL END) as avg_util,
                    AVG(CASE WHEN state = 'busy' THEN temperature_celsius ELSE NULL END) as avg_temp,
                    SUM(CASE WHEN state = 'busy' THEN power_watts ELSE 0 END) as total_power
                FROM gpus
            """).fetchone()
            
            # Job statistics
            job_stats = conn.execute("""
                SELECT 
                    SUM(CASE WHEN state = 'running' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN state = 'pending' THEN 1 ELSE 0 END) as pending
                FROM jobs
            """).fetchone()
            
            # Jobs per hour (last 24h)
            jobs_per_hour = conn.execute("""
                SELECT COUNT(*) / 24.0 as rate
                FROM jobs
                WHERE created_at > datetime('now', '-24 hours')
            """).fetchone()['rate']
            
            return ClusterMetrics(
                timestamp=datetime.utcnow().isoformat(),
                total_gpus=gpu_stats['total'] or 0,
                available_gpus=gpu_stats['available'] or 0,
                busy_gpus=gpu_stats['busy'] or 0,
                offline_gpus=gpu_stats['offline'] or 0,
                total_memory_mb=gpu_stats['total_memory'] or 0,
                used_memory_mb=gpu_stats['allocated_memory'] or 0,
                average_utilization=gpu_stats['avg_util'] or 0.0,
                average_temperature=gpu_stats['avg_temp'] or 0.0,
                total_power_watts=gpu_stats['total_power'] or 0.0,
                active_jobs=job_stats['active'] or 0,
                pending_jobs=job_stats['pending'] or 0,
                jobs_per_hour=jobs_per_hour or 0.0
            )
    
    async def _log_event(self, event_type: str, severity: str, source: str, message: str, metadata: Dict = None):
        """Log cluster events"""
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT INTO cluster_events 
                (timestamp, event_type, severity, source, message, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.utcnow().isoformat(),
                event_type,
                severity,
                source,
                message,
                json.dumps(metadata or {})
            ))
            conn.commit()
    
    async def _broadcast_metrics(self, metrics: ClusterMetrics):
        """Broadcast metrics to all connected WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message = json.dumps({
            "type": "metrics_update",
            "data": metrics.dict()
        })
        
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send_text(message)
            except:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected

# ==============================================================================
# FastAPI Application
# ==============================================================================

app = FastAPI(
    title="GPU Cluster Control Plane",
    description="Advanced GPU cluster management with scheduling, monitoring, and job management",
    version="2.0.0"
)

# Initialize components
db_manager = DatabaseManager(DB_FILE)
scheduler = GPUScheduler(db_manager)
monitor = ClusterMonitor(db_manager)

# Startup event
@app.on_event("startup")
async def startup_event():
    await monitor.start_monitoring()
    logger.info("GPU Cluster Control Plane started")

# ==============================================================================
# GPU Management Endpoints
# ==============================================================================

@app.post("/register", response_model=Dict[str, Any], tags=["GPU Management"])
async def register_gpu(request: RegisterGPURequest):
    """Register a new GPU node in the cluster"""
    try:
        with db_manager.get_connection() as conn:
            # Check if node already exists
            existing = conn.execute(
                "SELECT gpu_id FROM gpus WHERE node_id = ?",
                (request.node_id,)
            ).fetchone()
            
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Node {request.node_id} already registered"
                )
            
            # Insert new GPU
            now = datetime.utcnow().isoformat()
            cursor = conn.execute("""
                INSERT INTO gpus (
                    node_id, hostname, gpu_name, gpu_uuid, vendor,
                    memory_total_mb, driver_version, location, tags,
                    compute_capability, cuda_cores, tensor_cores,
                    memory_bandwidth_gb, pcie_gen, supports_nvlink, supports_mig,
                    registered_at, last_heartbeat
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.node_id, request.hostname, request.gpu_name,
                request.gpu_uuid, request.vendor, request.memory_total_mb,
                request.driver_version, request.location, json.dumps(request.tags),
                request.capabilities.compute_capability if request.capabilities else "",
                request.capabilities.cuda_cores if request.capabilities else 0,
                request.capabilities.tensor_cores if request.capabilities else 0,
                request.capabilities.memory_bandwidth_gb if request.capabilities else 0,
                request.capabilities.pcie_gen if request.capabilities else 3,
                request.capabilities.supports_nvlink if request.capabilities else False,
                request.capabilities.supports_mig if request.capabilities else False,
                now, now
            ))
            conn.commit()
            
            gpu_id = cursor.lastrowid
            
            await monitor._log_event(
                "gpu_registered",
                "info",
                f"gpu_{gpu_id}",
                f"GPU {request.gpu_name} registered on node {request.node_id}"
            )
            
            return {
                "gpu_id": gpu_id,
                "status": "registered",
                "message": f"GPU {request.gpu_name} successfully registered"
            }
            
    except sqlite3.Error as e:
        logger.error(f"Database error during GPU registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register GPU"
        )

@app.delete("/gpu/{node_id}", tags=["GPU Management"])
async def deregister_gpu(node_id: str = PathParam(..., description="Node ID to deregister")):
    """Remove a GPU from the cluster"""
    with db_manager.get_connection() as conn:
        # Check if GPU exists and has active jobs
        gpu = conn.execute(
            "SELECT gpu_id, state FROM gpus WHERE node_id = ?",
            (node_id,)
        ).fetchone()
        
        if not gpu:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"GPU node {node_id} not found"
            )
        
        if gpu['state'] == 'busy':
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot remove GPU with active jobs"
            )
        
        # Remove GPU
        conn.execute("DELETE FROM gpus WHERE node_id = ?", (node_id,))
        conn.commit()
        
        await monitor._log_event(
            "gpu_deregistered",
            "info",
            f"gpu_{gpu['gpu_id']}",
            f"GPU on node {node_id} removed from cluster"
        )
        
        return {"message": f"GPU node {node_id} successfully removed"}

@app.patch("/gpu/{node_id}/metrics", tags=["GPU Management"])
async def update_gpu_metrics(
    node_id: str,
    metrics: UpdateGPUMetricsRequest
):
    """Update real-time GPU metrics (called by GPU nodes)"""
    with db_manager.get_connection() as conn:
        # Build dynamic update query
        updates = []
        values = []
        
        for field, value in metrics.dict(exclude_none=True).items():
            updates.append(f"{field} = ?")
            values.append(value)
        
        if not updates:
            return {"message": "No metrics to update"}
        
        # Add heartbeat update
        updates.append("last_heartbeat = ?")
        values.append(datetime.utcnow().isoformat())
        
        # Update GPU metrics
        values.append(node_id)
        result = conn.execute(
            f"UPDATE gpus SET {', '.join(updates)} WHERE node_id = ?",
            values
        )
        
        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"GPU node {node_id} not found"
            )
        
        conn.commit()
        
        return {"message": "Metrics updated successfully"}

@app.patch("/gpu/{node_id}/state", tags=["GPU Management"])
async def update_gpu_state(
    node_id: str,
    state: GPUState
):
    """Manually set GPU state (for maintenance, etc.)"""
    with db_manager.get_connection() as conn:
        result = conn.execute(
            "UPDATE gpus SET state = ? WHERE node_id = ?",
            (state.value, node_id)
        )
        
        if result.rowcount == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"GPU node {node_id} not found"
            )
        
        conn.commit()
        
        await monitor._log_event(
            "gpu_state_change",
            "info",
            f"node_{node_id}",
            f"GPU state changed to {state.value}"
        )
        
        return {"message": f"GPU state updated to {state.value}"}

# ==============================================================================
# Cluster Status Endpoints
# ==============================================================================

@app.get("/status", response_model=Dict[str, Any], tags=["Cluster Status"])
async def get_cluster_status(
    format: str = Query("summary", pattern="^(summary|detailed|json)$"),
    include_metrics: bool = Query(False, description="Include performance metrics")
):
    """Get comprehensive cluster health overview"""
    with db_manager.get_connection() as conn:
        # Get all GPUs
        gpus = conn.execute("""
            SELECT gpu_id, node_id, hostname, gpu_name, vendor, state,
                   memory_total_mb, memory_allocated_mb, memory_used_mb,
                   temperature_celsius, utilization_percent, power_watts,
                   last_heartbeat, location, tags
            FROM gpus
            ORDER BY gpu_id
        """).fetchall()
        
        # Get running jobs
        jobs = conn.execute("""
            SELECT j.job_id, j.job_name, j.user_id, j.state, 
                   COUNT(ja.gpu_id) as gpu_count
            FROM jobs j
            LEFT JOIN job_assignments ja ON j.job_id = ja.job_id
            WHERE j.state IN ('running', 'scheduled')
            GROUP BY j.job_id
        """).fetchall()
        
        # Get cluster metrics
        metrics = monitor.get_cluster_metrics() if include_metrics else None
        
        if format == "summary":
            # Human-readable summary
            lines = [
                f"=== GPU CLUSTER STATUS ===",
                f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"",
                f"GPUs: {metrics.available_gpus}/{metrics.total_gpus} available" if metrics else "GPUs: N/A",
                f"Memory: {metrics.used_memory_mb:,}/{metrics.total_memory_mb:,} MB used" if metrics else "Memory: N/A",
                f"Power: {metrics.total_power_watts:.1f}W" if metrics else "Power: N/A",
                f"Jobs: {metrics.active_jobs} running, {metrics.pending_jobs} pending" if metrics else "Jobs: N/A",
                f"",
                f"GPU Details:"
            ]
            
            for gpu in gpus:
                state_icon = {
                    'available': '✓',
                    'busy': '●',
                    'offline': '✗',
                    'maintenance': '🔧',
                    'error': '⚠'
                }.get(gpu['state'], '?')
                
                mem_used_pct = (gpu['memory_allocated_mb'] / gpu['memory_total_mb'] * 100) if gpu['memory_total_mb'] > 0 else 0
                
                lines.append(
                    f"  [{state_icon}] GPU {gpu['gpu_id']:2d}: {gpu['gpu_name']:20s} "
                    f"| {gpu['utilization_percent']:3.0f}% util "
                    f"| {gpu['temperature_celsius']:3.0f}°C "
                    f"| {gpu['memory_allocated_mb']:5d}/{gpu['memory_total_mb']:5d} MB ({mem_used_pct:3.0f}%)"
                )
            
            return {"status": "\n".join(lines)}
            
        elif format == "detailed":
            # Detailed view with all information
            gpu_details = []
            for gpu in gpus:
                gpu_dict = dict(gpu)
                gpu_dict['tags'] = json.loads(gpu_dict.get('tags', '[]'))
                
                # Get jobs on this GPU
                gpu_jobs = conn.execute("""
                    SELECT j.job_id, j.job_name, j.user_id
                    FROM job_assignments ja
                    JOIN jobs j ON ja.job_id = j.job_id
                    WHERE ja.gpu_id = ? AND ja.released_at IS NULL
                """, (gpu['gpu_id'],)).fetchall()
                
                gpu_dict['active_jobs'] = [dict(j) for j in gpu_jobs]
                gpu_details.append(gpu_dict)
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "cluster_metrics": metrics.dict() if metrics else None,
                "gpus": gpu_details,
                "active_jobs": [dict(j) for j in jobs]
            }
            
        else:  # json format
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics.dict() if metrics else None,
                "gpus": [dict(g) for g in gpus],
                "jobs": [dict(j) for j in jobs]
            }

@app.get("/status/gpu/{node_id}", tags=["Cluster Status"])
async def get_gpu_status(node_id: str):
    """Get detailed status for a specific GPU"""
    with db_manager.get_connection() as conn:
        gpu = conn.execute(
            "SELECT * FROM gpus WHERE node_id = ?",
            (node_id,)
        ).fetchone()
        
        if not gpu:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"GPU node {node_id} not found"
            )
        
        gpu_dict = dict(gpu)
        gpu_dict['tags'] = json.loads(gpu_dict.get('tags', '[]'))
        
        # Get current jobs
        jobs = conn.execute("""
            SELECT j.* FROM jobs j
            JOIN job_assignments ja ON j.job_id = ja.job_id
            WHERE ja.gpu_id = ? AND ja.released_at IS NULL
        """, (gpu['gpu_id'],)).fetchall()
        
        gpu_dict['current_jobs'] = [dict(j) for j in jobs]
        
        # Get recent metrics
        metrics = conn.execute("""
            SELECT timestamp, temperature_celsius, utilization_percent, 
                   memory_used_mb, power_watts
            FROM gpu_metrics_history
            WHERE gpu_id = ?
            ORDER BY timestamp DESC
            LIMIT 100
        """, (gpu['gpu_id'],)).fetchall()
        
        gpu_dict['recent_metrics'] = [dict(m) for m in metrics]
        
        return gpu_dict

@app.get("/metrics", response_model=ClusterMetrics, tags=["Cluster Status"])
async def get_cluster_metrics():
    """Get current cluster-wide metrics"""
    return monitor.get_cluster_metrics()

@app.get("/metrics/history", tags=["Cluster Status"])
async def get_metrics_history(
    hours: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    gpu_id: Optional[int] = Query(None, description="Filter by specific GPU")
):
    """Get historical metrics data"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    with db_manager.get_connection() as conn:
        query = """
            SELECT gpu_id, timestamp, temperature_celsius, 
                   utilization_percent, memory_used_mb, power_watts
            FROM gpu_metrics_history
            WHERE timestamp > ?
        """
        params = [cutoff_time.isoformat()]
        
        if gpu_id is not None:
            query += " AND gpu_id = ?"
            params.append(gpu_id)
        
        query += " ORDER BY timestamp DESC"
        
        metrics = conn.execute(query, params).fetchall()
        
        return {
            "start_time": cutoff_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "metrics": [dict(m) for m in metrics]
        }

@app.get("/events", tags=["Cluster Status"])
async def get_cluster_events(
    hours: int = Query(24, ge=1, le=168),
    severity: Optional[str] = Query(None, pattern="^(info|warning|error)$"),
    event_type: Optional[str] = None
):
    """Get cluster event log"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    with db_manager.get_connection() as conn:
        query = "SELECT * FROM cluster_events WHERE timestamp > ?"
        params = [cutoff_time.isoformat()]
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        query += " ORDER BY timestamp DESC LIMIT 1000"
        
        events = conn.execute(query, params).fetchall()
        
        return {
            "events": [
                {
                    **dict(e),
                    "metadata": json.loads(e['metadata'])
                } for e in events
            ]
        }

# ==============================================================================
# Job Management Endpoints
# ==============================================================================

@app.post("/submit_job", response_model=JobResponse, tags=["Job Management"])
async def submit_job(request: JobRequest):
    """Submit a new job to the cluster"""
    job_id = str(uuid.uuid4())
    now = datetime.isoformat()
    
    with db_manager.get_connection() as conn:
        # Create job record
        conn.execute("""
            INSERT INTO jobs (
                job_id, job_name, user_id, priority, requested_gpus,
                memory_per_gpu_mb, expected_duration_minutes,
                scheduling_strategy, gpu_type_preference, require_nvlink,
                tags, metadata, state, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, request.job_name, request.user_id, request.priority,
            request.requested_gpus, request.memory_per_gpu_mb,
            request.expected_duration_minutes, request.scheduling_strategy.value,
            request.gpu_type_preference, request.require_nvlink,
            json.dumps(request.tags), json.dumps(request.metadata),
            JobState.PENDING.value, now, now
        ))
        
        # Try to schedule immediately
        assigned_gpus = scheduler.schedule_job(request)
        
        if assigned_gpus:
            # Update job state
            conn.execute("""
                UPDATE jobs 
                SET state = ?, scheduled_at = ?, updated_at = ?
                WHERE job_id = ?
            """, (JobState.SCHEDULED.value, now, now, job_id))
            
            # Create GPU assignments
            for gpu_id in assigned_gpus:
                conn.execute("""
                    INSERT INTO job_assignments (job_id, gpu_id, assigned_at)
                    VALUES (?, ?, ?)
                """, (job_id, gpu_id, now))
                
                # Update GPU state and allocated memory
                conn.execute("""
                    UPDATE gpus 
                    SET state = 'busy', 
                        memory_allocated_mb = memory_allocated_mb + ?
                    WHERE gpu_id = ?
                """, (request.memory_per_gpu_mb, gpu_id))
            
            conn.commit()
            
            await monitor._log_event(
                "job_scheduled",
                "info",
                f"job_{job_id}",
                f"Job {request.job_name} scheduled on GPUs {assigned_gpus}",
                {"user_id": request.user_id, "gpus": assigned_gpus}
            )
            
            return JobResponse(
                job_id=job_id,
                state=JobState.SCHEDULED,
                assigned_gpus=assigned_gpus,
                scheduled_at=now,
                message=f"Job scheduled on GPUs: {assigned_gpus}"
            )
        else:
            # Job queued
            conn.commit()
            
            await monitor._log_event(
                "job_queued",
                "info",
                f"job_{job_id}",
                f"Job {request.job_name} queued - no resources available",
                {"user_id": request.user_id}
            )
            
            return JobResponse(
                job_id=job_id,
                state=JobState.PENDING,
                assigned_gpus=[],
                message="Job queued - waiting for resources"
            )

@app.get("/jobs", tags=["Job Management"])
async def list_jobs(
    state: Optional[JobState] = None,
    user_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """List jobs with optional filtering"""
    with db_manager.get_connection() as conn:
        query = "SELECT * FROM jobs WHERE 1=1"
        params = []
        
        if state:
            query += " AND state = ?"
            params.append(state.value)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        jobs = conn.execute(query, params).fetchall()
        
        results = []
        for job in jobs:
            job_dict = dict(job)
            job_dict['tags'] = json.loads(job_dict.get('tags', '[]'))
            job_dict['metadata'] = json.loads(job_dict.get('metadata', '{}'))
            
            # Get assigned GPUs
            gpus = conn.execute("""
                SELECT gpu_id FROM job_assignments
                WHERE job_id = ? AND released_at IS NULL
            """, (job['job_id'],)).fetchall()
            
            job_dict['assigned_gpus'] = [g['gpu_id'] for g in gpus]
            results.append(job_dict)
        
        return {"jobs": results}

@app.get("/jobs/{job_id}", tags=["Job Management"])
async def get_job_status(job_id: str):
    """Get detailed status for a specific job"""
    with db_manager.get_connection() as conn:
        job = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        job_dict = dict(job)
        job_dict['tags'] = json.loads(job_dict.get('tags', '[]'))
        job_dict['metadata'] = json.loads(job_dict.get('metadata', '{}'))
        
        # Get GPU assignments with details
        assignments = conn.execute("""
            SELECT ja.*, g.node_id, g.gpu_name, g.hostname
            FROM job_assignments ja
            JOIN gpus g ON ja.gpu_id = g.gpu_id
            WHERE ja.job_id = ?
        """, (job_id,)).fetchall()
        
        job_dict['gpu_assignments'] = [dict(a) for a in assignments]
        
        return job_dict

@app.patch("/jobs/{job_id}/state", tags=["Job Management"])
async def update_job_state(
    job_id: str,
    state: JobState,
    error_message: Optional[str] = None
):
    """Update job state (for job completion, failure, etc.)"""
    now = datetime.utcnow().isoformat()
    
    with db_manager.get_connection() as conn:
        # Get current job
        job = conn.execute(
            "SELECT state, memory_per_gpu_mb FROM jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Update job state
        update_fields = ["state = ?", "updated_at = ?"]
        update_values = [state.value, now]
        
        if state == JobState.RUNNING:
            update_fields.append("started_at = ?")
            update_values.append(now)
        elif state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
            update_fields.append(f"{'completed_at' if state == JobState.COMPLETED else 'failed_at'} = ?")
            update_values.append(now)
            
            if error_message and state == JobState.FAILED:
                update_fields.append("error_message = ?")
                update_values.append(error_message)
        
        update_values.append(job_id)
        conn.execute(
            f"UPDATE jobs SET {', '.join(update_fields)} WHERE job_id = ?",
            update_values
        )
        
        # Release GPUs if job is ending
        if state in [JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED]:
            # Get assigned GPUs
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
                        memory_allocated_mb = memory_allocated_mb - ?,
                        total_jobs_completed = total_jobs_completed + 1
                    WHERE gpu_id = ?
                """, (job['memory_per_gpu_mb'], gpu_id))
        
        conn.commit()
        
        await monitor._log_event(
            f"job_{state.value.lower()}",
            "info" if state != JobState.FAILED else "warning",
            f"job_{job_id}",
            f"Job {job_id} state changed to {state.value}",
            {"error": error_message} if error_message else None
        )
        
        return {"message": f"Job state updated to {state.value}"}

@app.delete("/jobs/{job_id}", tags=["Job Management"])
async def cancel_job(job_id: str):
    """Cancel a pending or running job"""
    return await update_job_state(job_id, JobState.CANCELLED)

# ==============================================================================
# WebSocket Support
# ==============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    monitor.websocket_clients.add(websocket)
    
    try:
        # Send initial cluster state
        metrics = monitor.get_cluster_metrics()
        await websocket.send_json({
            "type": "connection",
            "data": {
                "message": "Connected to GPU cluster control plane",
                "metrics": metrics.dict()
            }
        })
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (ping/pong or commands)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                try:
                    message = json.loads(data)
                    
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message.get("type") == "subscribe":
                        # Client can subscribe to specific events
                        await websocket.send_json({
                            "type": "subscribed",
                            "data": {"message": "Subscription confirmed"}
                        })
                        
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "data": {"message": "Invalid JSON"}
                    })
                    
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        monitor.websocket_clients.discard(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        monitor.websocket_clients.discard(websocket)

# ==============================================================================
# Health and Debug Endpoints
# ==============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Service health check"""
    try:
        with db_manager.get_connection() as conn:
            # Check database connectivity
            conn.execute("SELECT 1").fetchone()
            
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": app.version
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/debug/scheduler", tags=["Debug"])
async def debug_scheduler():
    """Debug scheduler state"""
    with db_manager.get_connection() as conn:
        # Get pending jobs
        pending_jobs = conn.execute("""
            SELECT job_id, job_name, requested_gpus, memory_per_gpu_mb
            FROM jobs
            WHERE state = 'pending'
            ORDER BY priority DESC, created_at
        """).fetchall()
        
        # Get available GPU capacity
        available_gpus = conn.execute("""
            SELECT gpu_id, node_id, gpu_name, 
                   memory_total_mb - memory_allocated_mb as free_memory_mb
            FROM gpus
            WHERE state = 'available'
            ORDER BY free_memory_mb DESC
        """).fetchall()
        
        return {
            "pending_jobs": [dict(j) for j in pending_jobs],
            "available_gpus": [dict(g) for g in available_gpus],
            "scheduler_info": {
                "strategies": [s.value for s in SchedulingStrategy],
                "heartbeat_timeout": HEARTBEAT_TIMEOUT_SECONDS
            }
        }

# ==============================================================================
# Demo/Test Endpoints
# ==============================================================================

@app.post("/demo/populate", tags=["Demo"])
async def populate_demo_data():
    """Populate cluster with demo GPUs for testing"""
    demo_gpus = [
        {
            "node_id": f"demo-node-{i:02d}",
            "hostname": f"gpu-server-{i:02d}.cluster.local",
            "gpu_name": random.choice(["RTX 4090", "RTX 4080", "A100", "H100"]),
            "memory_total_mb": random.choice([24576, 16384, 40960, 80896]),
            "driver_version": "535.129.03",
            "location": f"rack-{i//4 + 1}",
            "tags": ["demo", f"zone-{(i % 3) + 1}"]
        }
        for i in range(8)
    ]
    
    created = []
    for gpu_data in demo_gpus:
        request = RegisterGPURequest(**gpu_data)
        try:
            result = await register_gpu(request)
            created.append(result)
        except HTTPException:
            pass  # Skip if already exists
    
    return {
        "message": f"Created {len(created)} demo GPUs",
        "gpus": created
    }

# ==============================================================================
# HTML Dashboard
# ==============================================================================

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    """Simple web dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GPU Cluster Control Plane</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #dee2e6; }
            .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .metric-label { color: #6c757d; font-size: 14px; }
            .status { margin: 20px 0; }
            .gpu-list { margin-top: 20px; }
            .gpu-item { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; display: flex; justify-content: space-between; align-items: center; }
            .gpu-available { border-left: 4px solid #28a745; }
            .gpu-busy { border-left: 4px solid #ffc107; }
            .gpu-offline { border-left: 4px solid #dc3545; }
            .actions { margin: 20px 0; }
            button { background: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-right: 10px; }
            button:hover { background: #0056b3; }
            #log { background: #f8f9fa; padding: 10px; border-radius: 5px; height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>GPU Cluster Control Plane</h1>
            
            <div class="metrics" id="metrics">
                <div class="metric-card">
                    <div class="metric-label">Total GPUs</div>
                    <div class="metric-value" id="total-gpus">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Available GPUs</div>
                    <div class="metric-value" id="available-gpus">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Active Jobs</div>
                    <div class="metric-value" id="active-jobs">-</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Average Utilization</div>
                    <div class="metric-value" id="avg-util">-</div>
                </div>
            </div>
            
            <div class="actions">
                <button onclick="refreshStatus()">Refresh</button>
                <button onclick="populateDemo()">Populate Demo GPUs</button>
                <button onclick="clearLog()">Clear Log</button>
            </div>
            
            <div class="status">
                <h2>GPU Status</h2>
                <div class="gpu-list" id="gpu-list">
                    Loading...
                </div>
            </div>
            
            <div class="status">
                <h2>Real-time Log</h2>
                <div id="log"></div>
            </div>
        </div>
        
        <script>
            let ws = null;
            
            function connectWebSocket() {
                ws = new WebSocket('ws://localhost:8000/ws');
                
                ws.onopen = () => {
                    log('Connected to control plane');
                };
                
                ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    
                    if (message.type === 'metrics_update') {
                        updateMetrics(message.data);
                    }
                    
                    if (message.type !== 'heartbeat') {
                        log(`[${new Date().toLocaleTimeString()}] ${message.type}: ${JSON.stringify(message.data)}`);
                    }
                };
                
                ws.onclose = () => {
                    log('Disconnected from control plane');
                    setTimeout(connectWebSocket, 5000);
                };
                
                ws.onerror = (error) => {
                    log('WebSocket error: ' + error);
                };
            }
            
            function updateMetrics(metrics) {
                document.getElementById('total-gpus').textContent = metrics.total_gpus;
                document.getElementById('available-gpus').textContent = metrics.available_gpus;
                document.getElementById('active-jobs').textContent = metrics.active_jobs;
                document.getElementById('avg-util').textContent = metrics.average_utilization.toFixed(1) + '%';
            }
            
            async function refreshStatus() {
                try {
                    const response = await fetch('/status?format=detailed&include_metrics=true');
                    const data = await response.json();
                    
                    // Update metrics
                    if (data.cluster_metrics) {
                        updateMetrics(data.cluster_metrics);
                    }
                    
                    // Update GPU list
                    const gpuList = document.getElementById('gpu-list');
                    gpuList.innerHTML = '';
                    
                    data.gpus.forEach(gpu => {
                        const item = document.createElement('div');
                        item.className = `gpu-item gpu-${gpu.state}`;
                        
                        const memUsed = gpu.memory_allocated_mb || 0;
                        const memTotal = gpu.memory_total_mb || 1;
                        const memPercent = ((memUsed / memTotal) * 100).toFixed(1);
                        
                        item.innerHTML = `
                            <div>
                                <strong>GPU ${gpu.gpu_id}</strong> - ${gpu.gpu_name} (${gpu.node_id})
                                <br>
                                <small>${gpu.state.toUpperCase()} | ${gpu.utilization_percent.toFixed(0)}% util | ${gpu.temperature_celsius.toFixed(0)}°C | Memory: ${memUsed}/${memTotal} MB (${memPercent}%)</small>
                            </div>
                            <div>
                                ${gpu.active_jobs.length > 0 ? `Jobs: ${gpu.active_jobs.length}` : ''}
                            </div>
                        `;
                        
                        gpuList.appendChild(item);
                    });
                    
                    log('Status refreshed');
                } catch (error) {
                    log('Error refreshing status: ' + error);
                }
            }
            
            async function populateDemo() {
                try {
                    const response = await fetch('/demo/populate', { method: 'POST' });
                    const data = await response.json();
                    log(data.message);
                    setTimeout(refreshStatus, 1000);
                } catch (error) {
                    log('Error populating demo data: ' + error);
                }
            }
            
            function log(message) {
                const logDiv = document.getElementById('log');
                const entry = document.createElement('div');
                entry.textContent = message;
                logDiv.appendChild(entry);
                logDiv.scrollTop = logDiv.scrollHeight;
                
                // Keep only last 100 messages
                while (logDiv.children.length > 100) {
                    logDiv.removeChild(logDiv.firstChild);
                }
            }
            
            function clearLog() {
                document.getElementById('log').innerHTML = '';
            }
            
            // Initialize
            connectWebSocket();
            refreshStatus();
            
            // Auto-refresh every 5 seconds
            setInterval(refreshStatus, 5000);
        </script>
    </body>
    </html>
    """

# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Cluster Control Plane")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Run server
    uvicorn.run(
        "control_plane:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
