#!/usr/bin/env python3
"""
GPU Node Agent - Integrates telemetry with control plane
"""

import asyncio
import json
import logging
import socket
from datetime import datetime
import aiohttp
import platform

# Import your telemetry module
from gpu_telemetry import create_telemetry_collector, GPUStats

logger = logging.getLogger(__name__)

class GPUNodeAgent:
    def __init__(self, control_plane_url: str, node_id: str = None):
        self.control_plane_url = control_plane_url.rstrip('/')
        self.node_id = node_id or f"{platform.node()}-{socket.gethostname()}"
        self.telemetry = create_telemetry_collector()
        self.registered_gpus = {}
        self.running = True
        
    async def start(self):
        """Start the GPU node agent"""
        logger.info(f"Starting GPU node agent for {self.node_id}")
        
        # Register GPUs
        await self.register_gpus()
        
        # Start monitoring tasks
        await asyncio.gather(
            self.telemetry_loop(),
            self.heartbeat_loop(),
            return_exceptions=True
        )
    
    async def register_gpus(self):
        """Register all detected GPUs with control plane"""
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
                        "supports_nvlink": False,  # Detect if needed
                        "supports_mig": False
                    }
                
                try:
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
                    logger.error(f"Error registering GPU {i}: {e}")
    
    async def telemetry_loop(self):
        """Send GPU metrics to control plane"""
        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    # Collect metrics for all GPUs
                    gpu_stats = self.telemetry.get_all_gpu_stats()
                    
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
                            
                            # Send update
                            try:
                                async with session.patch(
                                    f"{self.control_plane_url}/gpu/{node_id}/metrics",
                                    json=metrics
                                ) as resp:
                                    if resp.status != 200:
                                        logger.warning(f"Failed to update metrics for {node_id}")
                            except Exception as e:
                                logger.error(f"Error updating metrics for {node_id}: {e}")
                
                except Exception as e:
                    logger.error(f"Telemetry loop error: {e}")
                
                await asyncio.sleep(10)  # Update every 10 seconds
    
    async def heartbeat_loop(self):
        """Send periodic heartbeats"""
        async with aiohttp.ClientSession() as session:
            while self.running:
                for gpu_id, info in self.registered_gpus.items():
                    try:
                        # Heartbeat is sent with metrics update
                        pass
                    except Exception as e:
                        logger.error(f"Heartbeat error: {e}")
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
    
    async def stop(self):
        """Stop the agent"""
        self.running = False
        logger.info("GPU node agent stopped")

async def main():
    logging.basicConfig(level=logging.INFO)
    
    # Create and start agent
    agent = GPUNodeAgent("http://localhost:8000")
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
