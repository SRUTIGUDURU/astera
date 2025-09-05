#!/usr/bin/env python3
"""
Unified Telemetry Collection Module
Provides cross-platform telemetry collection for GPU and system metrics
with equal support for NVIDIA, AMD, and Intel GPUs.
"""

import platform
import subprocess
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
from enum import Enum, auto

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OSType(Enum):
    """Enumeration of supported operating system types."""
    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    UNKNOWN = auto()

class OSDetector:
    """
    A class to detect and provide information about the current operating system.
    """
    
    def __init__(self):
        """Initialize the OS detector."""
        self._system = platform.system()
        self._release = platform.release()
        self._version = platform.version()
        self._os_type = self._determine_os_type()
    
    def _determine_os_type(self) -> OSType:
        """
        Determine the operating system type based on platform information.
        
        Returns:
            OSType: The detected operating system type
        """
        system_lower = self._system.lower()
        
        if system_lower == "windows":
            return OSType.WINDOWS
        elif system_lower == "linux":
            return OSType.LINUX
        elif system_lower == "darwin":
            return OSType.MACOS
        else:
            return OSType.UNKNOWN
    
    @property
    def os_type(self) -> OSType:
        """Get the detected OS type."""
        return self._os_type
    
    @property
    def is_windows(self) -> bool:
        """Check if the OS is Windows."""
        return self._os_type == OSType.WINDOWS
    
    @property
    def is_linux(self) -> bool:
        """Check if the OS is Linux."""
        return self._os_type == OSType.LINUX
    
    @property
    def is_macos(self) -> bool:
        """Check if the OS is macOS."""
        return self._os_type == OSType.MACOS
    
    @property
    def is_unknown(self) -> bool:
        """Check if the OS is unknown/unsupported."""
        return self._os_type == OSType.UNKNOWN
    
    @property
    def system(self) -> str:
        """Get the raw system string from platform.system()."""
        return self._system
    
    @property
    def release(self) -> str:
        """Get the OS release version."""
        return self._release
    
    @property
    def version(self) -> str:
        """Get the OS version string."""
        return self._version
    
    def get_os_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the operating system.
        
        Returns:
            Dict containing OS information
        """
        return {
            "type": self._os_type.name,
            "system": self._system,
            "release": self._release,
            "version": self._version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    
    def get_os_name(self) -> str:
        """
        Get a human-readable name for the operating system.
        
        Returns:
            String representation of the OS name
        """
        if self.is_windows:
            return "Windows"
        elif self.is_linux:
            return "Linux"
        elif self.is_macos:
            return "macOS"
        else:
            return f"Unknown ({self._system})"
    
    def __str__(self) -> str:
        """String representation of the OS detector."""
        return f"OSDetector(type={self._os_type.name}, system={self._system}, release={self._release})"

@dataclass
class GPUStats:
    """Data schema for GPU metrics"""
    timestamp: str
    gpu_id: int
    name: str
    vendor: str
    temperature: int = 0  # Celsius
    utilization: int = 0  # Percentage
    memory_used: int = 0  # MB
    memory_total: int = 0  # MB
    power_usage: int = 0  # Watts
    power_limit: int = 0  # Watts
    driver_version: str = ""
    is_simulated: bool = False
    clock_core: int = 0  # MHz
    clock_memory: int = 0  # MHz
    fan_speed: int = 0  # Percentage

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

@dataclass
class SystemStats:
    """Data schema for system metrics"""
    timestamp: str
    cpu_usage: float = 0.0  # Percentage
    memory_used: int = 0  # MB
    memory_total: int = 0  # MB
    disk_usage: int = 0  # Percentage
    uptime: int = 0  # Seconds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

class TelemetryCollector:
    """Unified telemetry collector for all platforms and GPU vendors"""
    
    def __init__(self):
        """Initialize the telemetry collector."""
        self.os_detector = OSDetector()
        self.gpu_tools = self._detect_gpu_tools()
        self.gpus = self._detect_gpus()
        
        logger.info(f"Detected OS: {self.os_detector.get_os_name()}")
        if self.gpus:
            logger.info(f"Detected {len(self.gpus)} GPU(s)")
            for gpu in self.gpus:
                logger.info(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['vendor']})")
    
    def _detect_gpu_tools(self) -> Dict[str, str]:
        """Detect available GPU monitoring tools"""
        tools = {}
        
        # NVIDIA tools
        tools['nvidia_smi'] = self._find_tool('nvidia-smi', [
            '/usr/bin/nvidia-smi',
            '/opt/nvidia/bin/nvidia-smi',
            '/usr/local/bin/nvidia-smi',
            'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe',
            'C:\\Windows\\System32\\nvidia-smi.exe'
        ])
        
        # AMD tools
        tools['rocm_smi'] = self._find_tool('rocm-smi', [
            '/opt/rocm/bin/rocm-smi',
            '/usr/bin/rocm-smi'
        ])
        
        # Intel tools (oneAPI and others)
        tools['intel_gpu_top'] = self._find_tool('intel_gpu_top', [
            '/usr/bin/intel_gpu_top'
        ])
        
        return tools
    
    def _find_tool(self, tool_name: str, possible_paths: List[str]) -> Optional[str]:
        """Find a tool in PATH or common locations"""
        # Check if tool is in PATH
        try:
            if self.os_detector.is_windows:
                result = subprocess.run(["where", tool_name], capture_output=True, text=True, timeout=5,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                result = subprocess.run(['which', tool_name], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        
        # Check common locations
        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    def _detect_gpus(self) -> List[Dict[str, Any]]:
        """Detect available GPUs using platform-specific methods"""
        gpus = []
        
        if self.os_detector.is_windows:
            gpus = self._detect_windows_gpus()
        elif self.os_detector.is_linux:
            gpus = self._detect_linux_gpus()
        elif self.os_detector.is_macos:
            gpus = self._detect_macos_gpus()
        
        return gpus
    
    def _detect_windows_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPUs on Windows"""
        gpus = []
        
        # Method 1: WMI detection
        try:
            command = """
            Get-WmiObject -Class Win32_VideoController | Where-Object { 
                $_.AdapterDACType -ne 'Unknown' -and $_.Name -notlike '*Remote*' 
            } | Select-Object -Property Name,AdapterCompatibility,AdapterRAM,DriverVersion,DeviceID | ConvertTo-Json
            """
            
            result = self._run_powershell(command)
            if result:
                gpu_list = json.loads(result) if result.startswith('[') else [json.loads(result)]
                for i, gpu in enumerate(gpu_list):
                    vendor = gpu.get('AdapterCompatibility', 'Unknown')
                    name = gpu.get('Name', f'GPU {i}')
                    
                    # Skip basic display adapters and virtual GPUs
                    if 'Basic Display' in name or 'Microsoft' in vendor:
                        continue
                        
                    gpus.append({
                        'id': i,
                        'name': name,
                        'vendor': vendor,
                        'detection_method': 'wmi',
                        'device_id': gpu.get('DeviceID', '')
                    })
        except Exception as e:
            logger.debug(f"Failed to detect GPUs via WMI: {e}")
        
        return gpus
    
    def _detect_linux_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPUs on Linux"""
        gpus = []
        
        # Method 1: lspci detection
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA' in line or '3D' in line or 'Display' in line:
                        vendor = self._determine_gpu_vendor(line)
                        name = line.split(': ')[-1] if ': ' in line else line
                        
                        gpus.append({
                            'id': len(gpus),
                            'name': name,
                            'vendor': vendor,
                            'detection_method': 'lspci'
                        })
        except Exception as e:
            logger.debug(f"Failed to detect GPUs via lspci: {e}")
        
        # Method 2: Vendor-specific detection
        if self.gpu_tools['nvidia_smi']:
            try:
                result = subprocess.run(
                    [self.gpu_tools['nvidia_smi'], "--query-gpu=index,name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.split(',', 1)
                            if len(parts) >= 2:
                                gpu_id = parts[0].strip()
                                name = parts[1].strip()
                                
                                # Check if this GPU is already detected
                                if not any(g['name'] == name for g in gpus):
                                    gpus.append({
                                        'id': int(gpu_id),
                                        'name': name,
                                        'vendor': 'NVIDIA',
                                        'detection_method': 'nvidia-smi'
                                    })
            except Exception as e:
                logger.debug(f"Failed to detect NVIDIA GPUs: {e}")
        
        return gpus
    
    def _detect_macos_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPUs on macOS"""
        gpus = []
        
        try:
            # Use system_profiler to get GPU information
            result = subprocess.run([
                'system_profiler', 'SPDisplaysDataType', '-json'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                if 'SPDisplaysDataType' in data and len(data['SPDisplaysDataType']) > 0:
                    for display_info in data['SPDisplaysDataType']:
                        if 'sppci_model' in display_info:
                            vendor = self._determine_gpu_vendor(display_info.get('sppci_model', ''))
                            name = display_info.get('sppci_model', 'Unknown GPU')
                            
                            gpus.append({
                                'id': len(gpus),
                                'name': name,
                                'vendor': vendor,
                                'detection_method': 'system_profiler'
                            })
        except Exception as e:
            logger.debug(f"Failed to detect GPUs via system_profiler: {e}")
        
        # Fallback for Apple Silicon
        if not gpus:
            try:
                result = subprocess.run([
                    'sysctl', '-n', 'machdep.cpu.brand_string'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and 'Apple' in result.stdout:
                    gpus.append({
                        'id': 0,
                        'name': 'Apple Integrated GPU',
                        'vendor': 'Apple',
                        'detection_method': 'sysctl'
                    })
            except:
                pass
        
        return gpus
    
    def _determine_gpu_vendor(self, info: str) -> str:
        """Determine GPU vendor from various information sources"""
        info_lower = info.lower()
        
        if 'nvidia' in info_lower or 'geforce' in info_lower:
            return 'NVIDIA'
        elif 'amd' in info_lower or 'radeon' in info_lower or 'ati' in info_lower:
            return 'AMD'
        elif 'intel' in info_lower or 'iris' in info_lower or 'uhd graphics' in info_lower:
            return 'Intel'
        elif 'apple' in info_lower:
            return 'Apple'
        else:
            return 'Unknown'
    
    def _run_powershell(self, command: str, timeout: int = 10) -> Optional[str]:
        """Execute a PowerShell command and return the output (Windows only)"""
        if not self.os_detector.is_windows:
            return None
            
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True, 
                text=True, 
                timeout=timeout,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return None
    
    def get_gpu_count(self) -> int:
        """Get the number of available GPUs"""
        return len(self.gpus) if self.gpus else 1  # Assume at least 1 GPU
    
    def get_gpu_stats(self, gpu_id: int = 0) -> Optional[GPUStats]:
        """
        Get statistics for a specific GPU
        
        Args:
            gpu_id: The ID of the GPU to query
            
        Returns:
            GPUStats object or None if failed
        """
        # Get GPU info
        gpu_info = None
        if self.gpus and gpu_id < len(self.gpus):
            gpu_info = self.gpus[gpu_id]
        else:
            # Fallback: create basic GPU info
            gpu_info = {
                'id': gpu_id,
                'name': f"GPU {gpu_id}",
                'vendor': 'Unknown',
                'detection_method': 'fallback'
            }
        
        vendor = gpu_info.get('vendor', 'Unknown')
        
        # Try vendor-specific methods
        if vendor == 'NVIDIA' and self.gpu_tools['nvidia_smi']:
            stats = self._get_nvidia_stats(gpu_id)
            if stats:
                return stats
        elif vendor == 'AMD' and self.gpu_tools['rocm_smi']:
            stats = self._get_amd_stats(gpu_id)
            if stats:
                return stats
        elif vendor == 'Intel' and self.gpu_tools['intel_gpu_top']:
            stats = self._get_intel_stats(gpu_id)
            if stats:
                return stats
        
        # Fall back to platform-specific generic methods
        if self.os_detector.is_windows:
            return self._get_windows_generic_stats(gpu_id, gpu_info)
        elif self.os_detector.is_linux:
            return self._get_linux_generic_stats(gpu_id, gpu_info)
        elif self.os_detector.is_macos:
            return self._get_macos_generic_stats(gpu_id, gpu_info)
        
        # Final fallback
        return GPUStats(
            timestamp=datetime.now().isoformat(),
            gpu_id=gpu_id,
            name=gpu_info.get('name', f"GPU {gpu_id}"),
            vendor=vendor,
            is_simulated=False
        )
    
    def _get_nvidia_stats(self, gpu_id: int) -> Optional[GPUStats]:
        """Get NVIDIA GPU statistics using nvidia-smi"""
        try:
            result = subprocess.run([
                self.gpu_tools['nvidia_smi'],
                f"-i", f"{gpu_id}",
                "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,driver_version,clocks.current.graphics,clocks.current.memory,fan.speed",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if self.os_detector.is_windows else 0)
            
            if result.returncode == 0 and result.stdout.strip():
                values = [val.strip() for val in result.stdout.strip().split(',')]
                
                if len(values) >= 12:
                    def safe_float_parse(value, default=0):
                        try:
                            if not value or value == "N/A" or value == "[N/A]" or not value.replace('.', '').replace('-', '').isdigit():
                                return default
                            return float(value)
                        except:
                            return default
                    
                    timestamp = values[0]
                    name = values[1]
                    temp = int(safe_float_parse(values[2]))
                    util = int(safe_float_parse(values[3]))
                    mem_used = int(safe_float_parse(values[4]))
                    mem_total = int(safe_float_parse(values[5]))
                    power_usage = int(safe_float_parse(values[6]))
                    power_limit = int(safe_float_parse(values[7]))
                    driver_version = values[8] if values[8] not in ["N/A", "[N/A]"] else ""
                    clock_core = int(safe_float_parse(values[9]))
                    clock_memory = int(safe_float_parse(values[10]))
                    fan_speed = int(safe_float_parse(values[11]))
                    
                    return GPUStats(
                        timestamp=timestamp or datetime.now().isoformat(),
                        gpu_id=gpu_id,
                        name=name,
                        vendor="NVIDIA",
                        temperature=temp,
                        utilization=util,
                        memory_used=mem_used,
                        memory_total=mem_total,
                        power_usage=power_usage,
                        power_limit=power_limit,
                        driver_version=driver_version,
                        clock_core=clock_core,
                        clock_memory=clock_memory,
                        fan_speed=fan_speed,
                        is_simulated=False
                    )
        except Exception as e:
            logger.debug(f"Failed to get NVIDIA GPU stats: {e}")
        
        return None
    
    def _get_amd_stats(self, gpu_id: int) -> Optional[GPUStats]:
        """Get AMD GPU statistics using rocm-smi"""
        try:
            # Get temperature
            temp_result = subprocess.run([
                self.gpu_tools['rocm_smi'], "--showtemp", "-d", str(gpu_id)
            ], capture_output=True, text=True, timeout=10)
            
            temperature = 0
            if temp_result.returncode == 0:
                for line in temp_result.stdout.split('\n'):
                    if 'temperature' in line.lower():
                        try:
                            temp_str = line.split(':')[-1].strip().replace('C', '').strip()
                            temperature = int(float(temp_str))
                        except:
                            pass
            
            # Get utilization
            util_result = subprocess.run([
                self.gpu_tools['rocm_smi'], "--showuse", "-d", str(gpu_id)
            ], capture_output=True, text=True, timeout=10)
            
            utilization = 0
            if util_result.returncode == 0:
                for line in util_result.stdout.split('\n'):
                    if 'use' in line.lower() and '%' in line:
                        try:
                            util_str = line.split(':')[-1].strip().replace('%', '').strip()
                            utilization = int(float(util_str))
                        except:
                            pass
            
            # Get memory usage
            mem_result = subprocess.run([
                self.gpu_tools['rocm_smi'], "--showmemuse", "-d", str(gpu_id)
            ], capture_output=True, text=True, timeout=10)
            
            memory_used = 0
            memory_total = 0
            if mem_result.returncode == 0:
                for line in mem_result.stdout.split('\n'):
                    if 'vram' in line.lower() and 'total' in line.lower():
                        try:
                            mem_str = line.split(':')[-1].strip().replace('MB', '').strip()
                            memory_total = int(float(mem_str))
                        except:
                            pass
                    elif 'vram' in line.lower() and 'used' in line.lower():
                        try:
                            mem_str = line.split(':')[-1].strip().replace('MB', '').strip()
                            memory_used = int(float(mem_str))
                        except:
                            pass
            
            return GPUStats(
                timestamp=datetime.now().isoformat(),
                gpu_id=gpu_id,
                name=f"AMD GPU {gpu_id}",
                vendor="AMD",
                temperature=temperature,
                utilization=utilization,
                memory_used=memory_used,
                memory_total=memory_total,
                is_simulated=False
            )
        except Exception as e:
            logger.debug(f"Failed to get AMD GPU stats: {e}")
        
        return None
    
    def _get_intel_stats(self, gpu_id: int) -> Optional[GPUStats]:
        """Get Intel GPU statistics (limited support)"""
        # Intel GPU monitoring is more limited
        try:
            # Try to get basic info from sysfs (Linux)
            if self.os_detector.is_linux:
                temperature = 0
                # Check temperature files
                temp_paths = [
                    f"/sys/class/drm/card{gpu_id}/device/hwmon/hwmon*/temp1_input",
                    f"/sys/class/drm/card{gpu_id}/device/temp1_input"
                ]
                
                import glob
                for path_pattern in temp_paths:
                    for path in glob.glob(path_pattern):
                        try:
                            with open(path, 'r') as f:
                                temp_millic = int(f.read().strip())
                                temperature = temp_millic // 1000
                                break
                        except:
                            continue
                
                return GPUStats(
                    timestamp=datetime.now().isoformat(),
                    gpu_id=gpu_id,
                    name=f"Intel GPU {gpu_id}",
                    vendor="Intel",
                    temperature=temperature,
                    is_simulated=False
                )
        except Exception as e:
            logger.debug(f"Failed to get Intel GPU stats: {e}")
        
        return None
    
    def _get_windows_generic_stats(self, gpu_id: int, gpu_info: Dict[str, Any]) -> GPUStats:
        """Get generic GPU statistics on Windows"""
        # Try to get utilization via performance counters
        utilization = 0
        memory_used = 0
        
        try:
            command = f"""
            $utilization = 0
            $memoryUsed = 0
            try {{
                $counters = Get-Counter -Counter "\\GPU Engine(*)\\Utilization Percentage" -ErrorAction SilentlyContinue
                if ($counters) {{
                    $readings = 0
                    foreach ($counter in $counters.CounterSamples) {{
                        if ($counter.InstanceName -like "*_engtype_3D" -or $counter.InstanceName -like "*_engtype_Compute") {{
                            $currentGpuId = [int]($counter.InstanceName.Split('_')[0])
                            if ($currentGpuId -eq {gpu_id}) {{
                                $utilization += $counter.CookedValue
                                $readings++
                            }}
                        }}
                    }}
                    if ($readings -gt 0) {{ $utilization = [math]::Round($utilization / $readings) }}
                }}
                
                $memCounters = Get-Counter -Counter "\\GPU Process Memory(*)\\Local Usage" -ErrorAction SilentlyContinue
                if ($memCounters) {{
                    foreach ($counter in $memCounters.CounterSamples) {{
                        $currentGpuId = [int]($counter.InstanceName.Split('_')[0])
                        if ($currentGpuId -eq {gpu_id}) {{
                            $memoryUsed += $counter.CookedValue
                        }}
                    }}
                    $memoryUsed = [math]::Round($memoryUsed / 1MB)
                }}
            }} catch {{ }}
            "$utilization,$memoryUsed"
            """
            
            result = self._run_powershell(command)
            if result and ',' in result:
                util_str, mem_str = result.split(',')
                utilization = int(float(util_str)) if util_str.isdigit() else 0
                memory_used = int(float(mem_str)) if mem_str.isdigit() else 0
        except:
            pass
        
        return GPUStats(
            timestamp=datetime.now().isoformat(),
            gpu_id=gpu_id,
            name=gpu_info.get('name', f"GPU {gpu_id}"),
            vendor=gpu_info.get('vendor', 'Unknown'),
            utilization=utilization,
            memory_used=memory_used,
            is_simulated=False
        )
    
    def _get_linux_generic_stats(self, gpu_id: int, gpu_info: Dict[str, Any]) -> GPUStats:
        """Get generic GPU statistics on Linux"""
        temperature = 0
        
        # Try to get temperature from sysfs
        try:
            temp_paths = [
                f"/sys/class/drm/card{gpu_id}/device/hwmon/hwmon*/temp1_input",
                f"/sys/class/drm/card{gpu_id}/device/temp1_input",
                f"/sys/class/hwmon/hwmon*/temp1_input"
            ]
            
            import glob
            for path_pattern in temp_paths:
                for path in glob.glob(path_pattern):
                    try:
                        with open(path, 'r') as f:
                            temp_millic = int(f.read().strip())
                            temperature = temp_millic // 1000
                            break
                    except:
                        continue
        except:
            pass
        
        return GPUStats(
            timestamp=datetime.now().isoformat(),
            gpu_id=gpu_id,
            name=gpu_info.get('name', f"GPU {gpu_id}"),
            vendor=gpu_info.get('vendor', 'Unknown'),
            temperature=temperature,
            is_simulated=False
        )
    
    def _get_macos_generic_stats(self, gpu_id: int, gpu_info: Dict[str, Any]) -> GPUStats:
        """Get generic GPU statistics on macOS"""
        # macOS has limited GPU monitoring capabilities without vendor tools
        temperature = 0
        
        try:
            # Try to get temperature from powermetrics (requires sudo)
            result = subprocess.run([
                'sudo', 'powermetrics', '--samplers', 'thermal', '-n', '1', '-i', '100'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'GPU die temperature' in line:
                        try:
                            temp_str = line.split(':')[-1].strip().replace('C', '')
                            temperature = int(float(temp_str))
                            break
                        except:
                            pass
        except:
            pass
        
        return GPUStats(
            timestamp=datetime.now().isoformat(),
            gpu_id=gpu_id,
            name=gpu_info.get('name', f"GPU {gpu_id}"),
            vendor=gpu_info.get('vendor', 'Unknown'),
            temperature=temperature,
            is_simulated=False
        )
    
    def get_all_gpu_stats(self) -> List[GPUStats]:
        """Get statistics for all available GPUs"""
        stats = []
        gpu_count = self.get_gpu_count()
        
        for gpu_id in range(gpu_count):
            gpu_stat = self.get_gpu_stats(gpu_id)
            if gpu_stat:
                stats.append(gpu_stat)
        
        return stats
    
    def get_system_stats(self) -> SystemStats:
        """Get system-level statistics"""
        if self.os_detector.is_windows:
            return self._get_windows_system_stats()
        elif self.os_detector.is_linux:
            return self._get_linux_system_stats()
        elif self.os_detector.is_macos:
            return self._get_macos_system_stats()
        else:
            return SystemStats(timestamp=datetime.now().isoformat())
    
    def _get_windows_system_stats(self) -> SystemStats:
        """Get system statistics on Windows"""
        cpu_usage = 0.0
        memory_used = 0
        memory_total = 0
        disk_usage = 0
        uptime = 0
        
        try:
            # CPU usage
            command = """
            $cpuUsage = (Get-WmiObject -Class Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
            if (-not $cpuUsage) { $cpuUsage = 0 }
            
            # Memory usage
            $memory = Get-WmiObject -Class Win32_OperatingSystem
            $memoryUsed = [math]::Round(($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / 1KB)
            $memoryTotal = [math]::Round($memory.TotalVisibleMemorySize / 1KB)
            
            # Disk usage
            $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction SilentlyContinue
            $diskUsage = 0
            if ($disk -and $disk.Size -gt 0) {
                $diskUsage = [math]::Round(($disk.Size - $disk.FreeSpace) / $disk.Size * 100)
            }
            
            # Uptime
            $os = Get-WmiObject -Class Win32_OperatingSystem
            $lastBoot = $os.LastBootUpTime
            $bootTime = [System.Management.ManagementDateTimeConverter]::ToDateTime($lastBoot)
            $uptime = [math]::Round((Get-Date) - $bootTime).TotalSeconds
            
            "$cpuUsage,$memoryUsed,$memoryTotal,$diskUsage,$uptime"
            """
            
            result = self._run_powershell(command)
            if result and ',' in result:
                values = result.split(',')
                if len(values) >= 5:
                    cpu_usage = float(values[0]) if values[0].replace('.', '').isdigit() else 0.0
                    memory_used = int(values[1]) if values[1].isdigit() else 0
                    memory_total = int(values[2]) if values[2].isdigit() else 0
                    disk_usage = int(values[3]) if values[3].isdigit() else 0
                    uptime = int(values[4]) if values[4].isdigit() else 0
        except:
            pass
        
        return SystemStats(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_usage=disk_usage,
            uptime=uptime
        )
    
    def _get_linux_system_stats(self) -> SystemStats:
        """Get system statistics on Linux"""
        cpu_usage = 0.0
        memory_used = 0
        memory_total = 0
        disk_usage = 0
        uptime = 0
        
        try:
            # CPU usage from /proc/stat
            with open('/proc/stat', 'r') as f:
                first_line = f.readline().strip()
            
            if first_line.startswith('cpu '):
                values = first_line.split()[1:]
                if len(values) >= 4:
                    user, nice, system, idle = map(int, values[:4])
                    total = user + nice + system + idle
                    
                    time.sleep(0.1)
                    
                    with open('/proc/stat', 'r') as f:
                        first_line2 = f.readline().strip()
                    
                    if first_line2.startswith('cpu '):
                        values2 = first_line2.split()[1:]
                        if len(values2) >= 4:
                            user2, nice2, system2, idle2 = map(int, values2[:4])
                            total2 = user2 + nice2 + system2 + idle2
                            
                            total_delta = total2 - total
                            idle_delta = idle2 - idle
                            
                            if total_delta > 0:
                                cpu_usage = 100.0 * (total_delta - idle_delta) / total_delta
            
            # Memory usage from /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    memory_total = int(line.split()[1]) // 1024
                elif line.startswith('MemAvailable:'):
                    memory_available = int(line.split()[1]) // 1024
                    memory_used = memory_total - memory_available
            
            # Disk usage
            result = subprocess.run(['df', '/'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        usage_str = parts[4].replace('%', '')
                        disk_usage = int(usage_str) if usage_str.isdigit() else 0
            
            # Uptime from /proc/uptime
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
                uptime = int(uptime_seconds)
        except:
            pass
        
        return SystemStats(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_usage=disk_usage,
            uptime=uptime
        )
    
    def _get_macos_system_stats(self) -> SystemStats:
        """Get system statistics on macOS"""
        cpu_usage = 0.0
        memory_used = 0
        memory_total = 0
        disk_usage = 0
        uptime = 0
        
        try:
            # CPU usage from top
            result = subprocess.run([
                'top', '-l', '1', '-n', '0'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CPU usage' in line:
                        try:
                            usage_str = line.split('CPU usage:')[-1].strip()
                            idle_str = [part for part in usage_str.split(',') if 'idle' in part][0]
                            idle_percent = float(idle_str.split('%')[0].strip())
                            cpu_usage = 100.0 - idle_percent
                        except:
                            pass
            
            # Memory usage
            result = subprocess.run([
                'vm_stat'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                                # Parse vm_stat output
                page_size = 4096
                stats = {}
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().replace('"', '')
                        value = value.strip().replace('.', '')
                        if value.isdigit():
                            stats[key] = int(value)
                
                if 'Pages active' in stats and 'Pages wired' in stats:
                    used_pages = stats.get('Pages active', 0) + stats.get('Pages wired', 0)
                    memory_used = (used_pages * page_size) // (1024 * 1024)
            
            # Total memory
            result = subprocess.run([
                'sysctl', '-n', 'hw.memsize'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                memory_total = int(result.stdout.strip()) // (1024 * 1024)
            
            # Disk usage
            result = subprocess.run(['df', '/'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        usage_str = parts[4].replace('%', '')
                        disk_usage = int(usage_str) if usage_str.isdigit() else 0
            
            # Uptime
            result = subprocess.run(['sysctl', '-n', 'kern.boottime'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                boot_time_str = result.stdout.strip()
                boot_time_match = re.search(r'sec\s*=\s*(\d+)', boot_time_str)
                if boot_time_match:
                    boot_time = int(boot_time_match.group(1))
                    current_time = int(time.time())
                    uptime = current_time - boot_time
        except:
            pass
        
        return SystemStats(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_usage=disk_usage,
            uptime=uptime
        )


# Factory functions for easy creation
def create_os_detector() -> OSDetector:
    """
    Create and return an OSDetector instance.
    
    Returns:
        OSDetector instance
    """
    return OSDetector()

def create_telemetry_collector() -> TelemetryCollector:
    """
    Create and return a TelemetryCollector instance.
    
    Returns:
        TelemetryCollector instance
    """
    return TelemetryCollector()

# Utility functions for common use cases
def is_windows() -> bool:
    """
    Quick check if the current OS is Windows.
    
    Returns:
        True if running on Windows, False otherwise
    """
    return platform.system().lower() == "windows"

def is_linux() -> bool:
    """
    Quick check if the current OS is Linux.
    
    Returns:
        True if running on Linux, False otherwise
    """
    return platform.system().lower() == "linux"

def is_macos() -> bool:
    """
    Quick check if the current OS is macOS.
    
    Returns:
        True if running on macOS, False otherwise
    """
    return platform.system().lower() == "darwin"

def get_os_name() -> str:
    """
    Get a human-readable name for the current operating system.
    
    Returns:
        String representation of the OS name
    """
    system = platform.system().lower()
    if system == "windows":
        return "Windows"
    elif system == "linux":
        return "Linux"
    elif system == "darwin":
        return "macOS"
    else:
        return f"Unknown ({system})"


# Example usage and demonstration
if __name__ == "__main__":
    try:
        # Create detector instances
        os_detector = create_os_detector()
        collector = create_telemetry_collector()
        
        # Display OS information
        print(f"Detected OS: {os_detector.get_os_name()}")
        print(f"OS Type: {os_detector.os_type.name}")
        print(f"System: {os_detector.system}")
        print(f"Release: {os_detector.release}")
        print(f"Version: {os_detector.version}")
        
        # Display GPU detection info
        if collector.gpus:
            print(f"\nDetected {len(collector.gpus)} GPU(s):")
            for gpu in collector.gpus:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['vendor']})")
        
        # Get GPU information
        gpu_count = collector.get_gpu_count()
        print(f"\nTotal detected GPUs: {gpu_count}")
        
        # Get GPU stats
        gpu_stats = collector.get_all_gpu_stats()
        for stat in gpu_stats:
            print(f"\nGPU {stat.gpu_id} ({stat.name} - {stat.vendor}):")
            print(f"  Temperature: {stat.temperature}°C")
            print(f"  Utilization: {stat.utilization}%")
            print(f"  Memory: {stat.memory_used}/{stat.memory_total} MB")
            if stat.power_usage > 0:
                print(f"  Power: {stat.power_usage}/{stat.power_limit} W")
            if stat.clock_core > 0:
                print(f"  Clocks: {stat.clock_core} MHz core, {stat.clock_memory} MHz memory")
            if stat.fan_speed > 0:
                print(f"  Fan: {stat.fan_speed}%")
            print(f"  Driver: {stat.driver_version}")
        
        # Get system stats
        system_stats = collector.get_system_stats()
        print(f"\nSystem Statistics:")
        print(f"  CPU Usage: {system_stats.cpu_usage:.1f}%")
        print(f"  Memory: {system_stats.memory_used}/{system_stats.memory_total} MB")
        print(f"  Disk Usage: {system_stats.disk_usage}%")
        print(f"  Uptime: {system_stats.uptime} seconds")
        
        # Demonstrate utility functions
        print(f"\nUtility function checks:")
        print(f"  is_windows(): {is_windows()}")
        print(f"  is_linux(): {is_linux()}")
        print(f"  is_macos(): {is_macos()}")
        print(f"  get_os_name(): {get_os_name()}")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
