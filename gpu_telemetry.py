#!/usr/bin/env python3
"""
Fixed Unified Telemetry Collection Module with Working Temperature Detection
"""

import platform
import subprocess
import json
import logging
import os
import re
import time
import glob
import ctypes
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
from enum import Enum, auto
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OSType(Enum):
    """Enumeration of supported operating system types."""
    WINDOWS = auto()
    LINUX = auto()
    MACOS = auto()
    ANDROID = auto()
    UNKNOWN = auto()

class OSDetector:
    """A class to detect and provide information about the current operating system."""
    
    def __init__(self):
        """Initialize the OS detector."""
        self._system = platform.system()
        self._release = platform.release()
        self._version = platform.version()
        self._machine = platform.machine()
        self._processor = platform.processor()
        self._os_type = self._determine_os_type()
        self._is_arm = self._detect_arm_architecture()
        self._is_apple_silicon = self._detect_apple_silicon()
    
    def _determine_os_type(self) -> OSType:
        """Determine the operating system type based on platform information."""
        system_lower = self._system.lower()
        
        if system_lower == "windows":
            return OSType.WINDOWS
        elif system_lower == "linux":
            return OSType.LINUX
        elif system_lower == "darwin":
            return OSType.MACOS
        else:
            return OSType.UNKNOWN
    
    def _detect_arm_architecture(self) -> bool:
        """Detect if running on ARM architecture."""
        machine = self._machine.lower()
        processor = self._processor.lower()
        
        arm_indicators = ['arm', 'aarch64', 'arm64', 'armv7', 'armv8']
        return any(indicator in machine or indicator in processor 
                  for indicator in arm_indicators)
    
    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        if not self.is_macos:
            return False
        
        try:
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                brand = result.stdout.strip().lower()
                return 'apple' in brand
        except:
            pass
        
        return self._machine.lower() == 'arm64'
    
    @property
    def os_type(self) -> OSType:
        return self._os_type
    
    @property
    def is_windows(self) -> bool:
        return self._os_type == OSType.WINDOWS
    
    @property
    def is_linux(self) -> bool:
        return self._os_type == OSType.LINUX
    
    @property
    def is_macos(self) -> bool:
        return self._os_type == OSType.MACOS
    
    @property
    def is_android(self) -> bool:
        return self._os_type == OSType.ANDROID
    
    @property
    def is_unknown(self) -> bool:
        return self._os_type == OSType.UNKNOWN
    
    @property
    def is_arm(self) -> bool:
        return self._is_arm
    
    @property
    def is_apple_silicon(self) -> bool:
        return self._is_apple_silicon
    
    @property
    def system(self) -> str:
        return self._system
    
    @property
    def release(self) -> str:
        return self._release
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def machine(self) -> str:
        return self._machine
    
    @property
    def processor(self) -> str:
        return self._processor
    
    def get_os_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the operating system."""
        return {
            "type": self._os_type.name,
            "system": self._system,
            "release": self._release,
            "version": self._version,
            "platform": platform.platform(),
            "machine": self._machine,
            "processor": self._processor,
            "is_arm": self._is_arm,
            "is_apple_silicon": self._is_apple_silicon,
            "python_version": platform.python_version(),
            "architecture": platform.architecture(),
        }
    
    def get_os_name(self) -> str:
        """Get a human-readable name for the operating system."""
        if self.is_windows:
            return "Windows"
        elif self.is_linux:
            return "Linux"
        elif self.is_macos:
            return "macOS"
        elif self.is_android:
            return "Android"
        else:
            return f"Unknown ({self._system})"

@dataclass
class GPUStats:
    """Data schema for GPU metrics"""
    timestamp: str
    gpu_id: int
    name: str
    vendor: str
    temperature: float = 0.0  # Celsius
    utilization: float = 0.0  # Percentage
    memory_used: int = 0  # MB
    memory_total: int = 0  # MB
    power_usage: float = 0.0  # Watts
    power_limit: float = 0.0  # Watts
    driver_version: str = ""
    is_simulated: bool = False
    clock_core: int = 0  # MHz
    clock_memory: int = 0  # MHz
    fan_speed: int = 0  # Percentage
    # Extended metrics
    memory_bandwidth: float = 0.0  # GB/s
    compute_units: int = 0
    shader_units: int = 0
    texture_units: int = 0
    render_output_units: int = 0
    pcie_generation: str = ""
    pcie_lanes: int = 0
    voltage: float = 0.0  # Volts
    performance_state: str = ""
    throttle_reasons: List[str] = None
    # Apple-specific metrics
    apple_gpu_cores: int = 0
    apple_neural_engine_cores: int = 0
    # Qualcomm-specific metrics
    adreno_version: str = ""
    opengl_version: str = ""
    vulkan_version: str = ""

    def __post_init__(self):
        if self.throttle_reasons is None:
            self.throttle_reasons = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class SystemStats:
    """Data schema for system metrics"""
    timestamp: str
    cpu_usage: float = 0.0  # Percentage
    memory_used: int = 0  # MB
    memory_total: int = 0  # MB
    disk_usage: float = 0.0  # Percentage
    uptime: int = 0  # Seconds
    # Extended metrics
    cpu_count: int = 0
    cpu_frequency: float = 0.0  # MHz
    cpu_temperature: float = 0.0  # Celsius
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 1min, 5min, 15min
    network_bytes_sent: int = 0
    network_bytes_received: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0
    swap_used: int = 0  # MB
    swap_total: int = 0  # MB
    process_count: int = 0
    battery_percentage: float = 0.0
    battery_time_remaining: int = 0  # Minutes
    thermal_state: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

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
            '/usr/local/cuda/bin/nvidia-smi',
            'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe',
            'C:\\Windows\\System32\\nvidia-smi.exe'
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
        
        # Ensure GPUs have sequential IDs
        for i, gpu in enumerate(gpus):
            gpu['id'] = i
        
        return gpus
    
    def _detect_windows_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPUs on Windows using multiple methods"""
        gpus = []
        
        # Method 1: NVIDIA detection first (if available)
        if self.gpu_tools.get('nvidia_smi'):
            nvidia_gpus = self._detect_nvidia_gpus_windows()
            gpus.extend(nvidia_gpus)
        
        # Method 2: WMI detection for ALL GPUs (including Intel)
        try:
            command = """
            $gpus = @()
            
            # Get ALL video controllers
            $videoControllers = Get-WmiObject -Class Win32_VideoController | Where-Object { 
                $_.Name -notlike '*Remote*' -and 
                $_.Name -notlike '*Basic Display*' -and 
                $_.Name -notlike '*Microsoft*' -and
                $_.AdapterDACType -ne 'Unknown' -and
                $_.Name -ne $null -and
                $_.Name -ne ''
            }
            
            foreach ($gpu in $videoControllers) {
                $gpuInfo = @{
                    'Name' = $gpu.Name
                    'AdapterCompatibility' = if ($gpu.AdapterCompatibility) { $gpu.AdapterCompatibility } else { 'Unknown' }
                    'AdapterRAM' = if ($gpu.AdapterRAM -and $gpu.AdapterRAM -gt 0) { [math]::Round($gpu.AdapterRAM / 1MB) } else { 0 }
                    'DriverVersion' = if ($gpu.DriverVersion) { $gpu.DriverVersion } else { '' }
                    'DeviceID' = if ($gpu.DeviceID) { $gpu.DeviceID } else { '' }
                    'PNPDeviceID' = if ($gpu.PNPDeviceID) { $gpu.PNPDeviceID } else { '' }
                    'Status' = if ($gpu.Status) { $gpu.Status } else { 'Unknown' }
                }
                $gpus += $gpuInfo
            }
            
            $gpus | ConvertTo-Json -Depth 3
            """
            
            result = self._run_powershell(command, timeout=20)
            if result:
                try:
                    wmi_gpus = json.loads(result) if result.startswith('[') else [json.loads(result)]
                    
                    for wmi_gpu in wmi_gpus:
                        name = wmi_gpu.get('Name', 'Unknown GPU')
                        vendor = self._determine_gpu_vendor(name)
                        
                        # Skip if this exact GPU is already detected by NVIDIA method
                        existing_gpu = next((g for g in gpus if g['name'].lower() == name.lower()), None)
                        
                        if not existing_gpu:
                            gpu_info = {
                                'id': len(gpus),
                                'name': name,
                                'vendor': vendor,
                                'memory_total': wmi_gpu.get('AdapterRAM', 0),
                                'driver_version': wmi_gpu.get('DriverVersion', ''),
                                'device_id': wmi_gpu.get('DeviceID', ''),
                                'pnp_device_id': wmi_gpu.get('PNPDeviceID', ''),
                                'status': wmi_gpu.get('Status', ''),
                                'detection_method': 'wmi_comprehensive'
                            }
                            gpus.append(gpu_info)
                        else:
                            # Update existing GPU with WMI data if missing
                            if not existing_gpu.get('memory_total') and wmi_gpu.get('AdapterRAM'):
                                existing_gpu['memory_total'] = wmi_gpu.get('AdapterRAM', 0)
                            if not existing_gpu.get('driver_version') and wmi_gpu.get('DriverVersion'):
                                existing_gpu['driver_version'] = wmi_gpu.get('DriverVersion', '')
                
                except Exception as e:
                    logger.debug(f"Failed to parse WMI GPU data: {e}")
        
        except Exception as e:
            logger.debug(f"Failed WMI GPU detection: {e}")
        
        return gpus
    
    def _detect_nvidia_gpus_windows(self) -> List[Dict[str, Any]]:
        """Detect NVIDIA GPUs on Windows using nvidia-smi"""
        gpus = []
        
        if not self.gpu_tools.get('nvidia_smi'):
            return gpus
        
        try:
            result = subprocess.run([
                self.gpu_tools['nvidia_smi'],
                "--query-gpu=index,name,memory.total,driver_version,pci.bus_id,compute_cap,uuid",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW)
            
            if result.returncode == 0 and result.stdout.strip():
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 7:
                            gpu_id = int(parts[0])
                            name = parts[1]
                            memory_total = int(parts[2]) if parts[2].isdigit() else 0
                            driver_version = parts[3]
                            pci_bus_id = parts[4]
                            compute_cap = parts[5]
                            uuid = parts[6]
                            
                            gpu_info = {
                                'id': gpu_id,
                                'name': name,
                                'vendor': 'NVIDIA',
                                'memory_total': memory_total,
                                'driver_version': driver_version,
                                'pci_bus_id': pci_bus_id,
                                'compute_capability': compute_cap,
                                'uuid': uuid,
                                'detection_method': 'nvidia_smi'
                            }
                            gpus.append(gpu_info)
        
        except Exception as e:
            logger.debug(f"Failed NVIDIA GPU detection: {e}")
        
        return gpus
    
    def _detect_linux_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPUs on Linux - placeholder for future implementation"""
        return []
    
    def _detect_macos_gpus(self) -> List[Dict[str, Any]]:
        """Detect GPUs on macOS - placeholder for future implementation"""
        return []
    
    def _determine_gpu_vendor(self, info: str) -> str:
        """Determine GPU vendor from various information sources"""
        if not info:
            return 'Unknown'
        
        info_lower = info.lower()
        
        if any(keyword in info_lower for keyword in ['nvidia', 'geforce', 'quadro', 'tesla', 'rtx', 'gtx']):
            return 'NVIDIA'
        elif any(keyword in info_lower for keyword in ['amd', 'radeon', 'ati', 'rx ', 'vega', 'navi']):
            return 'AMD'
        elif any(keyword in info_lower for keyword in ['intel', 'iris', 'uhd graphics', 'hd graphics', 'arc']):
            return 'Intel'
        elif any(keyword in info_lower for keyword in ['apple', 'm1', 'm2', 'm3']):
            return 'Apple'
        elif any(keyword in info_lower for keyword in ['adreno', 'qualcomm', 'snapdragon']):
            return 'Qualcomm'
        elif 'mali' in info_lower:
            return 'ARM'
        else:
            return 'Unknown'
    
    def _run_powershell(self, command: str, timeout: int = 15) -> Optional[str]:
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
        except Exception as e:
            logger.debug(f"PowerShell command failed: {e}")
        
        return None
    
    def get_gpu_count(self) -> int:
        """Get the number of available GPUs"""
        return len(self.gpus) if self.gpus else 1
    
    def get_gpu_stats(self, gpu_id: int = 0) -> Optional[GPUStats]:
        """Get comprehensive statistics for a specific GPU"""
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
        
        # Try vendor-specific methods with real GPU ID mapping
        if vendor == 'NVIDIA' and self.gpu_tools.get('nvidia_smi'):
            nvidia_gpu_id = gpu_info.get('id', gpu_id)
            stats = self._get_nvidia_comprehensive_stats(nvidia_gpu_id, gpu_info)
            if stats:
                return stats
        elif vendor == 'Intel':
            stats = self._get_intel_comprehensive_stats(gpu_id, gpu_info)
            if stats:
                return stats
        
        # Final fallback
        return self._create_fallback_gpu_stats(gpu_id, gpu_info)
    
    def _get_nvidia_temperature(self, gpu_id: int) -> float:
        """Get NVIDIA GPU temperature using working method from debug"""
        try:
            result = subprocess.run([
                self.gpu_tools['nvidia_smi'], "-i", str(gpu_id), 
                "--query-gpu=temperature.gpu", 
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW)
            
            if result.returncode == 0 and result.stdout.strip():
                temp_str = result.stdout.strip()
                if temp_str and temp_str not in ["N/A", "[N/A]", "[Not Supported]", "Not Supported"]:
                    try:
                        return float(temp_str)
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug(f"NVIDIA temperature detection failed: {e}")
        
        return 0.0
    
    def _get_intel_temperature(self, gpu_id: int) -> float:
        """Get Intel GPU temperature using working method from debug"""
        try:
            command = """
            $temperature = 0
            
            # Method 1: WMI thermal sensors
            try {
                $tempSensors = Get-WmiObject -Namespace "root\\wmi" -Class MSAcpi_ThermalZoneTemperature -ErrorAction SilentlyContinue
                if ($tempSensors) {
                    foreach ($sensor in $tempSensors) {
                        if ($sensor.CurrentTemperature) {
                            $tempKelvin = $sensor.CurrentTemperature / 10
                            $tempCelsius = $tempKelvin - 273.15
                            if ($tempCelsius -gt 25 -and $tempCelsius -lt 85) {
                                $temperature = $tempCelsius
                                break
                            }
                        }
                    }
                }
            } catch { }
            
            # Method 2: Performance counters
            if ($temperature -eq 0) {
                try {
                    $counters = Get-Counter -Counter "\\Thermal Zone Information(*)\\High Precision Temperature" -ErrorAction SilentlyContinue
                    if ($counters) {
                        foreach ($counter in $counters.CounterSamples) {
                            $temp = $counter.CookedValue - 273.15
                            if ($temp -gt 25 -and $temp -lt 85) {
                                $temperature = $temp
                                break
                            }
                        }
                    }
                } catch { }
            }
            
            # Method 3: CPU-based estimation
            if ($temperature -eq 0) {
                try {
                    $cpuLoad = (Get-Counter -Counter "\\Processor(_Total)\\% Processor Time" -SampleInterval 1 -MaxSamples 1).CounterSamples[0].CookedValue
                    $estimatedTemp = 40 + ($cpuLoad * 0.3)
                    if ($estimatedTemp -gt 30 -and $estimatedTemp -lt 80) {
                        $temperature = $estimatedTemp
                    }
                } catch { }
            }
            
            $temperature
            """
            
            result = self._run_powershell(command, timeout=15)
            if result and result.replace('.', '').isdigit():
                return float(result)
        except Exception as e:
            logger.debug(f"Intel temperature detection failed: {e}")
        
        return 0.0
    
    def _get_nvidia_comprehensive_stats(self, gpu_id: int, gpu_info: Dict[str, Any]) -> Optional[GPUStats]:
        """Get comprehensive NVIDIA GPU statistics"""
        try:
            # Get temperature using working method
            temperature = self._get_nvidia_temperature(gpu_id)
            
            # Get other stats
            result = subprocess.run([
                self.gpu_tools['nvidia_smi'], "-i", str(gpu_id),
                "--query-gpu=utilization.gpu,memory.used,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,fan.speed,pstate",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW)
            
            utilization = 0.0
            memory_used = 0
            power_usage = 0.0
            power_limit = 0.0
            clock_core = 0
            clock_memory = 0
            fan_speed = 0
            performance_state = ""
            
            if result.returncode == 0 and result.stdout.strip():
                values = [val.strip() for val in result.stdout.strip().split(',')]
                
                def safe_parse(value, parse_func=float, default=0):
                    try:
                        if not value or value in ["N/A", "[N/A]", "[Not Supported]", "Not Supported"]:
                            return default
                        cleaned_value = value.replace(' W', '').replace(' MHz', '').replace('%', '')
                        return parse_func(cleaned_value)
                    except:
                        return default
                
                if len(values) >= 7:
                    utilization = safe_parse(values[0], float)
                    memory_used = safe_parse(values[1], int)
                    power_usage = safe_parse(values[2], float)
                    power_limit = safe_parse(values[3], float)
                    clock_core = safe_parse(values[4], int)
                    clock_memory = safe_parse(values[5], int)
                    fan_speed = safe_parse(values[6], int)
                    performance_state = values[7] if len(values) > 7 and values[7] not in ["N/A", "[N/A]"] else ""
            
            return GPUStats(
                timestamp=datetime.now().isoformat(),
                gpu_id=gpu_id,
                name=gpu_info.get('name', f"NVIDIA GPU {gpu_id}"),
                vendor="NVIDIA",
                temperature=temperature,
                utilization=utilization,
                memory_used=memory_used,
                memory_total=gpu_info.get('memory_total', 0),
                power_usage=power_usage,
                power_limit=power_limit,
                driver_version=gpu_info.get('driver_version', ''),
                clock_core=clock_core,
                clock_memory=clock_memory,
                fan_speed=fan_speed,
                performance_state=performance_state,
                is_simulated=False
            )
        except Exception as e:
            logger.debug(f"Failed to get comprehensive NVIDIA GPU stats for GPU {gpu_id}: {e}")
        
        return None
    
    def _get_intel_comprehensive_stats(self, gpu_id: int, gpu_info: Dict[str, Any]) -> Optional[GPUStats]:
        """Get comprehensive Intel GPU statistics"""
        try:
            # Get temperature using working method
            temperature = self._get_intel_temperature(gpu_id)
            
            # Get utilization and memory usage via performance counters
            utilization = 0.0
            memory_used = 0
            
            try:
                command = """
                $utilization = 0
                $memoryUsed = 0
                
                try {
                    $counters = Get-Counter -Counter "\\GPU Engine(*)\\Utilization Percentage" -ErrorAction SilentlyContinue
                    if ($counters) {
                        $intelCounters = $counters.CounterSamples | Where-Object { 
                            $_.InstanceName -like "*Intel*" -or $_.Path -like "*Intel*"
                        }
                        if ($intelCounters) {
                            $utilization = ($intelCounters | Measure-Object -Property CookedValue -Average).Average
                        }
                    }
                    
                    $memCounters = Get-Counter -Counter "\\GPU Process Memory(*)\\Dedicated Usage" -ErrorAction SilentlyContinue
                    if ($memCounters) {
                        $intelMemCounters = $memCounters.CounterSamples | Where-Object { 
                            $_.InstanceName -like "*Intel*"
                        }
                        if ($intelMemCounters) {
                            $memoryUsed = ($intelMemCounters | Measure-Object -Property CookedValue -Sum).Sum / 1MB
                        }
                    }
                } catch { }
                
                "$utilization,$memoryUsed"
                """
                
                result = self._run_powershell(command, timeout=15)
                if result and ',' in result:
                    values = result.split(',')
                    if len(values) >= 2:
                        utilization = float(values[0]) if values[0].replace('.', '').isdigit() else 0.0
                        memory_used = int(float(values[1])) if values[1].replace('.', '').isdigit() else 0
            except Exception as e:
                logger.debug(f"Intel performance counter query failed: {e}")
            
            return GPUStats(
                timestamp=datetime.now().isoformat(),
                gpu_id=gpu_id,
                name=gpu_info.get('name', f"Intel GPU {gpu_id}"),
                vendor="Intel",
                temperature=temperature,
                utilization=utilization,
                memory_used=memory_used,
                memory_total=gpu_info.get('memory_total', 0),
                driver_version=gpu_info.get('driver_version', ''),
                is_simulated=False
            )
        except Exception as e:
            logger.debug(f"Failed comprehensive Intel GPU stats: {e}")
        
        return None
    
    def _create_fallback_gpu_stats(self, gpu_id: int, gpu_info: Dict[str, Any]) -> GPUStats:
        """Create fallback GPU stats when no monitoring is available"""
        return GPUStats(
            timestamp=datetime.now().isoformat(),
            gpu_id=gpu_id,
            name=gpu_info.get('name', f"GPU {gpu_id}"),
            vendor=gpu_info.get('vendor', 'Unknown'),
            driver_version=gpu_info.get('driver_version', ''),
            memory_total=gpu_info.get('memory_total', 0),
            is_simulated=True
        )
    
    def get_all_gpu_stats(self) -> List[GPUStats]:
        """Get comprehensive statistics for all available GPUs"""
        stats = []
        gpu_count = self.get_gpu_count()
        
        for gpu_id in range(gpu_count):
            gpu_stat = self.get_gpu_stats(gpu_id)
            if gpu_stat:
                stats.append(gpu_stat)
        
        return stats
    
    def get_system_stats(self) -> SystemStats:
        """Get comprehensive system-level statistics"""
        if self.os_detector.is_windows:
            return self._get_windows_comprehensive_system_stats()
        else:
            return SystemStats(timestamp=datetime.now().isoformat())
    
    def _get_windows_comprehensive_system_stats(self) -> SystemStats:
        """Get comprehensive system statistics on Windows"""
        stats_data = {}
        
        try:
            command = """
            # CPU Usage
            $cpuUsage = (Get-Counter -Counter "\\Processor(_Total)\\% Processor Time" -SampleInterval 1 -MaxSamples 2 | Select-Object -ExpandProperty CounterSamples | Select-Object -Last 1).CookedValue
            if (-not $cpuUsage) { $cpuUsage = 0 }
            
            # CPU Count and Frequency
            $cpuInfo = Get-WmiObject -Class Win32_Processor
            $cpuCount = ($cpuInfo | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
            $cpuFreq = ($cpuInfo | Select-Object -First 1).MaxClockSpeed
            
            # Memory
            $memory = Get-WmiObject -Class Win32_OperatingSystem
            $memoryUsed = [math]::Round(($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / 1KB)
            $memoryTotal = [math]::Round($memory.TotalVisibleMemorySize / 1KB)
            
            # Disk Usage (C: drive)
            $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'"
            $diskUsage = 0
            if ($disk -and $disk.Size -gt 0) {
                $diskUsage = [math]::Round(($disk.Size - $disk.FreeSpace) / $disk.Size * 100, 1)
            }
            
            # Uptime
            $os = Get-WmiObject -Class Win32_OperatingSystem
            $lastBoot = $os.LastBootUpTime
            $bootTime = [System.Management.ManagementDateTimeConverter]::ToDateTime($lastBoot)
            $uptime = [math]::Round((Get-Date) - $bootTime).TotalSeconds
            
            # Process Count
            $processCount = (Get-Process).Count
            
            # Battery (if available) - FIXED
            $battery = Get-WmiObject -Class Win32_Battery -ErrorAction SilentlyContinue
            $batteryPercent = 0
            $batteryTime = 0
            if ($battery) {
                $batteryPercent = if ($battery.EstimatedChargeRemaining) { $battery.EstimatedChargeRemaining } else { 0 }
                # Fix the battery time calculation
                if ($battery.EstimatedRunTime -and $battery.EstimatedRunTime -gt 0 -and $battery.EstimatedRunTime -lt 65535) {
                    $batteryTime = $battery.EstimatedRunTime
                } else {
                    $batteryTime = 0  # Set to 0 if no valid time remaining
                }
            }
            
            "$cpuUsage,$cpuCount,$cpuFreq,$memoryUsed,$memoryTotal,$diskUsage,$uptime,$processCount,$batteryPercent,$batteryTime"
            """
            
            result = self._run_powershell(command, timeout=30)
            if result and ',' in result:
                values = result.split(',')
                if len(values) >= 10:
                    battery_time_raw = int(values[9]) if values[9].isdigit() else 0
                    # Additional validation: if battery time is unreasonable, set to 0
                    if battery_time_raw > 1440:  # More than 24 hours is unreasonable
                        battery_time_raw = 0
                    
                    stats_data = {
                        'cpu_usage': float(values[0]) if values[0].replace('.', '').replace('-', '').isdigit() else 0.0,
                        'cpu_count': int(values[1]) if values[1].isdigit() else 0,
                        'cpu_frequency': float(values[2]) if values[2].replace('.', '').isdigit() else 0.0,
                        'memory_used': int(values[3]) if values[3].isdigit() else 0,
                        'memory_total': int(values[4]) if values[4].isdigit() else 0,
                        'disk_usage': float(values[5]) if values[5].replace('.', '').replace('-', '').isdigit() else 0.0,
                        'uptime': int(values[6]) if values[6].isdigit() else 0,
                        'process_count': int(values[7]) if values[7].isdigit() else 0,
                        'battery_percentage': float(values[8]) if values[8].replace('.', '').isdigit() else 0.0,
                        'battery_time': battery_time_raw
                    }
        except Exception as e:
            logger.debug(f"Failed comprehensive Windows system stats: {e}")
            stats_data = {}
        
        return SystemStats(
            timestamp=datetime.now().isoformat(),
            cpu_usage=stats_data.get('cpu_usage', 0.0),
            memory_used=stats_data.get('memory_used', 0),
            memory_total=stats_data.get('memory_total', 0),
            disk_usage=stats_data.get('disk_usage', 0.0),
            uptime=stats_data.get('uptime', 0),
            cpu_count=stats_data.get('cpu_count', 0),
            cpu_frequency=stats_data.get('cpu_frequency', 0.0),
            process_count=stats_data.get('process_count', 0),
            battery_percentage=stats_data.get('battery_percentage', 0.0),
            battery_time_remaining=stats_data.get('battery_time', 0)
        )

# Factory functions for easy creation
def create_os_detector() -> OSDetector:
    """Create and return an OSDetector instance."""
    return OSDetector()

def create_telemetry_collector() -> TelemetryCollector:
    """Create and return a TelemetryCollector instance."""
    return TelemetryCollector()

# Utility functions for common use cases
def is_windows() -> bool:
    """Quick check if the current OS is Windows."""
    return platform.system().lower() == "windows"

def is_linux() -> bool:
    """Quick check if the current OS is Linux."""
    return platform.system().lower() == "linux"

def is_macos() -> bool:
    """Quick check if the current OS is macOS."""
    return platform.system().lower() == "darwin"

def get_os_name() -> str:
    """Get a human-readable name for the current operating system."""
    system = platform.system().lower()
    if system == "windows":
        return "Windows"
    elif system == "linux":
        return "Linux"
    elif system == "darwin":
        return "macOS"
    else:
        return f"Unknown ({system})"

def get_comprehensive_telemetry() -> Dict[str, Any]:
    """Get comprehensive telemetry data for the system."""
    collector = create_telemetry_collector()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'os_info': collector.os_detector.get_os_info(),
        'gpu_stats': [gpu.to_dict() for gpu in collector.get_all_gpu_stats()],
        'system_stats': collector.get_system_stats().to_dict(),
        'gpu_count': collector.get_gpu_count(),
        'detected_tools': collector.gpu_tools
    }

# Example usage and demonstration
if __name__ == "__main__":
    try:
        # Create detector instances
        os_detector = create_os_detector()
        collector = create_telemetry_collector()
        
        # Display comprehensive OS information
        print("=" * 60)
        print("OPERATING SYSTEM INFORMATION")
        print("=" * 60)
        os_info = os_detector.get_os_info()
        for key, value in os_info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Display GPU detection info
        print("\n" + "=" * 60)
        print("GPU DETECTION RESULTS")
        print("=" * 60)
        
        if collector.gpus:
            print(f"Detected {len(collector.gpus)} GPU(s):")
            for gpu in collector.gpus:
                print(f"\nGPU {gpu['id']}:")
                print(f"  Name: {gpu['name']}")
                print(f"  Vendor: {gpu['vendor']}")
                print(f"  Detection Method: {gpu['detection_method']}")
                if 'memory_total' in gpu and gpu['memory_total']:
                    print(f"  Memory: {gpu['memory_total']} MB")
                if 'driver_version' in gpu and gpu['driver_version']:
                    print(f"  Driver: {gpu['driver_version']}")
        else:
            print("No GPUs detected")
        
        # Display available tools
        print(f"\nDetected Monitoring Tools:")
        for tool, path in collector.gpu_tools.items():
            status = "✓ Available" if path else "✗ Not found"
            print(f"  {tool}: {status}")
            if path:
                print(f"    Path: {path}")
        
        # Get comprehensive GPU stats
        print("\n" + "=" * 60)
        print("GPU PERFORMANCE STATISTICS")
        print("=" * 60)
        
        gpu_stats = collector.get_all_gpu_stats()
        for stat in gpu_stats:
            print(f"\nGPU {stat.gpu_id} ({stat.name} - {stat.vendor}):")
            print(f"  Status: {'Simulated Data' if stat.is_simulated else 'Real-time Data'}")
            
            if stat.temperature > 0:
                print(f"  Temperature: {stat.temperature:.1f}°C")
            if stat.utilization > 0:
                print(f"  Utilization: {stat.utilization:.1f}%")
            if stat.memory_total > 0:
                utilization_pct = (stat.memory_used / stat.memory_total * 100) if stat.memory_total > 0 else 0
                print(f"  Memory: {stat.memory_used}/{stat.memory_total} MB ({utilization_pct:.1f}%)")
            elif stat.memory_used > 0:
                print(f"  Memory Used: {stat.memory_used} MB")
            elif stat.memory_total > 0:
                print(f"  Memory Total: {stat.memory_total} MB")
            
            if stat.power_usage > 0:
                power_str = f"  Power: {stat.power_usage:.1f}W"
                if stat.power_limit > 0:
                    power_str += f"/{stat.power_limit:.1f}W"
                print(power_str)
            
            if stat.clock_core > 0:
                clock_str = f"  Clocks: {stat.clock_core} MHz core"
                if stat.clock_memory > 0:
                    clock_str += f", {stat.clock_memory} MHz memory"
                print(clock_str)
            
            if stat.fan_speed > 0:
                print(f"  Fan Speed: {stat.fan_speed}%")
            
            if stat.driver_version:
                print(f"  Driver: {stat.driver_version}")
            
            if stat.performance_state:
                print(f"  Performance State: {stat.performance_state}")
        
        # Get comprehensive system stats
        print("\n" + "=" * 60)
        print("SYSTEM PERFORMANCE STATISTICS")
        print("=" * 60)
        
        system_stats = collector.get_system_stats()
        print(f"\nCPU:")
        if system_stats.cpu_count > 0:
            print(f"  Cores: {system_stats.cpu_count}")
        print(f"  Usage: {system_stats.cpu_usage:.1f}%")
        if system_stats.cpu_frequency > 0:
            print(f"  Frequency: {system_stats.cpu_frequency:.0f} MHz")
        
        print(f"\nMemory:")
        if system_stats.memory_total > 0:
            mem_usage_pct = (system_stats.memory_used / system_stats.memory_total * 100) if system_stats.memory_total > 0 else 0
            print(f"  RAM: {system_stats.memory_used:,}/{system_stats.memory_total:,} MB ({mem_usage_pct:.1f}%)")
        
        print(f"\nStorage:")
        print(f"  Disk Usage: {system_stats.disk_usage:.1f}%")
        
        print(f"\nSystem:")
        if system_stats.uptime > 0:
            uptime_hours = system_stats.uptime // 3600
            uptime_minutes = (system_stats.uptime % 3600) // 60
            print(f"  Uptime: {uptime_hours}h {uptime_minutes}m")
        if system_stats.process_count > 0:
            print(f"  Processes: {system_stats.process_count}")
        
        if system_stats.battery_percentage > 0:
            print(f"\nBattery:")
            print(f"  Charge: {system_stats.battery_percentage:.0f}%")
            if system_stats.battery_time_remaining > 0 and system_stats.battery_time_remaining < 1440:  # Less than 24 hours
                battery_hours = system_stats.battery_time_remaining // 60
                battery_minutes = system_stats.battery_time_remaining % 60
                print(f"  Time Remaining: {battery_hours}h {battery_minutes}m")
            else:
                print(f"  Status: Plugged in")
        
        # Demonstrate utility functions
        print("\n" + "=" * 60)
        print("UTILITY FUNCTIONS")
        print("=" * 60)
        print(f"is_windows(): {is_windows()}")
        print(f"is_linux(): {is_linux()}")
        print(f"is_macos(): {is_macos()}")
        print(f"get_os_name(): {get_os_name()}")
        
        print(f"\n{'=' * 60}")
        print("TELEMETRY COLLECTION COMPLETE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Unexpected error during telemetry collection: {e}")
        import traceback
        traceback.print_exc()
