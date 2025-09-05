#!/usr/bin/env python3
"""
Operating System Detection Module
Provides a clean, modular way to detect and work with different operating systems.
"""

import platform
import sys
from enum import Enum, auto
from typing import Optional, Dict, Any

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


# Factory function for easy creation
def create_os_detector() -> OSDetector:
    """
    Create and return an OSDetector instance.
    
    Returns:
        OSDetector instance
    """
    return OSDetector()


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
    # Create detector instance
    detector = OSDetector()
    
    # Display OS information
    print(f"Detected OS: {detector.get_os_name()}")
    print(f"OS Type: {detector.os_type.name}")
    print(f"System: {detector.system}")
    print(f"Release: {detector.release}")
    print(f"Version: {detector.version}")
    
    # Check specific OS types
    print(f"\nIs Windows: {detector.is_windows}")
    print(f"Is Linux: {detector.is_linux}")
    print(f"Is macOS: {detector.is_macos}")
    print(f"Is Unknown: {detector.is_unknown}")
    
    # Show comprehensive info
    print(f"\nComprehensive OS Info:")
    for key, value in detector.get_os_info().items():
        print(f"  {key}: {value}")
    
    # Demonstrate utility functions
    print(f"\nUtility function checks:")
    print(f"  is_windows(): {is_windows()}")
    print(f"  is_linux(): {is_linux()}")
    print(f"  is_macos(): {is_macos()}")
    print(f"  get_os_name(): {get_os_name()}")
#!/usr/bin/env python3
"""
Windows Telemetry Collection Module with Enhanced NVIDIA Support
Provides Windows-specific telemetry collection for GPU and system metrics.
"""

import platform
import subprocess
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class WindowsTelemetryCollector:
    """Collects telemetry data on Windows systems with enhanced NVIDIA support"""
    
    def __init__(self):
        """Initialize the Windows telemetry collector."""
        if platform.system() != "Windows":
            raise RuntimeError("This collector only works on Windows systems")
        
        self.nvidia_smi_path = self._find_nvidia_smi()
        self.nvidia_gpus = self._detect_nvidia_gpus()
        
        if self.nvidia_smi_path:
            logger.info(f"Found nvidia-smi at: {self.nvidia_smi_path}")
            if self.nvidia_gpus:
                logger.info(f"Detected {len(self.nvidia_gpus)} NVIDIA GPU(s)")
    
    def _find_nvidia_smi(self) -> Optional[str]:
        """Find nvidia-smi executable path on Windows with comprehensive search"""
        # Check if nvidia-smi is in PATH
        nvidia_smi = self._which('nvidia-smi')
        if nvidia_smi:
            return nvidia_smi
            
        # Common NVIDIA installation paths
        possible_paths = [
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
            r"C:\Windows\System32\nvidia-smi.exe",
            r"C:\Program Files\NVIDIA Corporation\DisplayDriver\*\nvidia-smi.exe",  # Wildcard for versioned paths
        ]
        
        # Add potential driver paths for different drives
        for drive in ['C:', 'D:', 'E:']:
            for program_files in ['Program Files', 'Program Files (x86)']:
                possible_paths.extend([
                    fr"{drive}\{program_files}\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                    fr"{drive}\{program_files}\NVIDIA Corporation\DisplayDriver\*\nvidia-smi.exe",
                    fr"{drive}\{program_files}\NVIDIA Corporation\NVIDIA GeForce Experience\nvidia-smi.exe",
                ])
        
        # Expand wildcards and check all paths
        import glob
        for path_pattern in possible_paths:
            try:
                for path in glob.glob(path_pattern):
                    if os.path.exists(path):
                        return path
            except:
                continue
                    
        logger.warning("nvidia-smi not found. NVIDIA GPU monitoring will be limited.")
        return None
    
    def _detect_nvidia_gpus(self) -> List[Dict[str, Any]]:
        """Detect NVIDIA GPUs using multiple methods"""
        nvidia_gpus = []
        
        # Method 1: Check via nvidia-smi
        if self.nvidia_smi_path:
            try:
                result = subprocess.run(
                    [self.nvidia_smi_path, "--query-gpu=index,name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n'):
                        if line.strip():
                            parts = line.split(',', 1)
                            if len(parts) >= 2:
                                gpu_id = parts[0].strip()
                                name = parts[1].strip()
                                nvidia_gpus.append({
                                    'id': int(gpu_id),
                                    'name': name,
                                    'detection_method': 'nvidia-smi'
                                })
            except Exception as e:
                logger.debug(f"Failed to detect NVIDIA GPUs via nvidia-smi: {e}")
        
        # Method 2: Check via WMI
        if not nvidia_gpus:
            try:
                command = """
                Get-WmiObject -Class Win32_VideoController | Where-Object { 
                    $_.Name -like "*NVIDIA*" -or $_.AdapterCompatibility -like "*NVIDIA*" 
                } | Select-Object -Property Name, @{Name="Index"; Expression={$_.Index}} | ConvertTo-Json
                """
                result = self._run_powershell(command)
                if result:
                    gpus = json.loads(result) if result.startswith('[') else [json.loads(result)]
                    for i, gpu in enumerate(gpus):
                        nvidia_gpus.append({
                            'id': i,
                            'name': gpu.get('Name', f'NVIDIA GPU {i}'),
                            'detection_method': 'wmi'
                        })
            except Exception as e:
                logger.debug(f"Failed to detect NVIDIA GPUs via WMI: {e}")
        
        return nvidia_gpus
    
    def _which(self, program: str) -> Optional[str]:
        """Windows implementation of Unix which command"""
        try:
            result = subprocess.run(["where", program], capture_output=True, text=True, timeout=5,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split('\n')[0]
        except:
            pass
        return None
    
    def _run_powershell(self, command: str, timeout: int = 10) -> Optional[str]:
        """Execute a PowerShell command and return the output"""
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
            else:
                logger.debug(f"PowerShell command failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"PowerShell command timed out: {command}")
        except Exception as e:
            logger.error(f"Error executing PowerShell command: {e}")
        return None
    
    def get_gpu_count(self) -> int:
        """Get the number of available GPUs"""
        # Try NVIDIA detection first
        if self.nvidia_gpus:
            return len(self.nvidia_gpus)
        
        # Fallback to WMI
        command = "Get-WmiObject -Class Win32_VideoController | Where-Object {$_.AdapterDACType -ne 'Unknown'} | Measure-Object | Select-Object -ExpandProperty Count"
        result = self._run_powershell(command)
        
        if result and result.isdigit():
            count = int(result)
            if count > 0:
                return count
        
        # Last resort
        logger.warning("Unable to detect GPUs, returning 1 as a safe default")
        return 1
    
    def get_gpu_stats(self, gpu_id: int = 0) -> Optional[GPUStats]:
        """
        Get statistics for a specific GPU
        
        Args:
            gpu_id: The ID of the GPU to query
            
        Returns:
            GPUStats object or None if failed
        """
        # Check if this is an NVIDIA GPU
        is_nvidia = any(gpu['id'] == gpu_id for gpu in self.nvidia_gpus)
        
        # Try NVIDIA-specific methods first if available
        if is_nvidia and self.nvidia_smi_path:
            nvidia_stats = self._get_nvidia_smi_gpu_stats(gpu_id)
            if nvidia_stats:
                return nvidia_stats
        
        # Fall back to WMI-based methods
        return self._get_windows_wmi_gpu_stats(gpu_id)
    
    def _get_nvidia_smi_gpu_stats(self, gpu_id: int) -> Optional[GPUStats]:
        """Get comprehensive statistics from NVIDIA GPU using nvidia-smi command"""
        try:
            # First verify this GPU exists
            list_result = subprocess.run(
                [self.nvidia_smi_path, "--list-gpus"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            if list_result.returncode != 0:
                return None
                
            gpus = list_result.stdout.strip().split('\n')
            if gpu_id >= len(gpus):
                return None
            
            # Get comprehensive stats in JSON format for better parsing
            result = subprocess.run([
                self.nvidia_smi_path, 
                f"-i", f"{gpu_id}", 
                "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,driver_version,clocks.current.graphics,clocks.current.memory,fan.speed",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            
            if result.returncode == 0 and result.stdout.strip():
                values = [val.strip() for val in result.stdout.strip().split(',')]
                
                if len(values) >= 12:
                    # Parse values with proper error handling for [N/A] values
                    def safe_float_parse(value, default=0):
                        try:
                            # Handle [N/A] and other non-numeric values
                            if not value or value == "N/A" or value == "[N/A]" or not value.replace('.', '').replace('-', '').isdigit():
                                return default
                            return float(value)
                        except (ValueError, AttributeError):
                            return default
                    
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
                        timestamp=datetime.now().isoformat(),
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
            
            # Fallback to simpler query if comprehensive one fails
            return self._get_nvidia_smi_simple_stats(gpu_id)
            
        except Exception as e:
            logger.error(f"Failed to get NVIDIA GPU stats via nvidia-smi: {e}")
            return None
    
    def _get_nvidia_smi_simple_stats(self, gpu_id: int) -> Optional[GPUStats]:
        """Get basic NVIDIA stats as fallback"""
        try:
            # Get basic stats
            result = subprocess.run([
                self.nvidia_smi_path, 
                f"-i", f"{gpu_id}", 
                "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            
            if result.returncode == 0 and result.stdout.strip():
                values = [val.strip() for val in result.stdout.strip().split(',')]
                
                if len(values) >= 5:
                    # Handle [N/A] values
                    def safe_parse(value, default=0):
                        try:
                            if not value or value == "N/A" or value == "[N/A]" or not value.replace('.', '').replace('-', '').isdigit():
                                return default
                            return int(float(value))
                        except (ValueError, AttributeError):
                            return default
                    
                    name = values[0]
                    temp = safe_parse(values[1])
                    util = safe_parse(values[2])
                    mem_used = safe_parse(values[3])
                    mem_total = safe_parse(values[4])
                    
                    return GPUStats(
                        timestamp=datetime.now().isoformat(),
                        gpu_id=gpu_id,
                        name=name,
                        vendor="NVIDIA",
                        temperature=temp,
                        utilization=util,
                        memory_used=mem_used,
                        memory_total=mem_total,
                        is_simulated=False
                    )
        except Exception as e:
            logger.debug(f"Failed to get basic NVIDIA GPU stats: {e}")
        
        return None
    
    def _get_windows_wmi_gpu_stats(self, gpu_id: int) -> Optional[GPUStats]:
        """Get GPU statistics on Windows using WMI"""
        try:
            # Get basic GPU information
            command = f"""
            $gpu = (Get-WmiObject -Class Win32_VideoController)[{gpu_id}]
            $gpu | Select-Object -Property Name,AdapterCompatibility,AdapterRAM,DriverVersion | ConvertTo-Json
            """
            
            result = self._run_powershell(command)
            if not result:
                return None
                
            data = json.loads(result)
            name = data.get("Name", f"GPU {gpu_id}")
            vendor = data.get("AdapterCompatibility", "Unknown")
            
            # Get memory info
            memory_total = 0
            adapter_ram = data.get("AdapterRAM")
            if adapter_ram:
                try:
                    memory_total = int(adapter_ram) // (1024 * 1024)
                except:
                    pass
            
            # Get driver version
            driver_version = data.get("DriverVersion", "")
            
            # Try to get performance metrics
            utilization = self._get_gpu_utilization(gpu_id)
            memory_used = self._get_gpu_memory_usage(gpu_id)
            temperature = self._get_gpu_temperature(gpu_id, vendor)
            
            return GPUStats(
                timestamp=datetime.now().isoformat(),
                gpu_id=gpu_id,
                name=name,
                vendor=vendor,
                temperature=temperature,
                utilization=utilization,
                memory_used=memory_used,
                memory_total=memory_total,
                driver_version=driver_version,
                is_simulated=False
            )
            
        except Exception as e:
            logger.error(f"Failed to get WMI GPU stats: {e}")
            return None
    
    def _get_gpu_utilization(self, gpu_id: int) -> int:
        """Get GPU utilization percentage"""
        command = f"""
        $utilization = 0
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
        }} catch {{ $utilization = 0 }}
        $utilization
        """
        
        result = self._run_powershell(command)
        return int(result) if result and result.isdigit() else 0
    
    def _get_gpu_memory_usage(self, gpu_id: int) -> int:
        """Get GPU memory usage in MB"""
        command = f"""
        $memoryUsed = 0
        try {{
            $counters = Get-Counter -Counter "\\GPU Process Memory(*)\\Local Usage" -ErrorAction SilentlyContinue
            if ($counters) {{
                foreach ($counter in $counters.CounterSamples) {{
                    $currentGpuId = [int]($counter.InstanceName.Split('_')[0])
                    if ($currentGpuId -eq {gpu_id}) {{
                        $memoryUsed += $counter.CookedValue
                    }}
                }}
                $memoryUsed = [math]::Round($memoryUsed / 1MB)
            }}
        }} catch {{ $memoryUsed = 0 }}
        $memoryUsed
        """
        
        result = self._run_powershell(command)
        return int(result) if result and result.isdigit() else 0
    
    def _get_gpu_temperature(self, gpu_id: int, vendor: str) -> int:
        """Get GPU temperature in Celsius"""
        # For NVIDIA GPUs, try nvidia-smi first
        if vendor.lower() == "nvidia" and self.nvidia_smi_path:
            try:
                result = subprocess.run([
                    self.nvidia_smi_path,
                    f"-i", f"{gpu_id}",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits"
                ], capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
                
                if result.returncode == 0 and result.stdout.strip():
                    temp = result.stdout.strip()
                    if temp != "N/A" and temp.replace('.', '').replace('-', '').isdigit():
                        return int(float(temp))
            except:
                pass
        
        # Generic fallback (not always available on Windows)
        return 0
    
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
        # Get CPU usage
        cpu_usage = self._get_cpu_usage()
        
        # Get memory usage
        memory_used, memory_total = self._get_memory_usage()
        
        # Get disk usage
        disk_usage = self._get_disk_usage()
        
        # Get uptime
        uptime = self._get_uptime()
        
        return SystemStats(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_usage=disk_usage,
            uptime=uptime
        )
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        command = """
        $cpuUsage = 0
        try {
            $cpuUsage = (Get-WmiObject -Class Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
            if (-not $cpuUsage) { 
                $counter = Get-Counter -Counter "\\Processor(_Total)\% Processor Time" -SampleInterval 1 -MaxSamples 1 -ErrorAction SilentlyContinue
                if ($counter) { $cpuUsage = $counter.CounterSamples.CookedValue }
            }
        } catch { $cpuUsage = 0 }
        $cpuUsage
        """
        
        result = self._run_powershell(command)
        try:
            return float(result) if result else 0.0
        except:
            return 0.0
    
    def _get_memory_usage(self) -> Tuple[int, int]:
        """Get memory usage in MB (used, total)"""
        command = """
        $used = 0
        $total = 0
        try {
            $memory = Get-WmiObject -Class Win32_OperatingSystem
            $used = [math]::Round(($memory.TotalVisibleMemorySize - $memory.FreePhysicalMemory) / 1KB)
            $total = [math]::Round($memory.TotalVisibleMemorySize / 1KB)
        } catch { }
        "$used,$total"
        """
        
        result = self._run_powershell(command)
        if result and ',' in result:
            try:
                used, total = result.split(',')
                return int(float(used)), int(float(total))
            except:
                pass
        return 0, 0
    
    def _get_disk_usage(self) -> int:
        """Get disk usage percentage for the system drive"""
        command = """
        $diskUsage = 0
        try {
            $disk = Get-WmiObject -Class Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction SilentlyContinue
            if ($disk -and $disk.Size -gt 0) {
                $diskUsage = [math]::Round(($disk.Size - $disk.FreeSpace) / $disk.Size * 100)
            }
        } catch { }
        $diskUsage
        """
        
        result = self._run_powershell(command)
        return int(result) if result and result.isdigit() else 0
    
    def _get_uptime(self) -> int:
        """Get system uptime in seconds"""
        command = """
    $uptime = 0
    try {
        $os = Get-WmiObject -Class Win32_OperatingSystem
        $lastBoot = $os.LastBootUpTime
        $bootTime = [System.Management.ManagementDateTimeConverter]::ToDateTime($lastBoot)
        $uptime = [math]::Round((Get-Date) - $bootTime).TotalSeconds
    } catch { 
        # Alternative method using performance counters
        try {
            $counter = Get-Counter -Counter "\\System\\System Up Time" -ErrorAction SilentlyContinue
            if ($counter) { $uptime = [math]::Round($counter.CounterSamples.CookedValue) }
        } catch { }
    }
    $uptime
    """
    
        result = self._run_powershell(command)
        return int(result) if result and result.isdigit() else 0


# Factory function for easy creation
def create_windows_telemetry_collector() -> WindowsTelemetryCollector:
    """
    Create and return a WindowsTelemetryCollector instance.
    
    Returns:
        WindowsTelemetryCollector instance
    """
    return WindowsTelemetryCollector()


# Example usage and demonstration
if __name__ == "__main__":
    try:
        # Create collector instance
        collector = create_windows_telemetry_collector()
        
        # Display NVIDIA detection info
        if collector.nvidia_smi_path:
            print(f"nvidia-smi found: {collector.nvidia_smi_path}")
        if collector.nvidia_gpus:
            print(f"Detected {len(collector.nvidia_gpus)} NVIDIA GPU(s):")
            for gpu in collector.nvidia_gpus:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['detection_method']})")
        
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
        
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
Linux Telemetry Collection Module
Provides Linux-specific telemetry collection for GPU and system metrics.
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
from typing import List, Dict, Any, Optional, Tuple
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class LinuxTelemetryCollector:
    """Collects telemetry data on Linux systems with NVIDIA and AMD GPU support"""
    
    def __init__(self):
        """Initialize the Linux telemetry collector."""
        if platform.system() != "Linux":
            raise RuntimeError("This collector only works on Linux systems")
        
        self.nvidia_smi_path = self._find_nvidia_smi()
        self.rocm_smi_path = self._find_rocm_smi()
        self.gpus = self._detect_gpus()
        
        if self.nvidia_smi_path:
            logger.info(f"Found nvidia-smi at: {self.nvidia_smi_path}")
        if self.rocm_smi_path:
            logger.info(f"Found rocm-smi at: {self.rocm_smi_path}")
        if self.gpus:
            logger.info(f"Detected {len(self.gpus)} GPU(s)")
    
    def _find_nvidia_smi(self) -> Optional[str]:
        """Find nvidia-smi executable path on Linux"""
        # Check if nvidia-smi is in PATH
        try:
            result = subprocess.run(['which', 'nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except:
            pass
        
        # Check common locations
        common_paths = [
            '/usr/bin/nvidia-smi',
            '/opt/nvidia/bin/nvidia-smi',
            '/usr/local/bin/nvidia-smi',
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    def _find_rocm_smi(self) -> Optional[str]:
        """Find rocm-smi executable path on Linux for AMD GPUs"""
        # Check if rocm-smi is in PATH
        try:
            result = subprocess.run(['which', 'rocm-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except:
            pass
        
        # Check common ROCm locations
        common_paths = [
            '/opt/rocm/bin/rocm-smi',
            '/usr/bin/rocm-smi',
            '/opt/rocm/opencl/bin/x86_64/rocm-smi',
        ]
        
        for path in common_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path
        
        return None
    
    def _detect_gpus(self) -> List[Dict[str, Any]]:
        """Detect available GPUs using multiple methods"""
        gpus = []
        
        # Method 1: Check PCI devices
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'VGA' in line or '3D' in line or 'Display' in line:
                        gpu_info = {
                            'id': len(gpus),
                            'name': line.split(': ')[-1] if ': ' in line else line,
                            'vendor': self._determine_gpu_vendor(line),
                            'detection_method': 'lspci'
                        }
                        gpus.append(gpu_info)
        except:
            pass
        
        # Method 2: Check NVIDIA GPUs via nvidia-smi
        if self.nvidia_smi_path:
            try:
                result = subprocess.run(
                    [self.nvidia_smi_path, "--query-gpu=index,name", "--format=csv,noheader"],
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
            except:
                pass
        
        # Method 3: Check AMD GPUs via rocm-smi
        if self.rocm_smi_path:
            try:
                result = subprocess.run(
                    [self.rocm_smi_path, "--showuniqueid"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Parse rocm-smi output to detect AMD GPUs
                    lines = result.stdout.strip().split('\n')
                    for i, line in enumerate(lines):
                        if 'card' in line.lower() and 'unique id' in line.lower():
                            if not any(g['vendor'] == 'AMD' for g in gpus):
                                gpus.append({
                                    'id': len(gpus),
                                    'name': f"AMD GPU {i}",
                                    'vendor': 'AMD',
                                    'detection_method': 'rocm-smi'
                                })
            except:
                pass
        
        return gpus
    
    def _determine_gpu_vendor(self, lspci_line: str) -> str:
        """Determine GPU vendor from lspci output"""
        line_lower = lspci_line.lower()
        if 'nvidia' in line_lower:
            return 'NVIDIA'
        elif 'amd' in line_lower or 'ati' in line_lower or 'radeon' in line_lower:
            return 'AMD'
        elif 'intel' in line_lower:
            return 'Intel'
        else:
            return 'Unknown'
    
    def get_gpu_count(self) -> int:
        """Get the number of available GPUs"""
        if self.gpus:
            return len(self.gpus)
        
        # Fallback methods
        try:
            # Try to count NVIDIA GPUs
            if self.nvidia_smi_path:
                result = subprocess.run(
                    [self.nvidia_smi_path, "--list-gpus"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return len(result.stdout.strip().split('\n'))
            
            # Try to count via lspci
            result = subprocess.run(
                ['lspci'], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                count = 0
                for line in result.stdout.split('\n'):
                    if 'VGA' in line or '3D' in line or 'Display' in line:
                        count += 1
                return count
        except:
            pass
        
        # Last resort - assume at least 1 GPU
        return 1
    
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
        
        # Try vendor-specific methods first
        if vendor == 'NVIDIA' and self.nvidia_smi_path:
            nvidia_stats = self._get_nvidia_smi_gpu_stats(gpu_id)
            if nvidia_stats:
                return nvidia_stats
        elif vendor == 'AMD' and self.rocm_smi_path:
            amd_stats = self._get_amd_rocm_gpu_stats(gpu_id)
            if amd_stats:
                return amd_stats
        
        # Fall back to generic methods
        return self._get_generic_gpu_stats(gpu_id, gpu_info)
    
    def _get_nvidia_smi_gpu_stats(self, gpu_id: int) -> Optional[GPUStats]:
        """Get comprehensive statistics from NVIDIA GPU using nvidia-smi"""
        try:
            # Get comprehensive stats in CSV format
            result = subprocess.run([
                self.nvidia_smi_path,
                f"-i", f"{gpu_id}",
                "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,driver_version,clocks.current.graphics,clocks.current.memory,fan.speed",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and result.stdout.strip():
                values = [val.strip() for val in result.stdout.strip().split(',')]
                
                if len(values) >= 12:
                    # Parse values with proper error handling
                    def safe_float_parse(value, default=0):
                        try:
                            if not value or value == "N/A" or value == "[N/A]" or not value.replace('.', '').replace('-', '').isdigit():
                                return default
                            return float(value)
                        except (ValueError, AttributeError):
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
            logger.error(f"Failed to get NVIDIA GPU stats via nvidia-smi: {e}")
        
        return None
    
    def _get_amd_rocm_gpu_stats(self, gpu_id: int) -> Optional[GPUStats]:
        """Get statistics from AMD GPU using rocm-smi"""
        try:
            # Get temperature
            temp_result = subprocess.run([
                self.rocm_smi_path, "--showtemp", "-d", str(gpu_id)
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
                self.rocm_smi_path, "--showuse", "-d", str(gpu_id)
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
                self.rocm_smi_path, "--showmemuse", "-d", str(gpu_id)
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
            logger.error(f"Failed to get AMD GPU stats via rocm-smi: {e}")
        
        return None
    
    def _get_generic_gpu_stats(self, gpu_id: int, gpu_info: Dict[str, Any]) -> GPUStats:
        """Get generic GPU statistics using system files and commands"""
        # Try to get temperature from sysfs (common on Linux)
        temperature = 0
        try:
            # Check common temperature file locations
            temp_paths = [
                f"/sys/class/drm/card{gpu_id}/device/hwmon/hwmon*/temp1_input",
                f"/sys/class/drm/card{gpu_id}/device/temp1_input",
                f"/sys/class/hwmon/hwmon*/temp1_input"
            ]
            
            for path_pattern in temp_paths:
                import glob
                for path in glob.glob(path_pattern):
                    try:
                        with open(path, 'r') as f:
                            temp_millic = int(f.read().strip())
                            temperature = temp_millic // 1000  # Convert millicelsius to celsius
                            break
                    except:
                        continue
                if temperature > 0:
                    break
        except:
            pass
        
        # Get basic info from GPU info
        name = gpu_info.get('name', f"GPU {gpu_id}")
        vendor = gpu_info.get('vendor', 'Unknown')
        
        return GPUStats(
            timestamp=datetime.now().isoformat(),
            gpu_id=gpu_id,
            name=name,
            vendor=vendor,
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
        # Get CPU usage
        cpu_usage = self._get_cpu_usage()
        
        # Get memory usage
        memory_used, memory_total = self._get_memory_usage()
        
        # Get disk usage
        disk_usage = self._get_disk_usage()
        
        # Get uptime
        uptime = self._get_uptime()
        
        return SystemStats(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_usage=disk_usage,
            uptime=uptime
        )
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage using /proc/stat"""
        try:
            # Read first line of /proc/stat
            with open('/proc/stat', 'r') as f:
                first_line = f.readline().strip()
            
            if first_line.startswith('cpu '):
                values = first_line.split()[1:]
                if len(values) >= 4:
                    user, nice, system, idle = map(int, values[:4])
                    total = user + nice + system + idle
                    
                    # Wait a short time and read again
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
                                usage = 100.0 * (total_delta - idle_delta) / total_delta
                                return max(0.0, min(100.0, usage))
        except:
            pass
        
        return 0.0
    
    def _get_memory_usage(self) -> Tuple[int, int]:
        """Get memory usage in MB from /proc/meminfo"""
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            total = 0
            free = 0
            available = 0
            
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    total = int(line.split()[1]) // 1024  # Convert kB to MB
                elif line.startswith('MemFree:'):
                    free = int(line.split()[1]) // 1024
                elif line.startswith('MemAvailable:'):
                    available = int(line.split()[1]) // 1024
            
            # Prefer available memory over free memory
            memory_used = total - (available if available > 0 else free)
            return memory_used, total
            
        except:
            return 0, 0
    
    def _get_disk_usage(self) -> int:
        """Get disk usage percentage for the root filesystem"""
        try:
            result = subprocess.run(['df', '/'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        usage_str = parts[4].replace('%', '')
                        return int(usage_str)
        except:
            pass
        
        return 0
    
    def _get_uptime(self) -> int:
        """Get system uptime in seconds from /proc/uptime"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.read().split()[0])
                return int(uptime_seconds)
        except:
            return 0


# Factory function for easy creation
def create_linux_telemetry_collector() -> LinuxTelemetryCollector:
    """
    Create and return a LinuxTelemetryCollector instance.
    
    Returns:
        LinuxTelemetryCollector instance
    """
    return LinuxTelemetryCollector()


# Example usage and demonstration
if __name__ == "__main__":
    try:
        # Create collector instance
        collector = create_linux_telemetry_collector()
        
        # Display detection info
        if collector.nvidia_smi_path:
            print(f"nvidia-smi found: {collector.nvidia_smi_path}")
        if collector.rocm_smi_path:
            print(f"rocm-smi found: {collector.rocm_smi_path}")
        if collector.gpus:
            print(f"Detected {len(collector.gpus)} GPU(s):")
            for gpu in collector.gpus:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['vendor']} - {gpu['detection_method']})")
        
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
        
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
#!/usr/bin/env python3
"""
macOS Telemetry Collection Module
Provides macOS-specific telemetry collection for GPU and system metrics.
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
from typing import List, Dict, Any, Optional, Tuple
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

class MacOSTelemetryCollector:
    """Collects telemetry data on macOS systems with Apple Silicon and Intel GPU support"""
    
    def __init__(self):
        """Initialize the macOS telemetry collector."""
        if platform.system() != "Darwin":
            raise RuntimeError("This collector only works on macOS systems")
        
        self.gpus = self._detect_gpus()
        self._system_profiler_data = None
        
        if self.gpus:
            logger.info(f"Detected {len(self.gpus)} GPU(s)")
    
    def _detect_gpus(self) -> List[Dict[str, Any]]:
        """Detect available GPUs using system_profiler"""
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
                            gpu_info = {
                                'id': len(gpus),
                                'name': display_info.get('sppci_model', 'Unknown GPU'),
                                'vendor': self._determine_gpu_vendor(display_info),
                                'detection_method': 'system_profiler',
                                'raw_data': display_info
                            }
                            gpus.append(gpu_info)
        except Exception as e:
            logger.error(f"Failed to detect GPUs via system_profiler: {e}")
        
        # Fallback: check if we're on Apple Silicon which has integrated GPU
        if not gpus:
            try:
                # Check if this is Apple Silicon
                result = subprocess.run([
                    'sysctl', '-n', 'machdep.cpu.brand_string'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0 and 'Apple' in result.stdout:
                    gpus.append({
                        'id': 0,
                        'name': 'Apple Integrated GPU',
                        'vendor': 'Apple',
                        'detection_method': 'sysctl',
                        'is_integrated': True
                    })
            except:
                pass
        
        return gpus
    
    def _determine_gpu_vendor(self, display_info: Dict[str, Any]) -> str:
        """Determine GPU vendor from system_profiler data"""
        model = display_info.get('sppci_model', '').lower()
        chipset = display_info.get('sppci_chipset_model', '').lower()
        
        if 'apple' in model or 'apple' in chipset:
            return 'Apple'
        elif 'amd' in model or 'radeon' in model:
            return 'AMD'
        elif 'nvidia' in model or 'geforce' in model:
            return 'NVIDIA'
        elif 'intel' in model:
            return 'Intel'
        else:
            return 'Unknown'
    
    def _get_system_profiler_data(self) -> Dict[str, Any]:
        """Get and cache system_profiler data"""
        if self._system_profiler_data is None:
            try:
                result = subprocess.run([
                    'system_profiler', '-json'
                ], capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0:
                    self._system_profiler_data = json.loads(result.stdout)
                else:
                    self._system_profiler_data = {}
            except:
                self._system_profiler_data = {}
        
        return self._system_profiler_data
    
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
        name = gpu_info.get('name', f"GPU {gpu_id}")
        
        # Try to get GPU stats
        stats = self._get_gpu_stats_apple(gpu_id, gpu_info)
        if stats:
            return stats
        
        # Fallback to basic info
        return GPUStats(
            timestamp=datetime.now().isoformat(),
            gpu_id=gpu_id,
            name=name,
            vendor=vendor,
            is_simulated=False
        )
    
    def _get_gpu_stats_apple(self, gpu_id: int, gpu_info: Dict[str, Any]) -> Optional[GPUStats]:
        """Get GPU statistics for Apple GPUs"""
        try:
            # Try to get GPU utilization using ioreg
            utilization = 0
            try:
                result = subprocess.run([
                    'ioreg', '-c', 'IOAccelerator', '-r'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    # Parse ioreg output for utilization hints
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Utilization' in line:
                            try:
                                util_str = line.split('=')[-1].strip()
                                utilization = int(util_str)
                                break
                            except:
                                pass
            except:
                pass
            
            # Try to get GPU memory usage
            memory_used = 0
            memory_total = 0
            try:
                # Use vm_stat to get memory info (this is system-wide, not GPU-specific)
                result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    # This is a rough approximation
                    memory_used = self._parse_vm_stat_memory(result.stdout)
            except:
                pass
            
            # Get system memory total for context
            try:
                result = subprocess.run([
                    'sysctl', '-n', 'hw.memsize'
                ], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    memory_total = int(result.stdout.strip()) // (1024 * 1024)  # Convert bytes to MB
            except:
                pass
            
            # Get temperature (if available)
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
                # Fallback to generic temperature reading
                try:
                    result = subprocess.run([
                        'istats', 'extra'
                    ], capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        for line in result.stdout.split('\n'):
                            if 'GPU' in line and 'temperature' in line:
                                try:
                                    temp_str = line.split(':')[-1].strip().split()[0]
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
                vendor=gpu_info.get('vendor', 'Apple'),
                temperature=temperature,
                utilization=utilization,
                memory_used=memory_used,
                memory_total=memory_total,
                is_simulated=False
            )
            
        except Exception as e:
            logger.error(f"Failed to get Apple GPU stats: {e}")
            return None
    
    def _parse_vm_stat_memory(self, vm_stat_output: str) -> int:
        """Parse vm_stat output to estimate used memory in MB"""
        try:
            lines = vm_stat_output.split('\n')
            page_size = 4096  # Default page size on macOS
            
            # Get page size if available
            for line in lines:
                if 'page size of' in line.lower():
                    try:
                        page_size_str = line.split('page size of')[-1].split('bytes')[0].strip()
                        page_size = int(page_size_str)
                    except:
                        pass
            
            # Parse memory stats
            stats = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().replace('"', '')
                    value = value.strip().replace('.', '')
                    if value.isdigit():
                        stats[key] = int(value)
            
            # Calculate used memory (approximation)
            # This is system memory, not GPU-specific
            if 'Pages active' in stats and 'Pages wired' in stats:
                used_pages = stats.get('Pages active', 0) + stats.get('Pages wired', 0)
                return (used_pages * page_size) // (1024 * 1024)  # Convert to MB
        except:
            pass
        
        return 0
    
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
        # Get CPU usage
        cpu_usage = self._get_cpu_usage()
        
        # Get memory usage
        memory_used, memory_total = self._get_memory_usage()
        
        # Get disk usage
        disk_usage = self._get_disk_usage()
        
        # Get uptime
        uptime = self._get_uptime()
        
        return SystemStats(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_used=memory_used,
            memory_total=memory_total,
            disk_usage=disk_usage,
            uptime=uptime
        )
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage using top command"""
        try:
            # Use top to get CPU usage
            result = subprocess.run([
                'top', '-l', '1', '-n', '0'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CPU usage' in line:
                        try:
                            # Parse line like: "CPU usage: 10.0% user, 5.0% sys, 85.0% idle"
                            usage_str = line.split('CPU usage:')[-1].strip()
                            idle_str = [part for part in usage_str.split(',') if 'idle' in part][0]
                            idle_percent = float(idle_str.split('%')[0].strip())
                            return max(0.0, min(100.0, 100.0 - idle_percent))
                        except:
                            pass
        except:
            pass
        
        # Fallback method using sysctl
        try:
            # Get CPU load averages
            result = subprocess.run([
                'sysctl', '-n', 'vm.loadavg'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                # Format: { loadavg = [1.0, 2.0, 3.0] }
                load_str = result.stdout.strip()
                load_values = re.findall(r'[\d.]+', load_str)
                if len(load_values) >= 1:
                    # Convert 1-minute load average to approximate CPU usage
                    load_1min = float(load_values[0])
                    # This is a very rough approximation
                    return min(100.0, load_1min * 25.0)  # Scale factor is arbitrary
        except:
            pass
        
        return 0.0
    
    def _get_memory_usage(self) -> Tuple[int, int]:
        """Get memory usage in MB"""
        try:
            # Use vm_stat to get memory info
            result = subprocess.run(['vm_stat'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                memory_used = self._parse_vm_stat_memory(result.stdout)
            
            # Get total memory
            result = subprocess.run([
                'sysctl', '-n', 'hw.memsize'
            ], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                memory_total = int(result.stdout.strip()) // (1024 * 1024)  # Convert bytes to MB
                return memory_used, memory_total
        except:
            pass
        
        return 0, 0
    
    def _get_disk_usage(self) -> int:
        """Get disk usage percentage for the root filesystem"""
        try:
            result = subprocess.run(['df', '/'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split()
                    if len(parts) >= 5:
                        usage_str = parts[4].replace('%', '')
                        return int(usage_str)
        except:
            pass
        
        return 0
    
    def _get_uptime(self) -> int:
        """Get system uptime in seconds"""
        try:
            result = subprocess.run(['sysctl', '-n', 'kern.boottime'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Format: { sec = 1234567890, usec = 0 } ...
                boot_time_str = result.stdout.strip()
                boot_time_match = re.search(r'sec\s*=\s*(\d+)', boot_time_str)
                if boot_time_match:
                    boot_time = int(boot_time_match.group(1))
                    current_time = int(time.time())
                    return current_time - boot_time
        except:
            pass
        
        # Fallback method
        try:
            result = subprocess.run(['uptime'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse uptime output like: "up 1 day, 2:30"
                uptime_str = result.stdout.strip()
                if 'up' in uptime_str.lower():
                    # This is complex to parse, so just return a default
                    return 3600  # 1 hour as fallback
        except:
            pass
        
        return 0


# Factory function for easy creation
def create_macos_telemetry_collector() -> MacOSTelemetryCollector:
    """
    Create and return a MacOSTelemetryCollector instance.
    
    Returns:
        MacOSTelemetryCollector instance
    """
    return MacOSTelemetryCollector()


# Example usage and demonstration
if __name__ == "__main__":
    try:
        # Create collector instance
        collector = create_macos_telemetry_collector()
        
        # Display detection info
        if collector.gpus:
            print(f"Detected {len(collector.gpus)} GPU(s):")
            for gpu in collector.gpus:
                print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['vendor']} - {gpu['detection_method']})")
        
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
        
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()