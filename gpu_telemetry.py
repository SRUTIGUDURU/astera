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
