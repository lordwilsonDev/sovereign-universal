#!/usr/bin/env python3
"""
🌡️ HARDWARE TELEMETRY SNIFFER
==============================
Python bridge to macOS system metrics.

Captures the "Physical Reality" of the Mac Mini:
- CPU Usage
- Memory Pressure  
- Thermal State
- Disk I/O

Uses psutil for cross-platform basics + macOS-specific calls for thermal.
"""

import os
import sys
import time
import subprocess
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from datetime import datetime


# ============================================================================
# 📊 SYSTEM METRICS
# ============================================================================

@dataclass
class SystemMetrics:
    """Real-time system metrics from hardware"""
    cpu_usage: float           # 0-100%
    memory_used_gb: float      # GB used
    memory_available_gb: float # GB available
    memory_pressure: str       # "normal", "warning", "critical"
    thermal_state: str         # "nominal", "fair", "serious", "critical"
    disk_read_mb: float        # MB/s
    disk_write_mb: float       # MB/s
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def is_hot(self) -> bool:
        """Check if system is thermally constrained"""
        return self.thermal_state in ("serious", "critical")
    
    @property
    def is_memory_constrained(self) -> bool:
        """Check if memory pressure is high (< 2GB available on 16GB system)"""
        return self.memory_available_gb < 2.0 or self.memory_pressure != "normal"
    
    @property
    def abundance_violation(self) -> bool:
        """Check if system state violates Abundance Axiom"""
        return self.is_hot or self.is_memory_constrained or self.cpu_usage > 85
    
    def to_dict(self) -> dict:
        return {
            "cpu_usage": self.cpu_usage,
            "memory_used_gb": self.memory_used_gb,
            "memory_available_gb": self.memory_available_gb,
            "memory_pressure": self.memory_pressure,
            "thermal_state": self.thermal_state,
            "disk_read_mb": self.disk_read_mb,
            "disk_write_mb": self.disk_write_mb,
            "timestamp": self.timestamp,
            "abundance_violation": self.abundance_violation
        }


# ============================================================================
# 🔌 TELEMETRY SNIFFER
# ============================================================================

class TelemetrySniffer:
    """
    Captures system hardware metrics for the Sovereign Stack.
    
    Translates "Voltage" and "Pressure" into System Awareness.
    """
    
    # Memory pressure thresholds (for 16GB system)
    MEMORY_WARNING_GB = 4.0    # Less than 4GB available = warning
    MEMORY_CRITICAL_GB = 2.0   # Less than 2GB available = critical
    
    def __init__(self):
        self._last_metrics: Optional[SystemMetrics] = None
        self._disk_last_read = 0
        self._disk_last_write = 0
        self._disk_last_time = time.time()
    
    def sniff(self) -> SystemMetrics:
        """Capture current system metrics"""
        
        # CPU Usage
        cpu_usage = self._get_cpu_usage()
        
        # Memory
        mem_used, mem_available = self._get_memory()
        mem_pressure = self._calculate_memory_pressure(mem_available)
        
        # Thermal
        thermal = self._get_thermal_state()
        
        # Disk I/O
        disk_read, disk_write = self._get_disk_io()
        
        metrics = SystemMetrics(
            cpu_usage=cpu_usage,
            memory_used_gb=mem_used,
            memory_available_gb=mem_available,
            memory_pressure=mem_pressure,
            thermal_state=thermal,
            disk_read_mb=disk_read,
            disk_write_mb=disk_write
        )
        
        self._last_metrics = metrics
        return metrics
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            # Use top command for CPU usage
            result = subprocess.run(
                ["top", "-l", "1", "-n", "0"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            for line in result.stdout.split('\n'):
                if 'CPU usage' in line:
                    # Parse: "CPU usage: 12.34% user, 5.67% sys, 81.99% idle"
                    parts = line.split(',')
                    if len(parts) >= 2:
                        user = float(parts[0].split(':')[1].strip().replace('%', '').split()[0])
                        sys = float(parts[1].strip().replace('%', '').split()[0])
                        return user + sys
        except:
            pass
        
        return 0.0
    
    def _get_memory(self) -> Tuple[float, float]:
        """Get memory usage in GB"""
        try:
            # Use vm_stat for memory info
            result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
            
            page_size = 16384  # Default for Apple Silicon
            
            # Parse vm_stat output
            stats = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':')
                    try:
                        stats[key.strip()] = int(value.strip().replace('.', ''))
                    except:
                        pass
            
            # Calculate memory
            wired = stats.get('Pages wired down', 0) * page_size
            active = stats.get('Pages active', 0) * page_size
            inactive = stats.get('Pages inactive', 0) * page_size
            free = stats.get('Pages free', 0) * page_size
            
            used_gb = (wired + active) / (1024 ** 3)
            available_gb = (inactive + free) / (1024 ** 3)
            
            # Also check with sysctl for total
            try:
                hw_result = subprocess.run(
                    ["sysctl", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                total_bytes = int(hw_result.stdout.split(':')[1].strip())
                total_gb = total_bytes / (1024 ** 3)
                # Recalculate available based on total
                available_gb = total_gb - used_gb
            except:
                pass
            
            return used_gb, available_gb
            
        except Exception as e:
            return 0.0, 16.0  # Default to 16GB available on error
    
    def _calculate_memory_pressure(self, available_gb: float) -> str:
        """Calculate memory pressure level"""
        if available_gb < self.MEMORY_CRITICAL_GB:
            return "critical"
        elif available_gb < self.MEMORY_WARNING_GB:
            return "warning"
        return "normal"
    
    def _get_thermal_state(self) -> str:
        """Get thermal state from macOS"""
        try:
            # Use powermetrics for thermal (requires sudo for full access)
            # Fallback to simple check
            result = subprocess.run(
                ["pmset", "-g", "therm"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            output = result.stdout.lower()
            if "cpu_speed_limit" in output:
                # Parse speed limit percentage
                for line in output.split('\n'):
                    if 'cpu_speed_limit' in line:
                        try:
                            value = int(line.split(':')[1].strip())
                            if value == 100:
                                return "nominal"
                            elif value >= 80:
                                return "fair"
                            elif value >= 50:
                                return "serious"
                            else:
                                return "critical"
                        except:
                            pass
            
            return "nominal"
            
        except:
            return "nominal"
    
    def _get_disk_io(self) -> Tuple[float, float]:
        """Get disk I/O rates in MB/s"""
        try:
            result = subprocess.run(
                ["iostat", "-c", "1"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 3:
                # Parse last line for disk0
                parts = lines[-1].split()
                if len(parts) >= 4:
                    read_kbs = float(parts[2])
                    write_kbs = float(parts[3])
                    return read_kbs / 1024, write_kbs / 1024
                    
        except:
            pass
        
        return 0.0, 0.0
    
    def print_status(self):
        """Print formatted status"""
        metrics = self.sniff()
        
        # Status icons
        thermal_icon = "🔥" if metrics.is_hot else "❄️"
        memory_icon = "⚠️" if metrics.is_memory_constrained else "✅"
        abundance_icon = "❌" if metrics.abundance_violation else "✅"
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🌡️ HARDWARE TELEMETRY                                       ║
╠══════════════════════════════════════════════════════════════╣
║  CPU:     {metrics.cpu_usage:5.1f}%                                         ║
║  Memory:  {metrics.memory_used_gb:5.1f} GB used / {metrics.memory_available_gb:5.1f} GB available  {memory_icon}      ║
║  Thermal: {metrics.thermal_state:<8} {thermal_icon}                                      ║
║  Disk:    {metrics.disk_read_mb:5.1f} MB/s read, {metrics.disk_write_mb:5.1f} MB/s write           ║
╠══════════════════════════════════════════════════════════════╣
║  Abundance Axiom: {abundance_icon}                                          ║
╚══════════════════════════════════════════════════════════════╝
""")
        return metrics


# ============================================================================
# 🔄 THERMAL INVERSION ADVISOR
# ============================================================================

class ThermalInversionAdvisor:
    """
    Generates hardware-aware prompts for MoIE.
    
    When Abundance Axiom is violated:
    - Thermal: HOT → Suggest lazy evaluation, reduce compute
    - Memory: HIGH → Suggest streaming, reduce allocation
    - CPU: OVERLOADED → Suggest async, offload to GPU
    """
    
    INVERSION_STRATEGIES = {
        "thermal_hot": [
            "Use lazy evaluation to reduce CPU cycles",
            "Defer computations that aren't immediately needed",
            "Consider Metal-accelerated operations to reduce CPU load",
            "Implement result caching to avoid recomputation"
        ],
        "memory_high": [
            "Use streaming/generators instead of loading all data",
            "Implement memory pooling for frequent allocations",
            "Clear unused references aggressively",
            "Consider memory-mapped files for large data"
        ],
        "cpu_high": [
            "Offload parallel operations to Metal/GPU",
            "Use async I/O to free up CPU cycles",
            "Implement early termination for expensive loops",
            "Consider batch processing with sleep intervals"
        ]
    }
    
    def __init__(self):
        self.sniffer = TelemetrySniffer()
    
    def generate_context(self) -> str:
        """Generate hardware context for MoIE prompt"""
        metrics = self.sniffer.sniff()
        
        context = f"""HARDWARE CONTEXT:
CPU: {metrics.cpu_usage:.0f}% Load | Thermal: {metrics.thermal_state.upper()} | Available RAM: {metrics.memory_available_gb:.1f}GB

"""
        
        if metrics.abundance_violation:
            context += "⚠️ ABUNDANCE AXIOM VIOLATION DETECTED\n\n"
            context += "MISSION: The current system state violates the Abundance Axiom.\n"
            context += "INVERSION TASK: Apply efficiency optimizations.\n\n"
            context += "RECOMMENDED INVERSIONS:\n"
            
            if metrics.is_hot:
                for strategy in self.INVERSION_STRATEGIES["thermal_hot"]:
                    context += f"  • {strategy}\n"
            
            if metrics.is_memory_constrained:
                for strategy in self.INVERSION_STRATEGIES["memory_high"]:
                    context += f"  • {strategy}\n"
            
            if metrics.cpu_usage > 85:
                for strategy in self.INVERSION_STRATEGIES["cpu_high"]:
                    context += f"  • {strategy}\n"
            
            context += "\nReduce resource footprint by 40% or the Executioner will terminate.\n"
        else:
            context += "✅ Hardware operating within Abundance Axiom bounds.\n"
        
        return context
    
    def should_throttle(self) -> bool:
        """Check if MoIE should throttle operations"""
        metrics = self.sniffer.sniff()
        return metrics.abundance_violation
    
    def get_throttle_factor(self) -> float:
        """Get throttle factor (0.0 = full throttle, 1.0 = no throttle)"""
        metrics = self.sniffer.sniff()
        
        factor = 1.0
        
        if metrics.is_hot:
            factor *= 0.7
        
        if metrics.is_memory_constrained:
            factor *= 0.8
        
        if metrics.cpu_usage > 85:
            factor *= 0.6
        
        return factor


# ============================================================================
# 📊 METRICS JSON EXPORT (for Swift/Dashboard)
# ============================================================================

def export_metrics_json(output_path: str = "/tmp/sovereign_telemetry.json"):
    """Export metrics as JSON for Swift dashboard consumption"""
    sniffer = TelemetrySniffer()
    metrics = sniffer.sniff()
    
    with open(output_path, 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    
    return output_path


# ============================================================================
# 🧪 MAIN
# ============================================================================

if __name__ == "__main__":
    print("🌡️ HARDWARE TELEMETRY SNIFFER TEST\n")
    
    sniffer = TelemetrySniffer()
    advisor = ThermalInversionAdvisor()
    
    # Show current status
    sniffer.print_status()
    
    # Show MoIE context
    print("MoIE Hardware Context:")
    print("-" * 40)
    print(advisor.generate_context())
    
    # Export for dashboard
    json_path = export_metrics_json()
    print(f"\n📁 Metrics exported to: {json_path}")
    
    print("\n✅ TELEMETRY SNIFFER TEST COMPLETE")
