"""
Hardware detection focused solely on CPU and Memory.
Relies only on standard libraries and the 'psutil' package.
"""

import logging
import os
import platform
import subprocess
from typing import Optional

import psutil
# Use pydantic for clear data structures
from pydantic import BaseModel, Field, validator

# Basic logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Data Models ---


class CPUInfo(BaseModel):
    """Detailed CPU information."""

    logical_cores: int = Field(..., gt=0, description="Number of logical CPU cores.")
    physical_cores: int = Field(..., gt=0, description="Number of physical CPU cores.")
    architecture: str = Field(
        ..., description="CPU architecture (e.g., 'x86_64', 'arm64')."
    )
    model_name: str = Field(..., description="CPU model name.")
    frequency_mhz: Optional[float] = Field(
        None, description="Maximum or current CPU frequency in MHz."
    )
    # Basic instruction set detection (optional, can be expanded)
    supports_avx2: bool = Field(
        False, description="Indicates if AVX2 instructions are supported."
    )

    @validator("model_name", pre=True, always=True)
    def ensure_model_name_string(cls, v):
        # Ensure model name is always a string, even if platform returns None
        return str(v) if v is not None else "Unknown"


class MemoryInfo(BaseModel):
    """System memory (RAM) information."""

    total_gb: float = Field(..., gt=0, description="Total physical RAM in GiB.")
    available_gb: float = Field(
        ..., ge=0, description="Available RAM (usable by new processes) in GiB."
    )
    used_gb: float = Field(..., description="Used RAM in GiB.")
    percent_used: float = Field(
        ..., ge=0, le=100, description="Percentage of RAM currently used."
    )


class HardwareInfo(BaseModel):
    """Container for detected hardware information."""

    cpu: CPUInfo
    memory: MemoryInfo


# --- Detection Functions ---


def _detect_cpu_avx2() -> bool:
    """Attempts to detect AVX2 support."""
    # Prioritize py-cpuinfo if available (more reliable)
    try:
        import py_cpuinfo  # type: ignore

        info = py_cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        if isinstance(flags, list):
            return "avx2" in [flag.lower() for flag in flags]
        logger.debug("py_cpuinfo flags format unexpected, falling back.")
    except ImportError:
        logger.debug("py_cpuinfo not installed, falling back to basic detection.")
    except Exception as e:
        logger.debug(f"Error using py_cpuinfo: {e}, falling back.")

    # Fallback for Linux: Check /proc/cpuinfo
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                content = f.read()
            return " avx2 " in content  # Look for flag with spaces
        except Exception:
            logger.debug("Could not check /proc/cpuinfo for AVX2.")

    # Fallback for macOS: Check sysctl (less common to show specific flags)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "machdep.cpu.features"],  # Check this specific key
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return "AVX2" in result.stdout  # Case-sensitive check? Adjust if needed
        except Exception:
            logger.debug("Could not check sysctl machdep.cpu.features for AVX2.")

    # No reliable method found for Windows without py-cpuinfo or external tools
    logger.warning(
        "Could not reliably determine AVX2 support on this platform without py_cpuinfo."
    )
    return False


def detect_cpu_capabilities() -> CPUInfo:
    """
    Detects CPU capabilities.
    """
    logical_cores = os.cpu_count() or 1
    physical_cores = logical_cores  # Default assumption
    try:
        phys_count = psutil.cpu_count(logical=False)
        if phys_count:
            physical_cores = phys_count
            # Ensure logical isn't less than physical
            logical_cores = max(
                logical_cores, psutil.cpu_count(logical=True) or logical_cores
            )
        else:  # psutil returned None or 0 for physical
            physical_cores = logical_cores // 2 if logical_cores > 1 else 1
            logger.debug("psutil returned no physical core count, estimating.")
    except NotImplementedError:
        logger.warning(
            "psutil could not determine physical core count on this platform."
        )
        physical_cores = logical_cores // 2 if logical_cores > 1 else 1  # Estimate
    except Exception as e:
        logger.warning(f"Error getting physical cores: {e}. Estimating.")
        physical_cores = logical_cores // 2 if logical_cores > 1 else 1  # Estimate

    architecture = platform.machine().lower()
    model_name = "Unknown"
    system = platform.system()

    # Get CPU model name (using previous robust logic)
    try:
        if system == "Windows":
            model_name = platform.processor()
            if not model_name:
                try:
                    # Use shell=True cautiously, ensure command is safe
                    result = subprocess.run(
                        "wmic cpu get name",
                        shell=True,
                        capture_output=True,
                        text=True,
                        check=False,
                        creationflags=subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NO_WINDOW,
                    )
                    if (
                        result.returncode == 0
                        and result.stdout
                        and len(result.stdout.splitlines()) > 1
                    ):
                        model_name = result.stdout.splitlines()[1].strip()
                except Exception:
                    pass  # Ignore wmic errors
        elif system == "Darwin":
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                model_name = result.stdout.strip()
            except Exception:
                model_name = platform.processor()
        else:  # Linux
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("model name"):
                            model_name = line.split(":", 1)[1].strip()
                            break
                    if model_name == "Unknown":
                        model_name = platform.processor()
            except Exception:
                model_name = platform.processor()
        if not model_name:
            model_name = platform.processor() or "Unknown CPU"
    except Exception as e:
        logger.error(f"Error getting CPU model name: {e}")
        model_name = "Unknown CPU"

    # Get CPU frequency
    frequency_mhz = None
    try:
        freq = psutil.cpu_freq()
        if freq:
            frequency_mhz = (
                freq.max if hasattr(freq, "max") and freq.max else freq.current
            )
    except Exception:
        pass  # Ignore frequency errors

    # Detect AVX2 support
    supports_avx2 = _detect_cpu_avx2()

    return CPUInfo(
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        architecture=architecture,
        model_name=model_name.strip(),
        frequency_mhz=frequency_mhz,
        supports_avx2=supports_avx2,
    )


def detect_memory_info() -> MemoryInfo:
    """
    Detects system memory (RAM) information.
    """
    try:
        mem = psutil.virtual_memory()
        return MemoryInfo(
            total_gb=round(mem.total / (1024**3), 2),
            available_gb=round(mem.available / (1024**3), 2),
            used_gb=round(mem.used / (1024**3), 2),
            percent_used=mem.percent,
        )
    except Exception as e:
        logger.error(f"Failed to get memory info: {e}")
        return MemoryInfo(total_gb=0.0, available_gb=0.0, used_gb=0.0, percent_used=0.0)


# --- Main Public Function ---


def detect_hardware_info() -> HardwareInfo:
    """
    Detects and returns CPU and Memory information.

    Returns:
        HardwareInfo: An object containing detected CPU and Memory details.
    """
    logger.info("Detecting CPU and Memory hardware information...")
    cpu_info = detect_cpu_capabilities()
    memory_info = detect_memory_info()
    logger.info(
        f"CPU: {cpu_info.model_name} ({cpu_info.physical_cores}c/{cpu_info.logical_cores}t), AVX2: {cpu_info.supports_avx2}"
    )
    logger.info(
        f"Memory: {memory_info.total_gb:.1f} GB Total, {memory_info.available_gb:.1f} GB Available"
    )

    return HardwareInfo(cpu=cpu_info, memory=memory_info)


# Example usage (optional)
if __name__ == "__main__":
    hw_info = detect_hardware_info()
    print("\n--- Hardware Information ---")
    print(f"CPU Model:       {hw_info.cpu.model_name}")
    print(f"Architecture:    {hw_info.cpu.architecture}")
    print(f"Physical Cores:  {hw_info.cpu.physical_cores}")
    print(f"Logical Cores:   {hw_info.cpu.logical_cores}")
    print(f"Frequency (MHz): {hw_info.cpu.frequency_mhz or 'N/A'}")
    print(f"AVX2 Support:    {hw_info.cpu.supports_avx2}")
    print("-" * 20)
    print(f"Total RAM:       {hw_info.memory.total_gb:.2f} GB")
    print(f"Available RAM:   {hw_info.memory.available_gb:.2f} GB")
    print(
        f"Used RAM:        {hw_info.memory.used_gb:.2f} GB ({hw_info.memory.percent_used}%)"
    )
    print("-" * 20)
