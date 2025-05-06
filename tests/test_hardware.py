# tests/test_hardware.py

import unittest
import platform
import subprocess
import sys
from unittest.mock import patch, MagicMock, mock_open, ANY

# Assuming src layout, adjust if necessary
from llamasearch.hardware import (
    CPUInfo,
    MemoryInfo,
    HardwareInfo,
    _detect_cpu_avx2,
    detect_cpu_capabilities,
    detect_memory_info,
    detect_hardware_info,
)
import llamasearch.hardware as hw_module # Import module for patching

# --- Test Pydantic Models ---

class TestPydanticModels(unittest.TestCase):

    def test_cpu_info_creation(self):
        cpu = CPUInfo(
            logical_cores=8,
            physical_cores=4,
            architecture="x86_64",
            model_name="Test CPU",
            frequency_mhz=3000.0,
            supports_avx2=True,
        )
        self.assertEqual(cpu.logical_cores, 8)
        self.assertEqual(cpu.physical_cores, 4)
        self.assertEqual(cpu.architecture, "x86_64")
        self.assertEqual(cpu.model_name, "Test CPU")
        self.assertEqual(cpu.frequency_mhz, 3000.0)
        self.assertTrue(cpu.supports_avx2)

    def test_cpu_info_model_name_validator(self):
        cpu1 = CPUInfo(logical_cores=1, physical_cores=1, architecture="test", model_name=None, frequency_mhz=None, supports_avx2=False)
        self.assertEqual(cpu1.model_name, "Unknown")

        cpu2 = CPUInfo(logical_cores=1, physical_cores=1, architecture="test", model_name=" Real CPU ", frequency_mhz=None, supports_avx2=False)
        self.assertEqual(cpu2.model_name, " Real CPU ")


    def test_memory_info_creation(self):
        mem = MemoryInfo(
            total_gb=15.6, available_gb=8.1, used_gb=7.5, percent_used=48.1
        )
        self.assertEqual(mem.total_gb, 15.6)
        self.assertEqual(mem.available_gb, 8.1)
        self.assertEqual(mem.used_gb, 7.5)
        self.assertEqual(mem.percent_used, 48.1)

    def test_memory_info_creation_zero(self):
        mem = MemoryInfo(total_gb=0.0, available_gb=0.0, used_gb=0.0, percent_used=0.0)
        self.assertEqual(mem.total_gb, 0.0)
        self.assertEqual(mem.available_gb, 0.0)
        self.assertEqual(mem.used_gb, 0.0)
        self.assertEqual(mem.percent_used, 0.0)

    def test_hardware_info_creation(self):
        cpu = CPUInfo(logical_cores=2, physical_cores=1, architecture="arm64", model_name="M1", frequency_mhz=None, supports_avx2=False)
        mem = MemoryInfo(total_gb=8.0, available_gb=4.0, used_gb=4.0, percent_used=50.0)
        hw = HardwareInfo(cpu=cpu, memory=mem)
        self.assertIs(hw.cpu, cpu)
        self.assertIs(hw.memory, mem)

# --- Test Helper Functions ---

class TestDetectAVX2(unittest.TestCase):

    @patch('llamasearch.hardware.platform.system', return_value="OtherOS")
    @patch('builtins.open', new_callable=mock_open)
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_pycpuinfo_present_true(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection when py_cpuinfo is present and reports AVX2."""
        mock_pycpuinfo = MagicMock()
        mock_pycpuinfo.get_cpu_info.return_value = {'flags': ['sse', 'avx', 'avx2', 'fma']}
        with patch.dict(sys.modules, {'py_cpuinfo': mock_pycpuinfo}):
            self.assertTrue(_detect_cpu_avx2())
        mock_pycpuinfo.get_cpu_info.assert_called_once()
        mock_open_func.assert_not_called()
        mock_run.assert_not_called()


    @patch('llamasearch.hardware.platform.system', return_value="OtherOS")
    @patch('builtins.open', new_callable=mock_open)
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_pycpuinfo_present_false(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection when py_cpuinfo is present but doesn't report AVX2."""
        mock_pycpuinfo = MagicMock()
        mock_pycpuinfo.get_cpu_info.return_value = {'flags': ['sse', 'avx', 'fma']}
        with patch.dict(sys.modules, {'py_cpuinfo': mock_pycpuinfo}):
            self.assertFalse(_detect_cpu_avx2())
        mock_pycpuinfo.get_cpu_info.assert_called_once()
        mock_open_func.assert_not_called()
        mock_run.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Linux")
    @patch('builtins.open', new_callable=mock_open, read_data="flags: avx2")
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_pycpuinfo_raises_exception(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection fallback when py_cpuinfo raises an error."""
        mock_pycpuinfo = MagicMock()
        mock_pycpuinfo.get_cpu_info.side_effect = Exception("pycpuinfo failed")
        with patch.dict(sys.modules, {'py_cpuinfo': mock_pycpuinfo}):
            # <<< FIX: Check logs separately if assertLogs fails >>>
            with self.assertLogs(logger='llamasearch.hardware', level='DEBUG') as cm_debug:
                 result = _detect_cpu_avx2()
                 # Check DEBUG log first
                 self.assertTrue(any("Error using py_cpuinfo: pycpuinfo failed" in log for log in cm_debug.output))
            # Check result after logs
            self.assertTrue(result)
        mock_pycpuinfo.get_cpu_info.assert_called_once()
        mock_open_func.assert_called_once_with('/proc/cpuinfo', 'r', encoding='utf-8')
        mock_run.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Linux")
    @patch('builtins.open', new_callable=mock_open, read_data="flags\t\t: fpu vme de pse tsc ... avx fma ... avx2 bmi1 ...\nmore stuff")
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_pycpuinfo_missing_linux_true(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection fallback to /proc/cpuinfo on Linux (AVX2 present)."""
        with patch.dict(sys.modules, {'py_cpuinfo': None}):
             with self.assertLogs(logger='llamasearch.hardware', level='DEBUG') as cm:
                self.assertTrue(_detect_cpu_avx2())
                self.assertTrue(any("py_cpuinfo not installed" in log for log in cm.output))
        mock_open_func.assert_called_once_with('/proc/cpuinfo', 'r', encoding='utf-8')
        mock_run.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Linux")
    @patch('builtins.open', new_callable=mock_open, read_data="flags\t\t: fpu vme de pse tsc ... avx fma ... bmi1 ...\nmore stuff")
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_pycpuinfo_missing_linux_false(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection fallback to /proc/cpuinfo on Linux (AVX2 missing)."""
        with patch.dict(sys.modules, {'py_cpuinfo': None}):
            self.assertFalse(_detect_cpu_avx2())
        mock_open_func.assert_called_once_with('/proc/cpuinfo', 'r', encoding='utf-8')
        mock_run.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Linux")
    @patch('builtins.open', side_effect=OSError("Cannot open"))
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_linux_proc_fails(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection when /proc/cpuinfo fails on Linux."""
        with patch.dict(sys.modules, {'py_cpuinfo': None}):
            # <<< FIX: Check logs separately >>>
            with self.assertLogs(logger='llamasearch.hardware', level='DEBUG') as cm_debug:
                 result = _detect_cpu_avx2() # Run the function
                 self.assertTrue(any("Could not check /proc/cpuinfo" in log for log in cm_debug.output))
            with self.assertLogs(logger='llamasearch.hardware', level='WARNING') as cm_warn:
                 # Re-run or just check the result from the first run
                 self.assertFalse(result)
                 self.assertTrue(any("Could not reliably determine AVX2" in log for log in cm_warn.output))
        mock_open_func.assert_called_once_with('/proc/cpuinfo', 'r', encoding='utf-8')
        mock_run.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Darwin")
    @patch('builtins.open', new_callable=mock_open)
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_pycpuinfo_missing_mac_true(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection fallback to sysctl on macOS (AVX2 present)."""
        mock_run.return_value = MagicMock(
            stdout="machdep.cpu.features: FPU VME DE PSE TSC ... AVX1.0 RDRAND FMA AVX2 BMI1 ...",
            returncode=0
        )
        with patch.dict(sys.modules, {'py_cpuinfo': None}):
            self.assertTrue(_detect_cpu_avx2())
        mock_run.assert_called_once_with(
            ["sysctl", "machdep.cpu.features"], capture_output=True, text=True, check=False
        )
        mock_open_func.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Darwin")
    @patch('builtins.open', new_callable=mock_open)
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_pycpuinfo_missing_mac_false(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection fallback to sysctl on macOS (AVX2 missing)."""
        mock_run.return_value = MagicMock(
            stdout="machdep.cpu.features: FPU VME DE PSE TSC ... AVX1.0 RDRAND FMA BMI1 ...",
            returncode=0
        )
        with patch.dict(sys.modules, {'py_cpuinfo': None}):
            self.assertFalse(_detect_cpu_avx2())
        mock_run.assert_called_once_with(
            ["sysctl", "machdep.cpu.features"], capture_output=True, text=True, check=False
        )
        mock_open_func.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Darwin")
    @patch('builtins.open', new_callable=mock_open)
    @patch('llamasearch.hardware.subprocess.run', side_effect=FileNotFoundError("sysctl not found"))
    def test_avx2_mac_sysctl_fails(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection when sysctl fails on macOS."""
        with patch.dict(sys.modules, {'py_cpuinfo': None}):
            # <<< FIX: Check logs separately >>>
            with self.assertLogs(logger='llamasearch.hardware', level='DEBUG') as cm_debug:
                 result = _detect_cpu_avx2() # Run the function
                 self.assertTrue(any("Could not check sysctl" in log for log in cm_debug.output))
            with self.assertLogs(logger='llamasearch.hardware', level='WARNING') as cm_warn:
                 # Re-run or just check the result from the first run
                 self.assertFalse(result)
                 self.assertTrue(any("Could not reliably determine AVX2" in log for log in cm_warn.output))
        mock_run.assert_called_once_with(
            ["sysctl", "machdep.cpu.features"], capture_output=True, text=True, check=False
        )
        mock_open_func.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Windows")
    @patch('builtins.open', new_callable=mock_open)
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_pycpuinfo_missing_windows_false(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection fallback on Windows (should be False)."""
        with patch.dict(sys.modules, {'py_cpuinfo': None}):
            self.assertFalse(_detect_cpu_avx2())
        mock_run.assert_not_called()
        mock_open_func.assert_not_called()

# --- Test Detection Functions ---

@patch('llamasearch.hardware.psutil')
@patch('llamasearch.hardware.platform')
@patch('llamasearch.hardware.os')
@patch('llamasearch.hardware.subprocess.run')
@patch('llamasearch.hardware._detect_cpu_avx2')
@patch('builtins.open', new_callable=mock_open)
class TestDetectCPU(unittest.TestCase):

    def test_detect_cpu_macos(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test CPU detection on a simulated macOS system."""
        mock_os.cpu_count.return_value = 8
        mock_psutil.cpu_count.side_effect = [8, 8]
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        mock_platform.processor.return_value = "Fallback Proc Name"
        mock_run.side_effect = lambda *args, **kwargs: MagicMock(stdout="Apple M1 Max ", returncode=0) if args[0] == ["sysctl", "-n", "machdep.cpu.brand_string"] else MagicMock()
        mock_psutil.cpu_freq.return_value = None
        mock_detect_avx2.return_value = False

        cpu_info = detect_cpu_capabilities()

        self.assertEqual(cpu_info.logical_cores, 8)
        self.assertEqual(cpu_info.physical_cores, 8)
        self.assertEqual(cpu_info.architecture, "arm64")
        self.assertEqual(cpu_info.model_name, "Apple M1 Max")
        self.assertIsNone(cpu_info.frequency_mhz)
        self.assertFalse(cpu_info.supports_avx2)
        mock_run.assert_any_call(
             ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        )
        mock_open_func.assert_not_called()
        mock_detect_avx2.assert_called_once()

    def test_detect_cpu_macos_sysctl_fail(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test CPU model name fallback on macOS when sysctl fails."""
        mock_os.cpu_count.return_value = 4
        mock_psutil.cpu_count.side_effect = [4, 4]
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = "Intel Fallback"
        mock_run.side_effect = subprocess.CalledProcessError(1, "sysctl")
        mock_psutil.cpu_freq.return_value = MagicMock(max=2500.0, current=2400.0)
        mock_detect_avx2.return_value = True

        cpu_info = detect_cpu_capabilities()

        self.assertEqual(cpu_info.model_name, "Intel Fallback")
        mock_run.assert_called_once_with(
             ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        )
        self.assertEqual(cpu_info.frequency_mhz, 2500.0)

    def test_detect_cpu_linux(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test CPU detection on Linux using /proc/cpuinfo."""
        mock_os.cpu_count.return_value = 16
        mock_psutil.cpu_count.side_effect = [8, 16]
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = "Fallback Proc"
        # <<< FIX: Mock the file handle __iter__ >>>
        mock_file_handle = mock_open_func.return_value
        mock_file_handle.__enter__.return_value.__iter__.return_value = iter([
            "processor\t: 0",
            "model name\t: AMD Ryzen 7 5800X 8-Core Processor",
            "cpu MHz\t\t: 3800.000",
        ])
        mock_psutil.cpu_freq.return_value = MagicMock(max=4700.0, current=3800.0)
        mock_detect_avx2.return_value = True

        cpu_info = detect_cpu_capabilities()

        self.assertEqual(cpu_info.logical_cores, 16)
        self.assertEqual(cpu_info.physical_cores, 8)
        # <<< FIX: Assert the correct name >>>
        self.assertEqual(cpu_info.model_name, "AMD Ryzen 7 5800X 8-Core Processor")
        self.assertEqual(cpu_info.frequency_mhz, 4700.0)
        self.assertTrue(cpu_info.supports_avx2)
        mock_open_func.assert_called_once_with('/proc/cpuinfo', 'r', encoding='utf-8')
        mock_run.assert_not_called()

    def test_detect_cpu_linux_proc_fail(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test CPU model name fallback on Linux when /proc/cpuinfo fails."""
        mock_os.cpu_count.return_value = 2
        mock_psutil.cpu_count.side_effect = [1, 2]
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "riscv64"
        mock_platform.processor.return_value = "Generic RISC-V CPU"
        mock_open_func.side_effect = OSError("Read error")
        mock_psutil.cpu_freq.return_value = None
        mock_detect_avx2.return_value = False

        cpu_info = detect_cpu_capabilities()

        self.assertEqual(cpu_info.model_name, "Generic RISC-V CPU")
        mock_open_func.assert_called_once_with('/proc/cpuinfo', 'r', encoding='utf-8')

    # <<< FIX: Apply patches using 'with' context manager >>>
    def test_detect_cpu_windows_wmic_fail(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test CPU model name fallback on Windows when wmic fails."""
        mock_os.cpu_count.return_value = 8
        mock_psutil.cpu_count.side_effect = [4, 8]
        mock_platform.system.return_value = "Windows"
        mock_platform.machine.return_value = "AMD64"
        mock_platform.processor.return_value = "Windows CPU Fallback"
        mock_run.side_effect = subprocess.CalledProcessError(1, "wmic")
        mock_psutil.cpu_freq.return_value = MagicMock(max=None, current=3200.0)
        mock_detect_avx2.return_value = True

        # Apply patches locally for constants if needed
        with patch('subprocess.DETACHED_PROCESS', 0x00000008, create=True), \
             patch('subprocess.CREATE_NO_WINDOW', 0x08000000, create=True):
            cpu_info = detect_cpu_capabilities()

        self.assertEqual(cpu_info.model_name, "Windows CPU Fallback")
        mock_run.assert_called_once_with(
            "wmic cpu get name", shell=True, capture_output=True, text=True, check=False,
            creationflags=getattr(subprocess, 'DETACHED_PROCESS', 0x00000008) | getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
        )
        self.assertEqual(cpu_info.frequency_mhz, 3200.0)

    def test_detect_cpu_physical_cores_fallback(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test physical core estimation when psutil fails."""
        mock_os.cpu_count.return_value = 4
        mock_psutil.cpu_count.side_effect = [None, 4]
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = "CPU Name"
        mock_psutil.cpu_freq.return_value = None
        mock_detect_avx2.return_value = False

        cpu_info = detect_cpu_capabilities()

        self.assertEqual(cpu_info.logical_cores, 4)
        self.assertEqual(cpu_info.physical_cores, 2)

    def test_detect_cpu_physical_cores_not_implemented(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test physical core estimation when psutil raises NotImplementedError."""
        mock_os.cpu_count.return_value = 6
        mock_psutil.cpu_count.side_effect = [NotImplementedError, 6]
        mock_platform.system.return_value = "FreeBSD"
        mock_platform.machine.return_value = "amd64"
        mock_platform.processor.return_value = "FreeBSD CPU"
        mock_psutil.cpu_freq.return_value = None
        mock_detect_avx2.return_value = False

        with self.assertLogs(logger='llamasearch.hardware', level='WARNING') as cm:
             cpu_info = detect_cpu_capabilities()
             self.assertTrue(any("psutil could not determine physical core count" in log for log in cm.output))

        self.assertEqual(cpu_info.logical_cores, 6)
        self.assertEqual(cpu_info.physical_cores, 3)

    def test_detect_cpu_freq_fails(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test when psutil.cpu_freq fails."""
        mock_os.cpu_count.return_value = 4
        mock_psutil.cpu_count.side_effect = [2, 4]
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = "CPU Name"
        mock_psutil.cpu_freq.side_effect = Exception("psutil freq error")
        mock_detect_avx2.return_value = True

        cpu_info = detect_cpu_capabilities()

        self.assertIsNone(cpu_info.frequency_mhz)


@patch('llamasearch.hardware.psutil')
class TestDetectMemory(unittest.TestCase):

    def test_detect_memory_success(self, mock_psutil):
        """Test successful memory detection."""
        mock_mem = MagicMock()
        mock_mem.total = 16 * (1024**3)
        mock_mem.available = 8.5 * (1024**3)
        mock_mem.used = 7.5 * (1024**3)
        mock_mem.percent = 46.875
        mock_psutil.virtual_memory.return_value = mock_mem

        mem_info = detect_memory_info()

        self.assertIsInstance(mem_info, MemoryInfo)
        self.assertEqual(mem_info.total_gb, 16.00)
        self.assertEqual(mem_info.available_gb, 8.50)
        self.assertEqual(mem_info.used_gb, 7.50)
        self.assertEqual(mem_info.percent_used, 46.875)
        mock_psutil.virtual_memory.assert_called_once()

    def test_detect_memory_fails(self, mock_psutil):
        """Test memory detection failure."""
        mock_psutil.virtual_memory.side_effect = Exception("psutil memory error")

        with self.assertLogs(logger='llamasearch.hardware', level='ERROR') as cm:
             mem_info = detect_memory_info()
             self.assertTrue(any("Failed to get memory info" in log for log in cm.output))

        self.assertIsInstance(mem_info, MemoryInfo)
        self.assertEqual(mem_info.total_gb, 0.0)
        self.assertEqual(mem_info.available_gb, 0.0)
        self.assertEqual(mem_info.used_gb, 0.0)
        self.assertEqual(mem_info.percent_used, 0.0)
        mock_psutil.virtual_memory.assert_called_once()


# --- Test Main Public Function ---

@patch('llamasearch.hardware.detect_cpu_capabilities')
@patch('llamasearch.hardware.detect_memory_info')
class TestDetectHardware(unittest.TestCase):

    def test_detect_hardware_info_assembly(self, mock_detect_memory, mock_detect_cpu):
        """Test that detect_hardware_info calls sub-detectors and assembles the result."""
        mock_cpu = CPUInfo(logical_cores=4, physical_cores=2, architecture="test_arch", model_name="Mock CPU", frequency_mhz=None, supports_avx2=True)
        mock_mem = MemoryInfo(total_gb=8.0, available_gb=4.0, used_gb=4.0, percent_used=50.0)
        mock_detect_cpu.return_value = mock_cpu
        mock_detect_memory.return_value = mock_mem

        hw_info = detect_hardware_info()

        self.assertIsInstance(hw_info, HardwareInfo)
        self.assertIs(hw_info.cpu, mock_cpu)
        self.assertIs(hw_info.memory, mock_mem)
        mock_detect_cpu.assert_called_once()
        mock_detect_memory.assert_called_once()

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)