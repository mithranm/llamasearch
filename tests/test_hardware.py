# tests/test_hardware.py

import unittest
import platform
import subprocess
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
        cpu1 = CPUInfo(logical_cores=1, physical_cores=1, architecture="test", model_name=None, supports_avx2=False)
        self.assertEqual(cpu1.model_name, "Unknown")

        cpu2 = CPUInfo(logical_cores=1, physical_cores=1, architecture="test", model_name=12345, supports_avx2=False)
        self.assertEqual(cpu2.model_name, "12345")

        cpu3 = CPUInfo(logical_cores=1, physical_cores=1, architecture="test", model_name=" Real CPU ", supports_avx2=False)
        # Note: The validator doesn't strip, the main function does later
        self.assertEqual(cpu3.model_name, " Real CPU ")


    def test_memory_info_creation(self):
        mem = MemoryInfo(
            total_gb=15.6, available_gb=8.1, used_gb=7.5, percent_used=48.1
        )
        self.assertEqual(mem.total_gb, 15.6)
        self.assertEqual(mem.available_gb, 8.1)
        self.assertEqual(mem.used_gb, 7.5)
        self.assertEqual(mem.percent_used, 48.1)

    def test_hardware_info_creation(self):
        cpu = CPUInfo(logical_cores=2, physical_cores=1, architecture="arm64", model_name="M1", supports_avx2=False)
        mem = MemoryInfo(total_gb=8.0, available_gb=4.0, used_gb=4.0, percent_used=50.0)
        hw = HardwareInfo(cpu=cpu, memory=mem)
        self.assertIs(hw.cpu, cpu)
        self.assertIs(hw.memory, mem)

# --- Test Helper Functions ---

class TestDetectAVX2(unittest.TestCase):

    @patch('llamasearch.hardware.platform.system', return_value="OtherOS")
    @patch('builtins.open', new_callable=mock_open) # Mock open for linux fallback
    @patch('llamasearch.hardware.subprocess.run') # Mock run for mac fallback
    def test_avx2_pycpuinfo_present_true(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection when py_cpuinfo is present and reports AVX2."""
        mock_pycpuinfo = MagicMock()
        mock_pycpuinfo.get_cpu_info.return_value = {'flags': ['sse', 'avx', 'avx2', 'fma']}
        with patch.dict('sys.modules', {'py_cpuinfo': mock_pycpuinfo}):
             # Ensure platform isn't linux/darwin to prevent fallbacks running unnecessarily
            self.assertTrue(_detect_cpu_avx2())
        mock_pycpuinfo.get_cpu_info.assert_called_once()
        mock_open_func.assert_not_called()
        mock_run.assert_not_called()


    @patch('llamasearch.hardware.platform.system', return_value="OtherOS")
    @patch('builtins.open', new_callable=mock_open) # Mock open for linux fallback
    @patch('llamasearch.hardware.subprocess.run') # Mock run for mac fallback
    def test_avx2_pycpuinfo_present_false(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection when py_cpuinfo is present but doesn't report AVX2."""
        mock_pycpuinfo = MagicMock()
        mock_pycpuinfo.get_cpu_info.return_value = {'flags': ['sse', 'avx', 'fma']}
        with patch.dict('sys.modules', {'py_cpuinfo': mock_pycpuinfo}):
            self.assertFalse(_detect_cpu_avx2())
        mock_pycpuinfo.get_cpu_info.assert_called_once()
        mock_open_func.assert_not_called()
        mock_run.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Linux")
    @patch('builtins.open', new_callable=mock_open, read_data="flags\t\t: fpu vme de pse tsc ... avx fma ... avx2 bmi1 ...\nmore stuff")
    @patch('llamasearch.hardware.subprocess.run') # Mock run for mac fallback
    def test_avx2_pycpuinfo_missing_linux_true(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection fallback to /proc/cpuinfo on Linux (AVX2 present)."""
        with patch.dict('sys.modules', {'py_cpuinfo': None}): # Simulate import failure
            self.assertTrue(_detect_cpu_avx2())
        mock_open_func.assert_called_once_with('/proc/cpuinfo', 'r', encoding='utf-8')
        mock_run.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Linux")
    @patch('builtins.open', new_callable=mock_open, read_data="flags\t\t: fpu vme de pse tsc ... avx fma ... bmi1 ...\nmore stuff")
    @patch('llamasearch.hardware.subprocess.run') # Mock run for mac fallback
    def test_avx2_pycpuinfo_missing_linux_false(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection fallback to /proc/cpuinfo on Linux (AVX2 missing)."""
        with patch.dict('sys.modules', {'py_cpuinfo': None}):
            self.assertFalse(_detect_cpu_avx2())
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
        with patch.dict('sys.modules', {'py_cpuinfo': None}):
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
        with patch.dict('sys.modules', {'py_cpuinfo': None}):
            self.assertFalse(_detect_cpu_avx2())
        mock_run.assert_called_once_with(
            ["sysctl", "machdep.cpu.features"], capture_output=True, text=True, check=False
        )
        mock_open_func.assert_not_called()

    @patch('llamasearch.hardware.platform.system', return_value="Windows")
    @patch('builtins.open', new_callable=mock_open)
    @patch('llamasearch.hardware.subprocess.run')
    def test_avx2_pycpuinfo_missing_windows_false(self, mock_run, mock_open_func, mock_system):
        """Test AVX2 detection fallback on Windows (should be False)."""
        with patch.dict('sys.modules', {'py_cpuinfo': None}):
            self.assertFalse(_detect_cpu_avx2())
        # No platform-specific fallbacks called
        mock_run.assert_not_called()
        mock_open_func.assert_not_called()

# --- Test Detection Functions ---

@patch('llamasearch.hardware.psutil')
@patch('llamasearch.hardware.platform')
@patch('llamasearch.hardware.os')
@patch('llamasearch.hardware.subprocess.run')
@patch('llamasearch.hardware._detect_cpu_avx2') # Mock the helper
@patch('builtins.open', new_callable=mock_open) # Mock open for linux cpu model
class TestDetectCPU(unittest.TestCase):

    
    def test_detect_cpu_macos(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test CPU detection on a simulated macOS system."""
         # Setup Mocks
        mock_os.cpu_count.return_value = 8
        mock_psutil.cpu_count.side_effect = [8, 8] # Simulate M1-like structure (phys=logical)
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"
        mock_platform.processor.return_value = "Fallback Proc Name" # Should not be used if sysctl works
        mock_run.return_value = MagicMock(stdout="Apple M1 Max ", returncode=0) # Note trailing space for strip test
        mock_psutil.cpu_freq.return_value = None # Simulate freq not available
        mock_detect_avx2.return_value = False

        # Execute
        cpu_info = detect_cpu_capabilities()

        # Assert
        self.assertEqual(cpu_info.logical_cores, 8)
        self.assertEqual(cpu_info.physical_cores, 8)
        self.assertEqual(cpu_info.architecture, "arm64")
        self.assertEqual(cpu_info.model_name, "Apple M1 Max") # Stripped name from mock_run
        self.assertIsNone(cpu_info.frequency_mhz)
        self.assertFalse(cpu_info.supports_avx2)
        mock_run.assert_called_once_with(
             ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        )
        mock_open_func.assert_not_called()
        mock_detect_avx2.assert_called_once()


    
    def test_detect_cpu_physical_cores_fallback(self, mock_open_func, mock_detect_avx2, mock_run, mock_os, mock_platform, mock_psutil):
        """Test physical core estimation when psutil fails."""
        mock_os.cpu_count.return_value = 4
        # Simulate psutil.cpu_count(logical=False) returning None or raising error
        mock_psutil.cpu_count.side_effect = [None, 4] # logical=False fails, logical=True works
        mock_platform.system.return_value = "Linux" # Doesn't matter for this part
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = "CPU Name"
        mock_psutil.cpu_freq.return_value = None
        mock_detect_avx2.return_value = False

        cpu_info = detect_cpu_capabilities()

        self.assertEqual(cpu_info.logical_cores, 4)
        self.assertEqual(cpu_info.physical_cores, 2) # Estimated as logical / 2


@patch('llamasearch.hardware.psutil')
class TestDetectMemory(unittest.TestCase):

    def test_detect_memory_success(self, mock_psutil):
        """Test successful memory detection."""
        mock_mem = MagicMock()
        mock_mem.total = 16 * (1024**3) # 16 GiB
        mock_mem.available = 8.5 * (1024**3) # 8.5 GiB
        mock_mem.used = 7.5 * (1024**3) # 7.5 GiB
        mock_mem.percent = 46.875
        mock_psutil.virtual_memory.return_value = mock_mem

        mem_info = detect_memory_info()

        self.assertIsInstance(mem_info, MemoryInfo)
        self.assertEqual(mem_info.total_gb, 16.00) # Check rounding
        self.assertEqual(mem_info.available_gb, 8.50)
        self.assertEqual(mem_info.used_gb, 7.50)
        self.assertEqual(mem_info.percent_used, 46.875)
        mock_psutil.virtual_memory.assert_called_once()

    
# --- Test Main Public Function ---

@patch('llamasearch.hardware.detect_cpu_capabilities')
@patch('llamasearch.hardware.detect_memory_info')
class TestDetectHardware(unittest.TestCase):

    def test_detect_hardware_info_assembly(self, mock_detect_memory, mock_detect_cpu):
        """Test that detect_hardware_info calls sub-detectors and assembles the result."""
        # Create mock return values for the sub-detectors
        mock_cpu = CPUInfo(logical_cores=4, physical_cores=2, architecture="test_arch", model_name="Mock CPU", supports_avx2=True)
        mock_mem = MemoryInfo(total_gb=8.0, available_gb=4.0, used_gb=4.0, percent_used=50.0)
        mock_detect_cpu.return_value = mock_cpu
        mock_detect_memory.return_value = mock_mem

        # Call the main function
        hw_info = detect_hardware_info()

        # Assertions
        self.assertIsInstance(hw_info, HardwareInfo)
        self.assertIs(hw_info.cpu, mock_cpu)
        self.assertIs(hw_info.memory, mock_mem)
        mock_detect_cpu.assert_called_once()
        mock_detect_memory.assert_called_once()

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)