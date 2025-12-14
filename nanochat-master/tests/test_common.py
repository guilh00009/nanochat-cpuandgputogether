
import pytest
import torch
from nanochat.common import autodetect_device_type

def test_autodetect_device_type():
    # This might depend on the machine running it, but we can check it returns a string
    device_type = autodetect_device_type()
    assert isinstance(device_type, str)
    assert device_type in ["cuda", "mps", "cpu", "xla"]

def test_imports():
    # Verify we can import key components
    from nanochat.common import get_dist_info
    assert callable(get_dist_info)
