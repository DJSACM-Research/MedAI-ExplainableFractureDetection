"""
Device utilities for MedAI.

Provides functions for automatic device selection (CUDA, MPS, CPU).
"""

import torch

__all__ = ["get_device", "require_mps", "DEVICE"]


def get_device() -> torch.device:
    """
    Dynamically selects the best available compute device.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: The selected device.
        
    Examples:
        >>> device = get_device()
        >>> model = model.to(device)
        >>> tensor = tensor.to(device)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def require_mps() -> torch.device:
    """
    Enforces MPS device availability (for Apple Silicon-specific scripts).
    
    Returns:
        torch.device: MPS device.
        
    Raises:
        RuntimeError: If MPS is not available on the system.
        
    Examples:
        >>> device = require_mps()  # Raises if not on Apple Silicon
    """
    if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS (Apple Silicon) is required but not available. "
            "This script requires an Apple Silicon Mac with macOS 12.3+."
        )
    return torch.device("mps")


def get_device_info() -> dict:
    """
    Returns detailed information about the current device configuration.
    
    Returns:
        dict: Dictionary containing device information.
    """
    info = {
        "device": str(get_device()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": getattr(torch.backends, "mps", None) is not None 
                         and torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        })
    
    return info


# Initialize default device at module load
DEVICE = get_device()
