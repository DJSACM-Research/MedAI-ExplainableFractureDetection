import torch

def get_device():
    """Dynamically selects CUDA, MPS, or falls back to CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def require_mps():
    """Enforces MPS device (for Mac-only scripts)."""
    if getattr(torch.backends, 'mps', None) is None or not torch.backends.mps.is_available():
        raise RuntimeError('MPS (Apple Silicon) is required but not available.')
    return torch.device('mps')

DEVICE = get_device()
