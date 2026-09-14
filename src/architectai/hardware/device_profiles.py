from typing import Any, Dict, List

DEVICE_PROFILES: Dict[str, Dict[str, Any]] = {
    "generic_cpu": {
        "name": "Generic x86_64 CPU",
        "category": "cpu",
        "max_memory_mb": 8192,
        "max_params": 100_000_000,
        "compute_gflops": 150.0,
        "power_budget_w": 65.0,
    },
    "nvidia_t4": {
        "name": "NVIDIA T4 Tensor Core GPU",
        "category": "gpu",
        "max_memory_mb": 16384,
        "max_params": 500_000_000,
        "compute_gflops": 8100.0,
        "power_budget_w": 70.0,
    },
    "jetson_nano": {
        "name": "NVIDIA Jetson Nano",
        "category": "edge",
        "max_memory_mb": 4096,
        "max_params": 15_000_000,
        "compute_gflops": 472.0,
        "power_budget_w": 10.0,
    },
    "raspberry_pi_4": {
        "name": "Raspberry Pi 4 Model B",
        "category": "edge",
        "max_memory_mb": 2048,
        "max_params": 5_000_000,
        "compute_gflops": 13.5,
        "power_budget_w": 5.0,
    },
    "mobile_arm": {
        "name": "Generic Mobile ARM Cortex-A78",
        "category": "mobile",
        "max_memory_mb": 3072,
        "max_params": 10_000_000,
        "compute_gflops": 100.0,
        "power_budget_w": 3.0,
    },
}


def get_device_profile(name: str) -> Dict[str, Any]:
    """Retrieve device profile by name or key."""
    key = name.lower().replace(" ", "_")
    if key in DEVICE_PROFILES:
        return DEVICE_PROFILES[key]
    for k, profile in DEVICE_PROFILES.items():
        if profile["name"].lower() == name.lower():
            return profile
    # Return generic fallback
    return DEVICE_PROFILES["generic_cpu"]


def list_devices() -> List[str]:
    """Return list of available device profile identifiers."""
    return list(DEVICE_PROFILES.keys())

