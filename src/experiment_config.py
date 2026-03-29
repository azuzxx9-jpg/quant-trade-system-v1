from __future__ import annotations
import copy
from typing import Any, Dict

PROFILE_FLAGS: Dict[str, Dict[str, bool]] = {
    "trend_only": {
        "enable_trend_long": True,
        "enable_pullback_long": False,
    },
    "full_system": {
        "enable_trend_long": True,
        "enable_pullback_long": True,
    },
    "next_research": {
        "enable_trend_long": True,
        "enable_pullback_long": True,
    },
}

def get_profile_config(profile: str) -> Dict[str, bool]:
    if profile not in PROFILE_FLAGS:
        raise ValueError(f"Unknown profile: {profile}")
    return copy.deepcopy(PROFILE_FLAGS[profile])

def build_config(base_config: Dict[str, Any], profile: str) -> Dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["profile_name"] = profile
    config["signal_profile"] = profile
    config["signal_config"] = get_profile_config(profile)
    return config
