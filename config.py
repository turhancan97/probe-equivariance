import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import yaml
except ImportError as e:
    raise RuntimeError(
        "PyYAML is required but not installed in Unreal's embedded Python. "
        "Install it with: <UE Python interpreter> -m pip install pyyaml"
    ) from e


@dataclass
class ObjectConfig:
    name: str
    mesh_path: str
    location: Dict[str, float]
    category: Optional[str] = None
    material_path: Optional[str] = None
    material_slot: int = 0
    scale: Optional[Dict[str, float]] = None
    yaw: float = 0.0


@dataclass
class RunConfig:
    output_dir: str
    image_width: int
    image_height: int
    yield_time: float
    modes: List[str]
    camera_line: Dict
    camera_orbit: Dict
    object_line: Dict
    object_orbit: Dict
    objects: List[ObjectConfig] = field(default_factory=list)


def load_config(path: str) -> RunConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    objects = [ObjectConfig(**o) for o in raw.get("objects", [])]

    return RunConfig(
        output_dir=raw["output_dir"],
        image_width=int(raw.get("image_width", 1280)),
        image_height=int(raw.get("image_height", 720)),
        yield_time=float(raw.get("yield_time", 1.0)),
        modes=raw.get("modes", ["camera_line", "camera_orbit", "object_line", "object_orbit"]),
        camera_line=raw.get("camera_line", {}),
        camera_orbit=raw.get("camera_orbit", {}),
        object_line=raw.get("object_line", {}),
        object_orbit=raw.get("object_orbit", {}),
        objects=objects,
    )
