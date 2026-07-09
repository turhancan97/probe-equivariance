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
    environment: str
    objects: List[ObjectConfig] = field(default_factory=list)


def _load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_objects(catalog_objects: List[Dict], environment: str, select: Dict) -> List[ObjectConfig]:
    names = select.get("names", ["all"])
    categories = select.get("categories", [])
    want_all_names = names == "all" or names == ["all"]

    resolved: List[ObjectConfig] = []
    for entry in catalog_objects:
        if not want_all_names and entry["name"] not in names:
            continue
        if categories and entry.get("category") not in categories:
            continue

        locations = entry.get("locations", {})
        if environment not in locations:
            raise ValueError(
                f"Object '{entry['name']}' has no location defined for environment '{environment}'"
            )
        loc = locations[environment]

        resolved.append(ObjectConfig(
            name=entry["name"],
            category=entry.get("category"),
            mesh_path=entry["mesh_path"],
            material_path=entry.get("material_path"),
            material_slot=int(entry.get("material_slot", 0)),
            scale=entry.get("scale"),
            location={"x": float(loc["x"]), "y": float(loc["y"]), "z": float(loc["z"])},
            yaw=float(loc.get("yaw", entry.get("yaw", 0.0))),
        ))

    if not resolved:
        raise ValueError(
            f"No objects matched select={select!r} for environment '{environment}' — check names/categories."
        )
    return resolved


def load_config(run_path: str) -> RunConfig:
    raw = _load_yaml(run_path)

    catalog_path = raw["assets_catalog"]
    if not os.path.isabs(catalog_path):
        catalog_path = os.path.join(os.path.dirname(os.path.abspath(run_path)), catalog_path)
    catalog = _load_yaml(catalog_path)

    environment = raw["environment"]
    select = raw.get("select", {"names": ["all"]})
    objects = _resolve_objects(catalog.get("objects", []), environment, select)

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
        environment=environment,
        objects=objects,
    )
