# unreal-motion-capture

Config-driven, single-object recording toolkit for Unreal Engine 5.4, generalizing
`record_line.py`, `record_orbit.py`, and `record_object_move.py`.

For each object listed in `configs/example.yaml`, spawns it alone at its configured
location and runs one or more of four recording modes:

- **camera_line** — camera flies along Y at a fixed X/Z offset from the object, always
  facing it. Saves per-frame camera pose JSON.
- **camera_orbit** — camera orbits 360° around the object at a configured radius/height,
  always facing it. Saves per-frame camera pose JSON.
- **object_line** — camera stays fixed; the object moves along a line in front of it,
  always facing the camera. Saves per-frame object pose JSON.
- **object_orbit** — camera stays fixed; the object orbits around a center point,
  always facing the camera. Saves per-frame object pose JSON.

## Requirements

Unreal's embedded Python needs PyYAML:

```
<UE Python interpreter> -m pip install pyyaml
```

## Usage

Edit `configs/example.yaml` (object list, spawn locations, per-mode radius/height/span/steps,
output directory), then run `main.py` from Unreal's Python console/execute-script.

## Layout

- `config.py` — YAML config loader
- `mesh_actor.py` — `MeshActor` spawn/move/face_point/destroy wrapper
- `camera.py` — `RenderCineCamera` spawn/move/look_at/screenshot wrapper
- `utils.py` — `PyTick` scheduler, `destroy_by_tag`
- `serialize.py` — camera/object pose JSON helpers
- `motion.py` — the four recording-mode generators
- `main.py` — entry point
