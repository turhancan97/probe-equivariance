# unreal-motion-capture

Config-driven, single-object recording toolkit for Unreal Engine 5.4, generalizing
`record_line.py`, `record_orbit.py`, and `record_object_move.py`.

Config is split into two layers:

- **Asset catalog** (`configs/assets.yaml`) — every object, with a spawn location+yaw
  per environment (`city`, `forest`, `bridge`, `desert`, `winter_town`).
- **Run config** (e.g. `configs/run.yaml`) — picks one `environment` and a `select`
  filter (`names: ["all"]` or a list of names, optionally narrowed by `categories`)
  over the catalog, plus output dir, image size, and per-mode motion params.

For each selected object, spawns it alone at its resolved location and runs one or
more of four recording modes:

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

Edit `configs/run.yaml` (environment, object selection, per-mode radius/height/span/steps,
output directory) — or point `main.py`'s `CONFIG_PATH` at a different run config, e.g.
`configs/run_animals_forest.yaml` — then run `main.py` from Unreal's Python console/execute-script.

To add a new object: add one entry to `configs/assets.yaml` with a `locations` map
covering whichever environments you need. To add a new environment: add that key to
every object's `locations` map you want available in it.

## Layout

- `config.py` — YAML config loader (run config + asset catalog resolution)
- `mesh_actor.py` — `MeshActor` spawn/move/face_point/destroy wrapper
- `camera.py` — `RenderCineCamera` spawn/move/look_at/screenshot wrapper
- `utils.py` — `PyTick` scheduler, `destroy_by_tag`
- `serialize.py` — camera/object pose JSON helpers
- `motion.py` — the four recording-mode generators
- `main.py` — entry point
