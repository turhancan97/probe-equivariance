import os
import sys
import importlib
import unreal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import camera
import mesh_actor
import motion
import utils

importlib.invalidate_caches()
importlib.reload(config)
importlib.reload(camera)
importlib.reload(mesh_actor)
importlib.reload(motion)
importlib.reload(utils)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "example.yaml")


def render_one(obj_cfg, mode_name, run_cfg):
    print(f"Starting {mode_name} for {obj_cfg.name}")

    loc = unreal.Vector(obj_cfg.location["x"], obj_cfg.location["y"], obj_cfg.location["z"])
    scale = unreal.Vector(**obj_cfg.scale) if obj_cfg.scale else None

    obj_mesh = mesh_actor.MeshActor(
        obj_cfg.mesh_path,
        loc=loc,
        rot=unreal.Rotator(0, 0, obj_cfg.yaw),
        label=obj_cfg.name,
        scale=scale,
        material_path=obj_cfg.material_path,
        material_slot=obj_cfg.material_slot,
    )
    cam = camera.RenderCineCamera(label=f"{obj_cfg.name}_camera")

    mode_cfg = getattr(run_cfg, mode_name)
    save_dir = os.path.join(run_cfg.output_dir, obj_cfg.name, mode_name)
    os.makedirs(save_dir, exist_ok=True)

    image_res = (run_cfg.image_width, run_cfg.image_height)

    try:
        yield from motion.MODES[mode_name](
            cam, obj_mesh, loc, mode_cfg, image_res, save_dir, mode_name, run_cfg.yield_time
        )
    finally:
        obj_mesh.destroy()
        cam.destroy()

    print(f"Finished {mode_name} for {obj_cfg.name}")


if __name__ == "__main__":
    utils.destroy_by_tag(tag="SCRIPT_GENERATED")

    run_cfg = config.load_config(CONFIG_PATH)

    pt = utils.PyTick()
    for obj_cfg in run_cfg.objects:
        for mode_name in run_cfg.modes:
            pt.schedule.append(render_one(obj_cfg, mode_name, run_cfg))
