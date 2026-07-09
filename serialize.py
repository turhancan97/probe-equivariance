# serialize.py (UE 5.4)
import json
import os
import unreal
from typing import Dict, List


def _vec(v: unreal.Vector) -> Dict[str, float]:
    return {"x": float(v.x), "y": float(v.y), "z": float(v.z)}


def _rot(r: unreal.Rotator) -> Dict[str, float]:
    return {"pitch": float(r.pitch), "yaw": float(r.yaw), "roll": float(r.roll)}


def camera_pose_dict(frame_name: str, cam_actor: unreal.Actor) -> Dict:
    return {
        "frame": frame_name,
        "location": _vec(cam_actor.get_actor_location()),
        "rotation": _rot(cam_actor.get_actor_rotation()),
    }


def object_pose_dict(frame_name: str, obj_actor: unreal.Actor) -> Dict:
    return {
        "frame": frame_name,
        "location": _vec(obj_actor.get_actor_location()),
        "rotation": _rot(obj_actor.get_actor_rotation()),
    }


def save_poses(pose_list: List[Dict], save_dir: str, prefix: str, suffix: str = "poses.json") -> str:
    path = os.path.join(save_dir, f"{prefix}_{suffix}")
    with open(path, "w") as f:
        json.dump(pose_list, f, indent=2)
    return path
