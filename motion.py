# motion.py (UE 5.4)
import math
import unreal

import serialize


def camera_line(cam, obj_mesh, obj_loc, mode_cfg, image_res, save_dir, prefix, yield_time):
    """Camera flies along Y at fixed X/Z offset from the object, always facing it."""
    distance_x = float(mode_cfg["distance_x"])
    height = float(mode_cfg["height"])
    half_span = float(mode_cfg["line_half_span"])
    steps = int(mode_cfg["steps"])

    fixed_x = obj_loc.x - distance_x
    fixed_z = obj_loc.z + height
    start_y = obj_loc.y - half_span
    end_y = obj_loc.y + half_span
    step_size = (end_y - start_y) / steps

    pose_list = []
    for step in range(steps + 1):
        y = start_y + step * step_size
        cam.move_to(unreal.Vector(fixed_x, y, fixed_z))
        cam.look_at(obj_mesh.actor)

        frame_name = f"{prefix}_line_{step:03}.jpg"
        out_path = f"{save_dir}/{frame_name}"
        cam.take_screenshot(out_path, res=image_res)

        pose_list.append(serialize.camera_pose_dict(frame_name, cam.actor))
        yield yield_time

    serialize.save_poses(pose_list, save_dir, prefix, suffix="camera_poses.json")


def camera_orbit(cam, obj_mesh, obj_loc, mode_cfg, image_res, save_dir, prefix, yield_time):
    """Camera orbits 360 degrees around the object at a fixed radius/height, always facing it."""
    radius = float(mode_cfg["radius"])
    height = float(mode_cfg["height"])
    steps = int(mode_cfg["steps"])

    pose_list = []
    for step in range(steps + 1):
        angle = (step / steps) * 2 * math.pi
        x = obj_loc.x + radius * math.cos(angle)
        y = obj_loc.y + radius * math.sin(angle)
        z = obj_loc.z + height

        cam.move_to(unreal.Vector(x, y, z))
        cam.look_at(obj_mesh.actor)

        frame_name = f"{prefix}_orbit_{step:03}.jpg"
        out_path = f"{save_dir}/{frame_name}"
        cam.take_screenshot(out_path, res=image_res)

        pose_list.append(serialize.camera_pose_dict(frame_name, cam.actor))
        yield yield_time

    serialize.save_poses(pose_list, save_dir, prefix, suffix="camera_poses.json")


def object_line(cam, obj_mesh, obj_loc, mode_cfg, image_res, save_dir, prefix, yield_time):
    """Camera fixed; object moves along a line in front of it, always facing the camera."""
    distance_x = float(mode_cfg["distance_x"])
    height = float(mode_cfg["height"])
    half_span = float(mode_cfg["line_half_span"])
    steps = int(mode_cfg["steps"])

    camera_loc = unreal.Vector(obj_loc.x - distance_x, obj_loc.y, obj_loc.z + height)
    cam.move_to(camera_loc)
    cam.look_at_point(obj_loc)

    start_y = obj_loc.y - half_span
    end_y = obj_loc.y + half_span
    step_size = (end_y - start_y) / steps

    pose_list = []
    for step in range(steps + 1):
        y = start_y + step * step_size
        new_loc = unreal.Vector(obj_loc.x, y, obj_loc.z)
        obj_mesh.move_to(new_loc)
        obj_mesh.face_point(cam.actor.get_actor_location())

        frame_name = f"{prefix}_line_{step:03}.jpg"
        out_path = f"{save_dir}/{frame_name}"
        cam.take_screenshot(out_path, res=image_res)

        pose_list.append(serialize.object_pose_dict(frame_name, obj_mesh.actor))
        yield yield_time

    serialize.save_poses(pose_list, save_dir, prefix, suffix="object_poses.json")


def object_orbit(cam, obj_mesh, obj_loc, mode_cfg, image_res, save_dir, prefix, yield_time):
    """Camera fixed; object orbits around a center point, always facing the camera."""
    radius = float(mode_cfg["radius"])
    height = float(mode_cfg["height"])
    camera_distance_x = float(mode_cfg["camera_distance_x"])
    steps = int(mode_cfg["steps"])

    camera_loc = unreal.Vector(obj_loc.x - camera_distance_x, obj_loc.y, obj_loc.z + height)
    cam.move_to(camera_loc)
    cam.look_at_point(obj_loc)

    pose_list = []
    for step in range(steps + 1):
        angle = (step / steps) * 2 * math.pi
        x = obj_loc.x + radius * math.cos(angle)
        y = obj_loc.y + radius * math.sin(angle)
        z = obj_loc.z

        new_loc = unreal.Vector(x, y, z)
        obj_mesh.move_to(new_loc)
        obj_mesh.face_point(cam.actor.get_actor_location())

        frame_name = f"{prefix}_orbit_{step:03}.jpg"
        out_path = f"{save_dir}/{frame_name}"
        cam.take_screenshot(out_path, res=image_res)

        pose_list.append(serialize.object_pose_dict(frame_name, obj_mesh.actor))
        yield yield_time

    serialize.save_poses(pose_list, save_dir, prefix, suffix="object_poses.json")


MODES = {
    "camera_line": camera_line,
    "camera_orbit": camera_orbit,
    "object_line": object_line,
    "object_orbit": object_orbit,
}
