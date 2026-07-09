# mesh_actor.py (UE 5.4)
import math
import unreal
from typing import Optional

editor = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


class MeshActor:
    def __init__(self,
                 mesh_path: str,
                 loc: unreal.Vector = None,
                 rot: unreal.Rotator = None,
                 label: Optional[str] = None,
                 scale: Optional[unreal.Vector] = None,
                 material_path: Optional[str] = None,
                 material_slot: int = 0):
        mesh = unreal.load_object(None, mesh_path)
        if not mesh:
            raise RuntimeError(f"Failed to load StaticMesh: {mesh_path}")

        if loc is None:
            loc = unreal.Vector(0, 0, 0)
        if rot is None:
            rot = unreal.Rotator(0, 0, 0)

        self.actor: unreal.StaticMeshActor = editor.spawn_actor_from_class(unreal.StaticMeshActor, loc, rot)
        self.actor.tags = [unreal.Name("SCRIPT_GENERATED")]
        self.actor.static_mesh_component.set_static_mesh(mesh)
        if scale is not None:
            self.actor.set_actor_scale3d(scale)
        if label:
            self.actor.set_actor_label(label, True)
        if material_path:
            self.set_material(material_path, material_slot)

    def _assert_alive(self):
        if self.actor is None:
            raise RuntimeError("MeshActor was destroyed")
        if not unreal.SystemLibrary.is_valid(self.actor):
            raise RuntimeError("MeshActor is not valid (pending kill)")

    def move_to(self, loc: unreal.Vector, rot: unreal.Rotator = None) -> "MeshActor":
        self._assert_alive()
        if rot is None:
            rot = self.actor.get_actor_rotation()
        self.actor.set_actor_location_and_rotation(loc, rot, sweep=False, teleport=True)
        return self

    def face_point(self, target_loc: unreal.Vector) -> "MeshActor":
        """Rotate (yaw only, pitch/roll = 0) so the actor faces target_loc."""
        self._assert_alive()
        origin = self.actor.get_actor_location()
        direction = target_loc - origin
        yaw = math.degrees(math.atan2(direction.y, direction.x))
        self.actor.set_actor_rotation(unreal.Rotator(0.0, 0.0, yaw), False)
        return self

    def set_material(self, mat_path: str, slot: int = 0) -> "MeshActor":
        self._assert_alive()
        mtl = unreal.load_object(None, mat_path)
        if not mtl:
            raise RuntimeError(f"Failed to load Material: {mat_path}")
        self.actor.static_mesh_component.set_material(slot, mtl)
        return self

    def destroy(self) -> None:
        if self.actor is not None:
            unreal.EditorLevelLibrary.destroy_actor(self.actor)
        self.actor = None
