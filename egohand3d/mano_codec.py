"""Explicit MANO interchange; WiLoR left hands use a reflected RIGHT model.

Rotations remain in the canonical right MANO model. Reflection is applied to
decoded geometry before camera translation; these are not left-model poses.
"""
from __future__ import annotations

import copy
import numpy as np
from scipy.spatial.transform import Rotation

SCHEMA = "egohand3d.mano/1"


def rotations(value, count: int = 16) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (count, 3, 3) or not np.isfinite(value).all():
        raise ValueError(f"Expected {count} finite 3x3 rotation matrices")
    if not np.allclose(value @ value.swapaxes(-1, -2), np.eye(3), atol=2e-4):
        raise ValueError("Rotation matrices are not orthonormal")
    if not np.allclose(np.linalg.det(value), 1, atol=2e-4):
        raise ValueError("Rotations must have determinant +1; reflection is separate")
    return value


def encode_hand(hand: dict) -> dict:
    params = hand["pred_mano_params"]
    arrays = {}
    for name in ["global_orient", "hand_pose", "betas"]:
        payload = params[name]
        array = np.asarray(payload["values"], dtype=np.float64)
        if list(array.shape) != payload["shape"]:
            raise ValueError(f"MANO {name}: declared shape differs from values")
        arrays[name] = array
    if arrays["global_orient"].shape != (1, 3, 3) or arrays["hand_pose"].shape != (15, 3, 3):
        raise ValueError("Runtime MANO must supply rotation matrices, not inferred axis angles")
    pose = rotations(np.concatenate([arrays["global_orient"], arrays["hand_pose"]]))
    width, height = hand["img_size_wh"]
    focal = float(hand["focal_length_px"])
    packet = {
        "schema": SCHEMA, "model_hand": "right", "hand_side": hand["hand_side"],
        "mirror_model_x": hand["hand_side"] == "left", "length_unit": "m", "angle_unit": "rad",
        "camera_axes": "x_right_y_down_z_forward", "pose_space": "canonical_right_mano",
        "geometry_origin": "MANO native origin; not explicitly wrist-centered",
        "pose_rotmat": pose.tolist(), "pose_axis_angle": Rotation.from_matrix(pose).as_rotvec().tolist(),
        "betas": arrays["betas"].tolist(), "translation_camera": hand["cam_t_full"],
        "intrinsics": [[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]],
        "intrinsics_source": "WiLoR assumed focal length; not a camera calibration",
        "image_size_wh": [width, height],
    }
    validate_packet(packet)
    return packet


def validate_packet(packet: dict) -> None:
    if packet.get("schema") != SCHEMA or packet.get("model_hand") != "right":
        raise ValueError("Unsupported MANO schema/model hand")
    if packet.get("hand_side") not in {"left", "right"}:
        raise ValueError("hand_side must be left or right")
    if packet.get("mirror_model_x") is not (packet["hand_side"] == "left"):
        raise ValueError("Hand side and reflection disagree")
    if packet.get("length_unit") != "m" or packet.get("angle_unit") != "rad":
        raise ValueError("Interchange uses meters and radians")
    pose = rotations(packet["pose_rotmat"])
    aa = np.asarray(packet["pose_axis_angle"], dtype=np.float64)
    if aa.shape != (16, 3) or not np.isfinite(aa).all():
        raise ValueError("Expected 16 finite axis-angle vectors")
    if not np.allclose(Rotation.from_rotvec(aa).as_matrix(), pose, atol=2e-4):
        raise ValueError("Axis angles and rotation matrices describe different poses")
    for key, shape in [("betas", (10,)), ("translation_camera", (3,)), ("intrinsics", (3, 3)),
                       ("image_size_wh", (2,))]:
        array = np.asarray(packet[key], dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid {key}; expected finite shape {shape}")
    k = np.asarray(packet["intrinsics"])
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("Invalid pinhole intrinsics")
    if min(packet["image_size_wh"]) <= 0:
        raise ValueError("Image dimensions must be positive")


def project(points_camera, intrinsics) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_camera, dtype=np.float64)
    valid = np.isfinite(points).all(axis=-1) & (points[:, 2] > 1e-8)
    uv = np.zeros((len(points), 2), dtype=np.float64)
    homogeneous = points[valid] @ np.asarray(intrinsics, dtype=float).T
    uv[valid] = homogeneous[:, :2] / homogeneous[:, 2:]
    return uv, valid


class ManoDecoder:
    def __init__(self, layer):
        self.layer = layer

    @classmethod
    def from_directory(cls, directory, device: str = "cpu"):
        # Use the same declared 21-joint mapping as the external runtime.
        from wilor.models.mano_wrapper import MANO
        return cls(MANO(model_path=str(directory), is_rhand=True, use_pca=False,
                        flat_hand_mean=True).to(device).eval())

    def decode(self, packet: dict) -> dict:
        import torch
        validate_packet(packet)
        device = next(self.layer.buffers()).device
        pose = torch.as_tensor(packet["pose_rotmat"], dtype=torch.float32, device=device)[None]
        betas = torch.as_tensor(packet["betas"], dtype=torch.float32, device=device)[None]
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode():
                output = self.layer(global_orient=pose[:, :1], hand_pose=pose[:, 1:], betas=betas, pose2rot=False)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        joints = output.joints[0].cpu().numpy().copy()
        vertices = output.vertices[0].cpu().numpy().copy()
        if packet["mirror_model_x"]:
            joints[:, 0] *= -1
            vertices[:, 0] *= -1
        translation = np.asarray(packet["translation_camera"])
        uv, visible_depth = project(joints + translation, packet["intrinsics"])
        return {"joints_model": joints, "vertices_model": vertices,
                "joints_camera": joints + translation, "vertices_camera": vertices + translation,
                "joints_2d": uv, "positive_depth": visible_depth}

    def check(self, packet: dict, hand: dict, tolerance_m: float = 1e-5, tolerance_px: float = 0.05) -> dict:
        decoded = self.decode(packet)
        joint_error = np.linalg.norm(decoded["joints_model"] - np.asarray(hand["joints_3d_root"]), axis=1)
        vertex_error = np.linalg.norm(decoded["vertices_model"] - np.asarray(hand["vertices_3d_root"]), axis=1)
        valid = decoded["positive_depth"]
        uv_error = np.linalg.norm(decoded["joints_2d"][valid] - np.asarray(hand["joints_2d_xy"])[valid], axis=1)
        maximum_px = float(uv_error.max()) if valid.any() else None
        return {"max_joint_error_m": float(joint_error.max()), "max_vertex_error_m": float(vertex_error.max()),
                "max_reprojection_error_px": maximum_px, "num_projectable_joints": int(valid.sum()),
                "tolerance_m": tolerance_m, "tolerance_px": tolerance_px,
                "passed": bool(joint_error.max() <= tolerance_m and vertex_error.max() <= tolerance_m
                               and maximum_px is not None and maximum_px <= tolerance_px)}


def interpolate_packets(first: dict, second: dict, weight: float) -> dict:
    """Geodesic interpolation of each rotation, linear shape and translation."""
    validate_packet(first)
    validate_packet(second)
    if first["hand_side"] != second["hand_side"] or not 0 <= weight <= 1:
        raise ValueError("Interpolation needs the same hand side and weight in [0,1]")
    if first["image_size_wh"] != second["image_size_wh"] or not np.allclose(first["intrinsics"], second["intrinsics"]):
        raise ValueError("Interpolation needs matching camera metadata")
    a, b = np.asarray(first["pose_rotmat"]), np.asarray(second["pose_rotmat"])
    relative = a.swapaxes(-1, -2) @ b
    delta = Rotation.from_matrix(relative).as_rotvec() * weight
    pose = a @ Rotation.from_rotvec(delta).as_matrix()
    out = copy.deepcopy(second)
    out["pose_rotmat"] = pose.tolist()
    out["pose_axis_angle"] = Rotation.from_matrix(pose).as_rotvec().tolist()
    for key in ["betas", "translation_camera"]:
        out[key] = ((1 - weight) * np.asarray(first[key]) + weight * np.asarray(second[key])).tolist()
    return out
