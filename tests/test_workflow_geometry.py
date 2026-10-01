import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egohand3d.mano_codec import encode_hand, interpolate_packets, project, validate_packet
from egohand3d.metrics import evaluate_frames, similarity_align
from egohand3d.sequence import HandTracker


def hand(side="right", x=0.):
    pose = np.repeat(np.eye(3)[None], 16, axis=0)
    return {"hand_index": 0, "hand_side": side, "is_right": side == "right",
            "bbox_xyxy": [x, 0., x + 100., 100.], "bbox_score": 0.9,
            "img_size_wh": [640, 480], "focal_length_px": 500., "cam_t_full": [0., 0., 1.],
            "pred_mano_params": {"global_orient": {"shape": [1, 3, 3], "values": pose[:1].tolist()},
             "hand_pose": {"shape": [15, 3, 3], "values": pose[1:].tolist()},
             "betas": {"shape": [10], "values": [0.] * 10}}}


def test_roundtrip_rotation_near_pi_and_left_convention():
    value = hand("left")
    matrix = Rotation.from_rotvec([np.pi - 1e-7, 0, 0]).as_matrix()
    value["pred_mano_params"]["global_orient"]["values"] = matrix[None].tolist()
    packet = encode_hand(value)
    assert packet["mirror_model_x"] is True
    assert packet["model_hand"] == "right"
    np.testing.assert_allclose(Rotation.from_rotvec(packet["pose_axis_angle"]).as_matrix(), packet["pose_rotmat"], atol=1e-7)
    invalid = copy.deepcopy(packet)
    invalid["pose_rotmat"][0][0][0] *= -1
    with pytest.raises(ValueError):
        validate_packet(invalid)


def test_geodesic_interpolation_crosses_angle_wrap_without_collapsing():
    a, b = encode_hand(hand()), encode_hand(hand())
    for packet, degrees in [(a, 179), (b, -179)]:
        pose = np.repeat(Rotation.from_euler("z", degrees, degrees=True).as_matrix()[None], 16, axis=0)
        packet["pose_rotmat"] = pose.tolist()
        packet["pose_axis_angle"] = Rotation.from_matrix(pose).as_rotvec().tolist()
    middle = interpolate_packets(a, b, 0.5)
    np.testing.assert_allclose(np.asarray(middle["pose_rotmat"])[0], np.diag([-1., -1., 1.]), atol=1e-7)
    validate_packet(middle)


def test_projection_uses_translation_and_positive_depth_mask():
    uv, valid = project([[0.2, 0.1, 2.], [0, 0, -1.]], [[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    np.testing.assert_allclose(uv[0], [370., 265.])
    assert valid.tolist() == [True, False]


def test_procrustes_recovers_rotation_scale_translation():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(21, 3))
    r = Rotation.from_rotvec([0.4, -0.2, 0.7]).as_matrix()
    y = 1.8 * x @ r + [1, 2, 3]
    np.testing.assert_allclose(similarity_align(x, y), y, atol=1e-10)
    with pytest.raises(ValueError):
        similarity_align(np.zeros((21, 3)), y)


def test_missing_hands_lower_all_gt_pck_and_interpolation_is_excluded():
    joints = np.zeros((21, 2)).tolist()
    label = {"hand_side": "right", "bbox_xyxy": [0, 0, 100, 100], "joints_2d_xy": joints}
    targets = [{"sample_id": str(i), "hands": [label]} for i in range(3)]
    predictions = {"0": {"hands": [dict(label)]}, "1": {"hands": [{**label, "observation": "interpolated"}]}}
    result = evaluate_frames(targets, predictions)
    assert result["matched_hands"] == 1
    assert result["hand_recall"] == pytest.approx(1 / 3)
    assert result["pck_px"]["5.0"]["matched_only"] == 1
    assert result["pck_px"]["5.0"]["all_gt_joints"] == pytest.approx(1 / 3)
    assert result["missing_prediction_frames"] == 1


def test_association_uses_bbox_not_best_keypoint_error():
    label = {"hand_side": "right", "bbox_xyxy": [0, 0, 100, 100], "joints_2d_xy": np.zeros((21, 2)).tolist()}
    close_box_bad_joints = {**label, "joints_2d_xy": np.ones((21, 2)).tolist()}
    distant_box_perfect_joints = {**label, "bbox_xyxy": [1000, 1000, 1100, 1100]}
    result = evaluate_frames([{"sample_id": "a", "hands": [label]}],
                             {"a": {"hands": [close_box_bad_joints, distant_box_perfect_joints]}})
    assert result["metrics"]["2d_px"]["mean"] == pytest.approx(np.sqrt(2))
    assert result["unmatched_predictions"] == 1


def test_tracker_keeps_ids_when_detection_order_changes_and_bounds_gaps():
    tracker = HandTracker(max_gap=1)
    left, right = hand("left"), hand("right", 200)
    for item in [left, right]:
        item["mano"] = encode_hand(item)
    first, _ = tracker.update([left, right], 0, 0.)
    second, _ = tracker.update([right, left], 1, 0.04)
    assert second[0]["track_id"] == first[1]["track_id"]
    tracker.update([], 2, 0.08)
    third, gaps = tracker.update([right], 3, 0.12)
    assert third[0]["track_id"] == first[1]["track_id"]
    assert len(gaps) == 1
    tracker.update([], 4, 0.16)
    tracker.update([], 5, 0.20)
    last, _ = tracker.update([right], 6, 0.24)
    assert last[0]["track_id"] != first[1]["track_id"]
    with pytest.raises(ValueError):
        tracker.update([], 7, 0.24)
