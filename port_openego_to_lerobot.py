#!/usr/bin/env python

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from openego.core.constants import MANO_JOINT_NAMES
from openego.core.utils import get_hdf5_data, get_sorted_paths, get_video_frames, get_video_info, load_json
from openego.data.openego import get_benchmark_name, get_egodex_intrinsic, get_egodex_joints


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return x


def normalize_visibility(v, n_frames, n_joints):
    if v is None:
        return np.ones((n_frames, n_joints), dtype=np.int32)
    a = np.asarray(v)
    if a.ndim == 1:
        a = np.repeat(a[:, None], n_joints, axis=1)
    if a.ndim != 2:
        a = np.ones((n_frames, n_joints), dtype=np.int32)
    if a.shape[1] != n_joints:
        a = np.repeat(a[:, :1], n_joints, axis=1)
    return a.astype(np.int32)


def empty_annotation(video_info, task=""):
    return {"task": task, "actions": [], "video_info": video_info}


def load_demo(video_path: Path):
    benchmark = get_benchmark_name(video_path)
    video_info = get_video_info(video_path)
    if benchmark == "egodex":
        joints = get_egodex_joints(video_path)
        ann_path = (
            video_path.parents[2]
            / "annotations"
            / video_path.parent.parent.name
            / video_path.parent.name
            / f"{video_path.stem}.json"
        )
        task = video_path.parent.name.replace("_", " ")
        annotation = load_json(ann_path) if ann_path.exists() else empty_annotation(video_info, task=task)
        metadata = {"intrinsics": get_egodex_intrinsic(video_path), **video_info}
    else:
        demo_dir = video_path.parent
        joints = get_hdf5_data(demo_dir / "joints.hdf5")
        ann_path = demo_dir / "annotation.json"
        annotation = load_json(ann_path) if ann_path.exists() else empty_annotation(video_info)
        metadata_path = demo_dir / "metadata.hdf5"
        metadata = get_hdf5_data(metadata_path) if metadata_path.exists() else {}
    if "actions" not in annotation:
        annotation["actions"] = []
    if "task" not in annotation:
        annotation["task"] = ""
    if "video_info" not in annotation:
        annotation["video_info"] = video_info
    return joints, annotation, metadata, video_info


def get_demo_fps(annotation, metadata, video_info):
    return float(annotation.get("video_info", {}).get("fps") or metadata.get("fps") or video_info.get("fps") or 0.0)


def build_features(image_shape):
    hand_axes = [f"{j}_{a}" for j in MANO_JOINT_NAMES for a in ("x", "y", "z")]
    state_axes = [f"left_{a}" for a in hand_axes] + [f"right_{a}" for a in hand_axes]
    vis_axes = [f"left_{j}" for j in MANO_JOINT_NAMES] + [f"right_{j}" for j in MANO_JOINT_NAMES]
    n_joints = len(MANO_JOINT_NAMES)
    return {
        "is_first": {"dtype": "bool", "shape": (1,), "names": None},
        "is_last": {"dtype": "bool", "shape": (1,), "names": None},
        "is_terminal": {"dtype": "bool", "shape": (1,), "names": None},
        "subtask": {"dtype": "string", "shape": (1,), "names": None},
        "subtask_objects": {"dtype": "string", "shape": (1,), "names": None},
        "subtask_actors": {"dtype": "string", "shape": (1,), "names": None},
        "observation.images.egocentric": {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.state.intrinsics": {"dtype": "float32", "shape": (3, 3), "names": None},
        "observation.state": {
            "dtype": "float32",
            "shape": (n_joints * 6,),
            "names": {"axes": state_axes},
        },
        "observation.state.visibility": {
            "dtype": "int32",
            "shape": (n_joints * 2,),
            "names": {"axes": vis_axes},
        },
        "action": {
            "dtype": "float32",
            "shape": (n_joints * 6,),
            "names": {"axes": state_axes},
        },
        "action.visibility": {
            "dtype": "int32",
            "shape": (n_joints * 2,),
            "names": {"axes": vis_axes},
        },
    }


def port_openego(
    raw_dir: Path,
    repo_id: str,
    push_to_hub: bool = False,
    private: bool = False,
    root: Path | None = None,
    overwrite: bool = False,
):
    video_paths = [p for p in get_sorted_paths(raw_dir, "*.mp4") if not p.name.startswith(".")]
    if not video_paths:
        raise ValueError(f"No mp4 files found under {raw_dir}")
    _, first_annotation, first_metadata, first_info = load_demo(video_paths[0])
    dataset_fps = int(round(get_demo_fps(first_annotation, first_metadata, first_info))) or 30
    features = build_features((first_info["height"], first_info["width"], 3))
    output_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if output_root.exists():
        if overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(
                f"Output path already exists: {output_root}. "
                "Use --overwrite to remove it first, or set --root to a new path."
            )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="openego",
        fps=dataset_fps,
        features=features,
        root=output_root,
    )
    for episode_index, video_path in enumerate(
        tqdm(video_paths, desc="Episodes", unit="episode", dynamic_ncols=True)
    ):
        joints, annotation, metadata, video_info = load_demo(video_path)
        left = np.asarray(joints.get("left_hand"), dtype=np.float32)
        right = np.asarray(joints.get("right_hand"), dtype=np.float32)
        if left.ndim != 3 or right.ndim != 3:
            logging.warning(f"Skipping {video_path}: invalid hand joint tensors")
            continue
        n_joints = len(MANO_JOINT_NAMES)
        left_vis = normalize_visibility(joints.get("left_hand_visibility"), left.shape[0], n_joints)
        right_vis = normalize_visibility(joints.get("right_hand_visibility"), right.shape[0], n_joints)
        intrinsics = np.asarray(joints.get("intrinsics", metadata.get("intrinsics", np.eye(3))), dtype=np.float32).reshape(3, 3)
        task = str(annotation.get("task", ""))
        actions = sorted(annotation.get("actions", []), key=lambda x: float(x.get("start_timestamp", 0.0)))
        rel_demo = str(video_path.relative_to(raw_dir))
        fps_value = get_demo_fps(annotation, metadata, video_info) or float(dataset_fps)
        if int(round(fps_value)) != dataset_fps:
            logging.warning(f"Per-demo fps={fps_value} differs from dataset fps={dataset_fps} for {rel_demo}")
        n_frames = min(int(video_info["num_frames"]), left.shape[0], right.shape[0], left_vis.shape[0], right_vis.shape[0])
        rgb = get_video_frames(video_path)
        if rgb.size == 0:
            logging.warning(f"Skipping {video_path}: cannot read video frames")
            continue
        n_frames = min(n_frames, rgb.shape[0])
        logging.info(f"{episode_index + 1}/{len(video_paths)} {rel_demo} frames={n_frames}")
        action_i = 0
        for frame_index in tqdm(
            range(n_frames),
            desc=rel_demo,
            unit="frame",
            leave=False,
            dynamic_ncols=True,
        ):
            t = frame_index / fps_value if fps_value > 0 else float(frame_index)
            while action_i < len(actions) and t >= float(actions[action_i].get("end_timestamp", -1.0)):
                action_i += 1
            active = {}
            if action_i < len(actions):
                start_t = float(actions[action_i].get("start_timestamp", np.inf))
                end_t = float(actions[action_i].get("end_timestamp", -np.inf))
                if start_t <= t < end_t:
                    active = actions[action_i]
            left_state = left[frame_index].reshape(-1).astype(np.float32)
            right_state = right[frame_index].reshape(-1).astype(np.float32)
            state = np.concatenate([left_state, right_state]).astype(np.float32)
            visibility = np.concatenate([left_vis[frame_index], right_vis[frame_index]]).astype(np.int32)
            next_index = min(frame_index + 1, n_frames - 1)
            next_left_state = left[next_index].reshape(-1).astype(np.float32)
            next_right_state = right[next_index].reshape(-1).astype(np.float32)
            next_state = np.concatenate([next_left_state, next_right_state]).astype(np.float32)
            next_visibility = np.concatenate([left_vis[next_index], right_vis[next_index]]).astype(np.int32)
            frame = {
                "is_first": np.array([frame_index == 0]),
                "is_last": np.array([frame_index == n_frames - 1]),
                "is_terminal": np.array([frame_index == n_frames - 1]),
                "task": task,
                "subtask": str(active.get("label", "")),
                "subtask_objects": json.dumps(to_jsonable(active.get("objects", [])), ensure_ascii=False),
                "subtask_actors": json.dumps(to_jsonable(active.get("actors", [])), ensure_ascii=False),
                "observation.images.egocentric": rgb[frame_index],
                "observation.state.intrinsics": intrinsics,
                "observation.state": state,
                "observation.state.visibility": visibility,
                "action": next_state,
                "action.visibility": next_visibility,
            }
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    if push_to_hub:
        dataset.push_to_hub(tags=["openego"], private=private)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    port_openego(**vars(args))


if __name__ == "__main__":
    main()
