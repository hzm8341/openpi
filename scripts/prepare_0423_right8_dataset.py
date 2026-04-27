"""Prepare the 0423 dataset for right-arm pi0.5 fine-tuning.

This creates a derived LeRobot dataset without modifying the source dataset:
- keep right arm joints plus right gripper: original dims 7..14 -> 8 dims
- keep chest camera and right wrist camera only
- repair terminal next.done metadata on the final frame
- recompute LeRobot-style metadata stats over the derived dataset
- write OpenPI normalization stats for the transformed training space
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Literal

import numpy as np
import pandas as pd


RIGHT_ARM_INDICES = np.asarray([7, 8, 9, 10, 11, 12, 13, 14])
RIGHT_ARM_NAMES = ["Right_J1", "Right_J2", "Right_J3", "Right_J4", "Right_J5", "Right_J6", "Right_J7", "right_gripper"]
VIDEO_KEYS = ("observation.images.chest_cam", "observation.images.wrist_cam_right")
NUMERIC_STATS_KEYS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "reward",
    "next.reward",
)


def convert_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    assets_dir: str | Path | None = None,
    dataset_name: str = "0423_right8_chest_rightwrist",
    overwrite: bool = False,
    video_copy_mode: Literal["copy", "hardlink"] = "copy",
) -> None:
    """Convert the source dataset into a right-arm-only derived dataset."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    assets_path = Path(assets_dir) if assets_dir is not None else None

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}. Pass overwrite=True or --overwrite.")
        shutil.rmtree(output_dir)

    (output_dir / "meta").mkdir(parents=True)
    (output_dir / "data/chunk-000").mkdir(parents=True)

    info = _load_json(source_dir / "meta/info.json")
    episodes = _load_jsonl(source_dir / "meta/episodes.jsonl")
    tasks = _load_jsonl(source_dir / "meta/tasks.jsonl")

    converted_episode_stats = []
    lerobot_stats_batches: dict[str, list[np.ndarray]] = {key: [] for key in NUMERIC_STATS_KEYS}
    norm_state_batches: list[np.ndarray] = []
    norm_action_batches: list[np.ndarray] = []

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        input_path = source_dir / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        output_path = output_dir / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        df = pd.read_parquet(input_path)
        converted = _convert_episode_dataframe(df)
        converted.to_parquet(output_path)

        episode_stats = _compute_stats_for_frame(converted)
        converted_episode_stats.append({"episode_index": episode_index, "stats": episode_stats})
        for key in NUMERIC_STATS_KEYS:
            lerobot_stats_batches[key].append(_column_to_2d_array(converted, key))

        state = _column_to_2d_array(converted, "observation.state")
        actions = _column_to_2d_array(converted, "action")
        delta_actions = actions.copy()
        delta_actions[:, :7] -= state[:, :7]
        norm_state_batches.append(state)
        norm_action_batches.append(delta_actions)

    _copy_selected_videos(source_dir, output_dir, [int(ep["episode_index"]) for ep in episodes], video_copy_mode)

    output_info = _convert_info(info, dataset_name=dataset_name)
    _write_json(output_dir / "meta/info.json", output_info)
    _write_json(output_dir / "meta/modality.json", _make_modality())
    _write_jsonl(output_dir / "meta/tasks.jsonl", tasks)
    _write_jsonl(output_dir / "meta/episodes.jsonl", episodes)
    _write_jsonl(output_dir / "meta/episodes_stats.jsonl", converted_episode_stats)
    _write_json(output_dir / "meta/stats.json", _merge_stats_batches(lerobot_stats_batches))

    if assets_path is not None:
        _write_openpi_norm_stats(
            assets_path,
            {
                "state": np.concatenate(norm_state_batches, axis=0),
                "actions": np.concatenate(norm_action_batches, axis=0),
            },
        )


def _convert_episode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    converted["action"] = converted["action"].map(_right_arm_vector)
    converted["observation.state"] = converted["observation.state"].map(_right_arm_vector)
    if "next.done" in converted:
        converted.loc[converted.index[-1], "next.done"] = True
    return converted


def _right_arm_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape[-1] != 15:
        raise ValueError(f"Expected 15-dim source vector, got shape {vector.shape}")
    return vector[RIGHT_ARM_INDICES].astype(np.float32, copy=True)


def _convert_info(info: dict, *, dataset_name: str) -> dict:
    output = dict(info)
    output["dataset_name"] = dataset_name
    output["total_videos"] = int(info["total_episodes"]) * len(VIDEO_KEYS)
    features = dict(info["features"])

    action_feature = dict(features["action"])
    action_feature["shape"] = [8]
    action_feature["names"] = RIGHT_ARM_NAMES
    state_feature = dict(features["observation.state"])
    state_feature["shape"] = [8]
    state_feature["names"] = RIGHT_ARM_NAMES

    output_features = {
        "action": action_feature,
        "observation.state": state_feature,
    }
    for key in VIDEO_KEYS:
        output_features[key] = features[key]
    for key, value in features.items():
        if value.get("dtype") != "video" and key not in output_features:
            output_features[key] = value
    output["features"] = output_features
    return output


def _make_modality() -> dict:
    return {
        "state": {
            "right_arm": {"start": 0, "end": 7},
            "right_gripper": {"start": 7, "end": 8},
        },
        "action": {
            "right_arm": {"start": 0, "end": 7},
            "right_gripper": {"start": 7, "end": 8},
        },
        "video": {
            "chest_cam": {"original_key": "observation.images.chest_cam"},
            "wrist_cam_right": {"original_key": "observation.images.wrist_cam_right"},
        },
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }


def _copy_selected_videos(
    source_dir: Path,
    output_dir: Path,
    episode_indices: list[int],
    video_copy_mode: Literal["copy", "hardlink"],
) -> None:
    for video_key in VIDEO_KEYS:
        target_dir = output_dir / "videos/chunk-000" / video_key
        target_dir.mkdir(parents=True, exist_ok=True)
        for episode_index in episode_indices:
            src = source_dir / "videos/chunk-000" / video_key / f"episode_{episode_index:06d}.mp4"
            dst = target_dir / src.name
            if video_copy_mode == "hardlink":
                dst.hardlink_to(src)
            else:
                shutil.copy2(src, dst)


def _compute_stats_for_frame(df: pd.DataFrame) -> dict:
    return {key: _stats_for_array(_column_to_2d_array(df, key)) for key in NUMERIC_STATS_KEYS}


def _merge_stats_batches(batches: dict[str, list[np.ndarray]]) -> dict:
    return {key: _stats_for_array(np.concatenate(values, axis=0)) for key, values in batches.items()}


def _column_to_2d_array(df: pd.DataFrame, key: str) -> np.ndarray:
    values = df[key].to_numpy()
    first = values[0]
    if isinstance(first, np.ndarray):
        return np.vstack(values).astype(np.float64)
    return values.astype(np.float64).reshape(-1, 1)


def _stats_for_array(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def _write_openpi_norm_stats(assets_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    norm_stats = {
        key: {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
        for key, values in arrays.items()
    }
    _write_json(assets_dir / "norm_stats.json", {"norm_stats": norm_stats})


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("0423_converted"))
    parser.add_argument("--output-dir", type=Path, default=Path("0423_right8_chest_rightwrist"))
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("assets/pi05_0423_right8_chest_wrist_finetune/0423_right8_chest_rightwrist"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-copy-mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()

    convert_dataset(
        args.source_dir,
        args.output_dir,
        assets_dir=args.assets_dir,
        overwrite=args.overwrite,
        video_copy_mode=args.video_copy_mode,
    )


if __name__ == "__main__":
    main()
