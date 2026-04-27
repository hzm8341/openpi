import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_0423_right8_dataset import convert_dataset


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _make_source_dataset(root: Path) -> None:
    features = {
        "action": {
            "dtype": "float32",
            "shape": [15],
            "names": [f"joint_{i}" for i in range(14)] + ["right_gripper"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [15],
            "names": [f"joint_{i}" for i in range(14)] + ["right_gripper"],
        },
        "observation.images.chest_cam": {"dtype": "video", "shape": [480, 640, 3], "names": ["h", "w", "c"]},
        "observation.images.head_cam": {"dtype": "video", "shape": [480, 640, 3], "names": ["h", "w", "c"]},
        "observation.images.wrist_cam_left": {"dtype": "video", "shape": [480, 640, 3], "names": ["h", "w", "c"]},
        "observation.images.wrist_cam_right": {"dtype": "video", "shape": [480, 640, 3], "names": ["h", "w", "c"]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "reward": {"dtype": "float32", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        "done": {"dtype": "bool", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "dataset_name": "source",
            "robot_type": "custom",
            "total_episodes": 2,
            "total_frames": 5,
            "total_tasks": 1,
            "total_videos": 8,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": 15.0,
            "splits": {"train": "0:2"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        },
    )
    _write_json(root / "meta/modality.json", {})
    _write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": "pick test tube"}])
    _write_jsonl(
        root / "meta/episodes.jsonl",
        [
            {"episode_index": 0, "tasks": ["pick test tube"], "length": 3},
            {"episode_index": 1, "tasks": ["pick test tube"], "length": 2},
        ],
    )
    _write_jsonl(root / "meta/episodes_stats.jsonl", [])
    _write_json(root / "meta/stats.json", {})

    global_index = 0
    for episode_index, length in [(0, 3), (1, 2)]:
        rows = []
        for frame_index in range(length):
            base = episode_index * 100 + frame_index * 10
            action = np.asarray([base + i for i in range(15)], dtype=np.float32)
            state = np.asarray([base + i + 0.5 for i in range(15)], dtype=np.float32)
            done = frame_index == length - 1
            rows.append(
                {
                    "action": action,
                    "observation.state": state,
                    "timestamp": np.float32(frame_index / 15.0),
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "index": global_index,
                    "task_index": 0,
                    "reward": np.float32(0.0),
                    "done": done,
                    "next.reward": np.float32(0.0),
                    "next.done": False,
                }
            )
            global_index += 1
        rows[-2]["next.done"] = True
        path = root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path)

    for video_key in [
        "observation.images.chest_cam",
        "observation.images.head_cam",
        "observation.images.wrist_cam_left",
        "observation.images.wrist_cam_right",
    ]:
        video_dir = root / "videos/chunk-000" / video_key
        video_dir.mkdir(parents=True, exist_ok=True)
        for episode_index in [0, 1]:
            (video_dir / f"episode_{episode_index:06d}.mp4").write_bytes(f"{video_key}-{episode_index}".encode())


def test_convert_dataset_keeps_right_arm_and_selected_cameras(tmp_path: Path):
    source = tmp_path / "source"
    output = tmp_path / "right8"
    assets = tmp_path / "assets"
    _make_source_dataset(source)

    convert_dataset(source, output, assets_dir=assets, video_copy_mode="copy")

    info = json.loads((output / "meta/info.json").read_text())
    assert info["dataset_name"] == "0423_right8_chest_rightwrist"
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 5
    assert info["total_videos"] == 4
    assert info["features"]["action"]["shape"] == [8]
    assert info["features"]["observation.state"]["shape"] == [8]
    assert set(k for k, v in info["features"].items() if v["dtype"] == "video") == {
        "observation.images.chest_cam",
        "observation.images.wrist_cam_right",
    }

    df = pd.read_parquet(output / "data/chunk-000/episode_000000.parquet")
    np.testing.assert_array_equal(df["action"].iloc[0], np.asarray([7, 8, 9, 10, 11, 12, 13, 14], dtype=np.float32))
    np.testing.assert_array_equal(
        df["observation.state"].iloc[0], np.asarray([7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5], dtype=np.float32)
    )
    assert df["done"].tolist() == [False, False, True]
    assert df["next.done"].tolist() == [False, True, True]

    assert (output / "videos/chunk-000/observation.images.chest_cam/episode_000000.mp4").read_bytes()
    assert (output / "videos/chunk-000/observation.images.wrist_cam_right/episode_000001.mp4").read_bytes()
    assert not (output / "videos/chunk-000/observation.images.head_cam").exists()

    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["episode_index"]["min"] == [0.0]
    assert stats["episode_index"]["max"] == [1.0]
    assert stats["action"]["count"] == [5]
    assert stats["action"]["min"] == [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    assert stats["action"]["max"] == [117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0]

    norm_stats = json.loads((assets / "norm_stats.json").read_text())["norm_stats"]
    assert set(norm_stats) == {"state", "actions"}
    assert len(norm_stats["state"]["mean"]) == 8
    assert len(norm_stats["actions"]["q01"]) == 8
