import dataclasses

import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class RightArmInputs(transforms.DataTransformFn):
    """Inputs for right-arm 0423 pi0.5 fine-tuning.

    Expected repacked inputs:
    - image/chest: chest camera image
    - image/right_wrist: right wrist camera image
    - state: right arm joints plus right gripper, shape [8]
    - actions: optional right arm action chunk, shape [horizon, 8]
    - prompt: optional task prompt
    """

    def __call__(self, data: dict) -> dict:
        chest_image = _parse_image(data["image"]["chest"])
        right_wrist_image = _parse_image(data["image"]["right_wrist"])

        inputs = {
            "state": np.asarray(data["state"]),
            "image": {
                "base_0_rgb": chest_image,
                "left_wrist_0_rgb": np.zeros_like(chest_image),
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class RightArmOutputs(transforms.DataTransformFn):
    """Outputs right-arm actions only."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
