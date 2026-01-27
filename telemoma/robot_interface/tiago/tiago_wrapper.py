import gym
import numpy as np

from telemoma.robot_interface.tiago.tiago_gym import TiagoGym

import cv2


class TiagoWrapper(TiagoGym):
    """
    Thin wrapper around TiagoGym that:
      - crops the head camera image horizontally
      - renames 'tiago_head_image' -> 'tiago_agentview_image'
    """

    def __init__(
        self,
        frequency=10,
        head_policy=None,
        base_enabled=False,
        torso_enabled=False,
        right_arm_enabled=True,
        left_arm_enabled=True,
        right_gripper_type=None,
        left_gripper_type=None,
        external_cams=None,
    ):
        if external_cams is None:
            external_cams = {}
        super().__init__(
            frequency=frequency,
            head_policy=head_policy,
            base_enabled=base_enabled,
            torso_enabled=torso_enabled,
            right_arm_enabled=right_arm_enabled,
            left_arm_enabled=left_arm_enabled,
            right_gripper_type=right_gripper_type,
            left_gripper_type=left_gripper_type,
            external_cams=external_cams,
        )

    @property
    def observation_space(self):
        """
        Mirror the parent observation space but expose the renamed key.
        """
        base_space = super().observation_space
        if isinstance(base_space, gym.spaces.Dict):
            spaces = dict(base_space.spaces)
            if "tiago_head_image" in spaces:
                spaces["tiago_agentview_image"] = spaces.pop("tiago_head_image")
            return gym.spaces.Dict(spaces)
        return base_space

    def _process_obs(self, obs):
        """
        Apply cropping and key rename on a single observation dict.
        """
        if "tiago_head_image" in obs and obs["tiago_head_image"] is not None:
            # Convert to a proper image array for OpenCV
            arr = np.asarray(obs["tiago_head_image"])
            # Ensure we have a numeric, contiguous array with at most 4 channels
            if arr.dtype == np.object_:
                arr = np.array(arr.tolist())
            arr = np.ascontiguousarray(arr)
            if arr.ndim >= 2 and arr.shape[1] > 160:
                arr = arr[:, 80:-80]
            # Resize it to 256x256; cast to uint8 so OpenCV has a supported type
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_AREA)
            # Rename key
            obs["tiago_agentview_image"] = arr
            del obs["tiago_head_image"]


        del_keys = []
        # if right_arm  is not enabled, remove right
        if not self.right_arm_enabled:
            del obs["right"]
        if not self.left_arm_enabled:
            del obs["left"]
        if not self.base_enabled:
            del obs["base"]
        if not self.torso_enabled:
            del obs["torso"]
        return obs

    def _observation(self):
        base_obs = super()._observation()
        return self._process_obs(base_obs)

