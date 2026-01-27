import argparse
import copy
import os
import select
import sys

import h5py
import numpy as np
import rospy
from importlib.machinery import SourceFileLoader
from termcolor import colored

from telemoma.human_interface.teleop_policy import TeleopPolicy
from telemoma.robot_interface.tiago.tiago_wrapper import TiagoWrapper
from telemoma.utils.general_utils import AttrDict
from telemoma.configs.only_keyboard import teleop_config as default_teleop_config


class TranslationExplorationPolicy:
    """
    Simple real-robot exploration policy that translates the base along the y axis.

    Behavior (in base frame):
    - Move in +y until the base y position reaches +0.05 (relative to the start).
    - Then move in -y until the base y position reaches -0.05.
    - Finally move in +y until the base y position is back at 0.0.

    The commanded y velocity magnitude is sampled uniformly in [0.75, 1.0].

    This mirrors the structure of the simulated translate exploration policies in robocasa,
    but operates directly on the Tiago observations and returns only a base command.
    """

    def __init__(self, env, velocity_mode: str = "random"):
        """
        Args:
            env: TiagoWrapper (or TiagoGym-like) environment that provides 'base' in its obs.
            velocity_mode: "random" (sample speed in [0.75, 1.0]) or "fixed".
        """
        self.env = env
        self.velocity_mode = velocity_mode

        self._initialized = False
        self.init_y = 0.0
        self.target_delta_value = 1.0

        self.stage_one_done = False
        self.stage_two_done = False
        self.stage_three_done = False

        self.velocity = 0.8

    def begin(self, init_obs=None):
        """
        Initialize the exploration trajectory from the current observation.
        """
        if init_obs is None:
            raise ValueError("TranslationExplorationPolicy.begin requires an initial observation.")

        base = np.asarray(init_obs["base"]).reshape(-1)
        assert base.shape[0] >= 2, "Expected base observation to have at least (x, y, theta)."

        # Treat the current y as 0.0 and define targets relative to it.
        self.init_y = float(base[1])
        self.target_y_pos = self.init_y + self.target_delta_value
        self.target_y_neg = self.init_y - self.target_delta_value

        if self.velocity_mode == "random":
            self.velocity = float(np.random.uniform(0.75, 1.0))
        elif self.velocity_mode == "fixed":
            self.velocity = 0.85
        else:
            raise ValueError(f"Invalid velocity mode: {self.velocity_mode}")

        self.stage_one_done = False
        self.stage_two_done = False
        self.stage_three_done = False
        self._initialized = True

    def explore_env(self, obs):
        """
        Given the latest observation, return a base command [vx, vy, wz].

        Returns:
            np.ndarray of shape (3,) for the base command, or
            None when the exploration trajectory is complete.
        """
        if not self._initialized:
            self.begin(init_obs=obs)

        base = np.asarray(obs["base"]).reshape(-1)
        print(   f"base: {base}")
        y = float(base[1])

        # Small tolerance on the target positions to avoid oscillations.
        eps = 0.02

        if not self.stage_one_done and y >= self.target_y_pos - eps:
            self.stage_one_done = True
        elif self.stage_one_done and not self.stage_two_done and y <= self.target_y_neg + eps:
            self.stage_two_done = True
        elif (
            self.stage_one_done
            and self.stage_two_done
            and not self.stage_three_done
            and abs(y - self.init_y) <= eps
        ):
            self.stage_three_done = True

        if self.stage_one_done and self.stage_two_done and not self.stage_three_done:
            print(   f"distance: {abs(y - self.init_y)}")

        if self.stage_one_done and self.stage_two_done and self.stage_three_done:
            # Exploration finished.
            return None

        # Choose direction based on which stage we're in.
        if not self.stage_one_done:
            vy = self.velocity  # move towards +0.05
        elif not self.stage_two_done:
            vy = -self.velocity  # move towards -0.05
        else:
            vy = self.velocity  # move back towards 0.0

        base_action = np.array([0.0, vy, 0.0], dtype=np.float32)
        print(   f"base_action: {base_action}")
        return base_action


def clear_input_buffer():
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.read(1)


def user_input(text, valid_inputs):
    _input = input(text)
    while _input not in valid_inputs:
        _input = input(text)
    return _input


def confirm_user(question, info_string=None):
    clear_input_buffer()
    if info_string is not None:
        print(colored(info_string, "magenta"))
    _input = user_input(question, valid_inputs=["y", "n"])
    return _input == "y"


def load_teleop_config(path: str) -> AttrDict:
    """
    Load a teleop config from a python file, or fall back to the Oculus-only config.
    """
    if path is None:
        return default_teleop_config
    module = SourceFileLoader("telemoma_conf", path).load_module()
    return module.teleop_config


def flatten_action(action, env) -> np.ndarray:
    """
    Convert a TeleopAction into a single flat vector, similar to robosuite.
    Order (when enabled): [left, base].
    """
    parts = []
    assert env.left_arm_enabled and env.base_enabled
    parts.append(np.asarray(action.left).reshape(-1))
    parts.append(np.asarray(action.base).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)

def unflatten_action(action, env) -> dict:
    """
    Convert a flat action vector back into a TeleopAction.
    """
    assert len(action) == 11
    return TeleopAction(
        left=action[:8],
        base=action[-3],
    )

def collect_trajectory(teleop_cfg: AttrDict, use_exploration: bool) -> dict:
    """
    Run one teleop session on Tiago and return a trajectory dict.
    The returned structure is close to robosuite / robocasa:

        {
            "obs": {key: np.array[T, ...], ...},
            "actions": np.array[T, act_dim],
            "policy_mode": np.array[T],  # 1 if X pressed at that step, else 0
        }
    """
    env = TiagoWrapper(
        frequency=10,
        head_policy=None,
        base_enabled=teleop_cfg.base_controller is not None,
        torso_enabled=None,
        right_arm_enabled=None,
        left_arm_enabled=teleop_cfg.arm_left_controller is not None,
        right_gripper_type=None,
        left_gripper_type="pal",
    )

    obs = env.reset()

    teleop = TeleopPolicy(teleop_cfg)
    teleop.start()

    def shutdown_helper():
        teleop.stop()

    rospy.on_shutdown(shutdown_helper)

    traj = dict(obs={}, actions=[], policy_mode=[])

    # -------------------------------------------------------------------------
    # Automatic base translation exploration before handing control to the user.
    # -------------------------------------------------------------------------
    if env.base_enabled and use_exploration:
        print(colored("Starting automatic translation exploration of the base.", "cyan"))
        exploration_policy = TranslationExplorationPolicy(env)
        exploration_policy.begin(init_obs=obs)

        while not rospy.is_shutdown():
            base_cmd = exploration_policy.explore_env(obs)
            if base_cmd is None:
                break

            # Use the teleop policy's default action structure and fill only the base.
            action = teleop.get_default_action()
            action.base = base_cmd

            next_obs, reward, done_env, info = env.step(action)

            # Log transition: autonomous exploration, so policy_mode = 0.
            traj["actions"].append(flatten_action(action, env))
            traj["policy_mode"].append(1)

            log_obs = copy.deepcopy(obs)
            for k, v in log_obs.items():
                if v is None:
                    continue
                arr = np.asarray(v)
                traj["obs"].setdefault(k, []).append(arr)

            obs = next_obs

        if rospy.is_shutdown():
            shutdown_helper()
            return None

        if not confirm_user(
            "Exploration finished. Continue teleoperation and data collection? (y/n): "
        ):
            shutdown_helper()
            return None

    # -------------------------------------------------------------------------
    # Main teleoperation loop.
    # -------------------------------------------------------------------------
    while not rospy.is_shutdown():
        action = teleop.get_action(obs)
        buttons = action.extra.get("buttons", {}) if hasattr(action, "extra") else {}

        # Step the real robot
        next_obs, reward, done_env, info = env.step(action)

        # User control of episode end / cancel via Oculus buttons
        done = bool(done_env or buttons.get("A", False))
        cancel = bool(buttons.get("B", False))

        # Log transition
        traj["actions"].append(flatten_action(action, env))
        # policy_mode: 1 if X button is pressed at this step, else 0
        traj["policy_mode"].append(0)

        print(   f"obs: {obs.keys()}")
        # Copy obs to avoid accidental mutation
        log_obs = copy.deepcopy(obs)
        for k, v in log_obs.items():
            if v is None:
                continue
            arr = np.asarray(v)
            traj["obs"].setdefault(k, []).append(arr)

        obs = next_obs

        if cancel:
            shutdown_helper()
            return None
        if done:
            break

    shutdown_helper()

    # Convert lists to numpy arrays
    traj["actions"] = np.asarray(traj["actions"], dtype=np.float32)
    traj["policy_mode"] = np.asarray(traj["policy_mode"], dtype=np.int32)

    obs_dict = {}
    for k, seq in traj["obs"].items():
        if len(seq) == 0:
            continue
        obs_dict[k] = np.stack(seq, axis=0)
    traj["obs"] = obs_dict

    return traj


def save_trajectory_to_hdf5(traj: dict, save_dir: str) -> str:
    """
    Save a single trajectory to an HDF5 file with a robosuite-like structure:

        /data/demo_<id>/
            actions       [T, act_dim]
            policy_mode   [T]
            obs/<key>     [T, ...]
    """
    os.makedirs(save_dir, exist_ok=True)
    existing = [f for f in os.listdir(save_dir) if f.endswith((".hdf5", ".h5"))]
    demo_id = len(existing)

    path = os.path.join(save_dir, f"demo_{demo_id}.hdf5")

    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        ep_name = f"demo_{demo_id}"
        ep_grp = data_grp.create_group(ep_name)

        ep_grp.create_dataset("actions", data=traj["actions"])
        ep_grp.create_dataset("policy_mode", data=traj["policy_mode"])

        obs_grp = ep_grp.create_group("obs")
        for k, v in traj["obs"].items():
            print(   f"k: {k}, v: {v}")
            obs_grp.create_dataset(k, data=v, compression="gzip")

        ep_grp.attrs["num_samples"] = traj["actions"].shape[0]

    print(f"Saved trajectory with {traj['actions'].shape[0]} steps to {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        type=str,
        default="tiago",
        help="Robot to use (only 'tiago' is supported here).",
    )
    parser.add_argument(
        "--teleop_config",
        type=str,
        default=None,
        help="Path to teleop config .py (if omitted, use Oculus-only config).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory where demo_<id>.hdf5 files will be written.",
    )
    parser.add_argument(
        "--exploration",
        action="store_true",
        help="If set, run the TranslationExplorationPolicy before teleoperation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.robot != "tiago":
        raise ValueError("This demo_collect script currently supports only the Tiago robot.")

    rospy.init_node("telemoma_tiago_collect")

    teleop_cfg = load_teleop_config(args.teleop_config)
    traj = collect_trajectory(teleop_cfg, use_exploration=args.exploration)

    if traj is not None:
        info = f"Collected trajectory with {traj['actions'].shape[0]} timesteps."
        if confirm_user("Save this trajectory? (y/n): ", info_string=info):
            save_trajectory_to_hdf5(traj, args.save_dir)
        else:
            print("User chose not to save trajectory.")
    else:
        print("Trajectory cancelled. Nothing saved.")


if __name__ == "__main__":
    main()

