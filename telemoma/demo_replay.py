import argparse
import time

import h5py
import numpy as np
import rospy

from telemoma.human_interface.teleop_core import TeleopAction
from telemoma.robot_interface.tiago.tiago_wrapper import TiagoWrapper
from telemoma.demo_collect import load_teleop_config, unflatten_action


def replay_trajectory(hdf5_path: str, teleop_cfg):
    """
    Replay actions from an HDF5 file saved by demo_collect.py.
    """
    # Load trajectory from HDF5
    with h5py.File(hdf5_path, "r") as f:
        # Find the demo group (usually "demo_0", "demo_1", etc.)
        data_grp = f["data"]
        demo_name = list(data_grp.keys())[0]  # Get first demo
        demo_grp = data_grp[demo_name]
        
        actions = demo_grp["actions"][:]  # [T, 11]
        num_steps = actions.shape[0]
        print(f"Loaded {num_steps} actions from {hdf5_path}")

    # Initialize environment (same config as collection)
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

    time.sleep(3)  # Wait for robot to reset
    obs = env.reset(reset_arms=True)

    # Replay actions
    print(f"Replaying {num_steps} actions...")
    for i in range(num_steps):
        # Unflatten action: [left (8), base (3)] = 11 dims
        action = unflatten_action(actions[i], env)
        print(f"action: {action}")
        obs, reward, done, info = env.step(action)
        
        if (i + 1) % 10 == 0:
            print(f"Step {i + 1}/{num_steps}")

    print("Replay complete!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to the HDF5 file to replay.",
    )
    parser.add_argument(
        "--teleop_config",
        type=str,
        default=None,
        help="Path to teleop config .py (should match the one used for collection).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rospy.init_node("telemoma_tiago_replay")

    teleop_cfg = load_teleop_config(args.teleop_config)
    replay_trajectory(args.hdf5_path, teleop_cfg)


if __name__ == "__main__":
    main()
