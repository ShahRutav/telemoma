import rospy

from telemoma.human_interface.teleop_policy import TeleopPolicy
from telemoma.human_interface.teleop_core import TeleopAction

from importlib.machinery import SourceFileLoader

COMPATIBLE_ROBOTS = ['tiago', 'hsr']

def main(args):
    teleop_config = SourceFileLoader('conf', args.teleop_config).load_module().teleop_config
    print(teleop_config)

    if args.robot == 'tiago':
        from telemoma.robot_interface.tiago.tiago_gym import TiagoGym
        from telemoma.robot_interface.tiago.tiago_wrapper import TiagoWrapper
        env = TiagoWrapper(
                frequency=10,
                head_policy=None,
                base_enabled=teleop_config.base_controller is not None,
                torso_enabled=teleop_config.base_controller is not None,
                right_arm_enabled=None,
                left_arm_enabled=teleop_config.arm_left_controller is not None,
                right_gripper_type=None,
                left_gripper_type='pal'
             )
    elif args.robot == 'hsr':
        from telemoma.robot_interface.hsr.hsr_gym import HSRGym
        env = HSRGym(
                frequency=10,
                head_policy=None,
                base_enabled=True,
                torso_enabled=False,
                arm_enabled=True,
            )
    else:
        raise ValueError(f'Unknown robot: {args.robot}')
    obs = env.reset()

    teleop = TeleopPolicy(teleop_config)
    teleop.start()

    def shutdown_helper():
        teleop.stop()
    rospy.on_shutdown(shutdown_helper)

    while not rospy.is_shutdown():
        action = teleop.get_action(obs) # get_random_action()
        buttons = action.extra['buttons'] if 'buttons' in action.extra else {}
        # print("buttons:")
        # for key, value in buttons.items():
        #     print(f"  {key}: {value}")
    
        if buttons.get('A', False) or buttons.get('B', False):
            break

        print(   f"action:\n{action}")
        obs, _, _, _ = env.step(action)
        # print(obs.keys())
        # print(   obs['right'])
        # print(   obs['left'])
        print(   f"base:\n{obs['base']}")
        # print(   "base_velocity: ", obs['base_velocity'])

    shutdown_helper()

if __name__ == "__main__":
    print("rospy.init_node('telemoma_real')")
    rospy.init_node('telemoma_real')
    print("rospy.init_node('telemoma_real') done")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot', type=str, default='tiago', help='Robot to use. Choose between tiago and hsr.')
    parser.add_argument('--teleop_config', type=str, help='Path to the teleop config to use.')
    args = parser.parse_args()

    assert args.robot in COMPATIBLE_ROBOTS, f'Unknown robots. Choose one from: {" ".join(COMPATIBLE_ROBOTS)}' 
    main(args)