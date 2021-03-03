import gym
import pybullet as p
from time import sleep


class RobotGrasping(gym.Env):

    def __init__(self, display=False, obj='cube', random_obj=False, pos_cam=[1.3, 180, -40],
                 gripper_display=False, steps_to_roll=1):
        self.obj = obj
        self.display = display
        self.random_obj = random_obj
        self.pos_cam = pos_cam
        self.gripper_display = gripper_display
        self.steps_to_roll = steps_to_roll
        if self.display:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        # setup the world
        self.setup_world()

        # save the state
        self.save_state = p.saveState()

        # configure display if necessary
        if self.display:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(self.pos_cam[0], self.pos_cam[1], self.pos_cam[2], [0, 0, 0])
            p.getCameraImage(320, 200, renderer=p.ER_BULLET_HARDWARE_OPENGL)

            if self.gripper_display:
                self.line_width = 4
                self.line = p.addUserDebugLine([0, 0, 0], [0.2, 0, 0], [255, 0, 0], lineWidth=self.line_width,
                                               parentObjectUniqueId=self.robot_id, parentLinkIndex=self.end_effector_id)
            sleep(1.)

    def setup_world(self):
        raise NotImplementedError('Setup world is not implemented.')

    def step(self):
        if self.display and self.gripper_display:
            self.line = p.addUserDebugLine([0, 0, 0], [0.2, 0, 0], [255, 0, 0], lineWidth=self.line_width,
                                           replaceItemUniqueId=self.line,
                                           parentObjectUniqueId=self.robot_id,
                                           parentLinkIndex=self.end_effector_id)

    def reset(self):
        p.restoreState(self.savestate)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
