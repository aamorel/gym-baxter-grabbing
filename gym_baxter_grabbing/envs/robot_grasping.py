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
                self.line_1 = p.addUserDebugLine([0, 0, 0], [0.2, 0, 0], [255, 0, 0], lineWidth=self.line_width,
                                                 parentObjectUniqueId=self.robot_id,
                                                 parentLinkIndex=self.end_effector_id)
                self.line_2 = p.addUserDebugLine([0, 0, 0], [0, 0.2, 0], [0, 255, 0], lineWidth=self.line_width,
                                                 parentObjectUniqueId=self.robot_id,
                                                 parentLinkIndex=self.end_effector_id)
                self.line_3 = p.addUserDebugLine([0, 0, 0], [0, 0, 0.2], [0, 0, 255], lineWidth=self.line_width,
                                                 parentObjectUniqueId=self.robot_id,
                                                 parentLinkIndex=self.end_effector_id)
            sleep(1.)

    def setup_world(self):
        raise NotImplementedError('setup_world is not implemented.')

    def actuate(self):
        raise NotImplementedError('actuate is not implemented.')

    def compute_joint_poses(self):
        raise NotImplementedError('compute_joint_poses is not implemented')

    def compute_grip_info(self):
        raise NotImplementedError('compute_grip_info is not implemented')
    
    def compute_self_contact(self):
        pass

    def step(self, action):
        self.action = action
        self.actuate()

        # roll the world (motor control doesn't have to be done every loop)
        for _ in range(self.steps_to_roll):
            p.stepSimulation()

        # get information on target object
        obj = p.getBasePositionAndOrientation(self.obj_id)
        obj_pos = list(obj[0])  # x, y, z
        obj_orientation = list(obj[1])

        # get information on gripper
        grip = p.getLinkState(self.robot_id, self.end_effector_id)
        grip_pos = list(grip[0])  # x, y, z
        grip_orientation = list(grip[1])

        self.info = {}
        self.compute_joint_poses()
        self.compute_grip_info()

        observation = [obj_pos, obj_orientation, grip_pos, grip_orientation, self.joint_poses]
        
        contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.obj_id)
        self.info['contact_points'] = contact_points
        self.compute_self_contact()

        if self.display and self.gripper_display:
            self.line_1 = p.addUserDebugLine([0, 0, 0], [0.2, 0, 0], [255, 0, 0], lineWidth=self.line_width,
                                             replaceItemUniqueId=self.line_1,
                                             parentObjectUniqueId=self.robot_id,
                                             parentLinkIndex=self.end_effector_id)
            self.line_2 = p.addUserDebugLine([0, 0, 0], [0, 0.2, 0], [0, 255, 0], lineWidth=self.line_width,
                                             replaceItemUniqueId=self.line_2,
                                             parentObjectUniqueId=self.robot_id,
                                             parentLinkIndex=self.end_effector_id)
            self.line_3 = p.addUserDebugLine([0, 0, 0], [0, 0, 0.2], [0, 0, 255], lineWidth=self.line_width,
                                             replaceItemUniqueId=self.line_3,
                                             parentObjectUniqueId=self.robot_id,
                                             parentLinkIndex=self.end_effector_id)            
        reward = None
        done = False
        return observation, reward, done, self.info

    def reset(self):
        p.restoreState(self.save_state)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
