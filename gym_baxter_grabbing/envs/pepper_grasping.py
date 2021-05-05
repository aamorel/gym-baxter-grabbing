import pybullet as p
import qibullet as q
import gym
import os
import random
from pathlib import Path
from gym_baxter_grabbing.envs.robot_grasping import RobotGrasping



class PepperGrasping(RobotGrasping):

    def __init__(self, display=False, obj='cube', random_obj=False, steps_to_roll=1, random_var=None,
                 delta_pos=[0, 0], mode='joint positions'):
                 
        self.pepper = q.PepperVirtual()
        self.joints = ['HipRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LHand']

        super().__init__(
            robot=lambda: self.pepper.loadRobot(translation=[0.2, -0.38, -0.8], quaternion=[0,0,0,1], physicsClientId=self.physicsClientId).getRobotModel(),
            display=display,
            obj=obj,
            mode=mode,
            pos_cam=[0.5, 180, -40],
            gripper_display=False,
            steps_to_roll=steps_to_roll,
            random_var=random_var,
            delta_pos=delta_pos,
            initial_position_object=[0, -0.15, 0],
            table_height=0.8,
            end_effector_id=self.pepper.joint_dict['LHand'].getIndex(),
            joint_ids=[self.pepper.joint_dict[joint].getIndex() for joint in self.joints],
            n_actions=len(self.joints),
            center_workspace=0,
            radius=1,
            contact_ids=list(range(36, 50)),
        )

        # self.joints = ['HipRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw', 'LFinger21', 'LFinger22', 'LFinger23', 'LFinger11', 'LFinger12', 'LFinger13', 'LFinger41', 'LFinger42', 'LFinger43', 'LFinger31', 'LFinger32', 'LFinger33', 'LThumb1', 'LThumb2']

        
    def get_obj(self, obj=None):
        if obj == 'cuboid':
            return {"shape":'cuboid', "x":0.04, "y":0.04, "z":0.04}
        else:
            return obj
            
    def step(self, action):
        if self.mode == 'joint posisitons':
            # we want one action per joint, all fingers are controlled with LHand
            assert(len(action) == self.n_actions)
            self.info['closed gripper'] = action[-1]<0
            commands = (np.array(action)+1)/2*(self.upperLimits-self.lowerLimits) + self.lowerLimits

        # apply the commands
        self.pepper.setAngles(joint_names=self.joints, joint_values=commands, percentage_speed=1)
        
        return super().step() # do not set the motors as we already dit it
