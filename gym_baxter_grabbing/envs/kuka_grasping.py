import pybullet as p
import pybullet_data
import numpy as np
import gym
import os
import random
from pathlib import Path
from gym_baxter_grabbing.envs.robot_grasping import RobotGrasping



class KukaGrasping(RobotGrasping):

    def __init__(self,
        display=False,
        obj='cube',
        steps_to_roll=1,
        random_var=None,
        mode='joint positions',
        delta_pos=[0, 0],
        obstacle_pos=None,
        obstacle_size=0.1,
        reset_random_initial_state=None,
        object_position=[0, 0.1, 0],
        object_xyzw=[0,0,0,1],
        joint_positions=None
	):
        
        self.obstacle_pos = None if obstacle_pos is None else np.array(obstacle_pos)
        self.obstacle_size = obstacle_size
        
        def load_kuka():
            id = self.p.loadSDF("kuka_iiwa/kuka_with_gripper.sdf")[0] # kuka_with_gripper2 gripper have a continuous joint (7)
            self.p.resetBasePositionAndOrientation(id, [-0.1, -0.5, -0.5], [0., 0., 0., 1.])
            return id

        super().__init__(
            robot=load_kuka,
            display=display,
            obj=obj,
            pos_cam=[1.3, 180, -40],
            gripper_display=True,
            steps_to_roll=steps_to_roll,
            random_var=random_var,
            delta_pos=delta_pos,
            reset_random_initial_state=reset_random_initial_state,
            object_position=object_position,
            object_xyzw=object_xyzw,
            joint_positions=joint_positions,
            table_height=0.8,
            joint_ids=[0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 13],
            contact_ids=[8, 9, 10, 11, 12, 13],
            n_control_gripper=4,
            mode = mode,
            end_effector_id = 6,
            n_actions = 8,
            center_workspace = 0,
            radius = 1.2,
            #disable_collision_pair = [[11,13]],
            change_dynamics = { # change joints ranges for gripper
                id:{'lateralFriction':1, 'jointLowerLimit':l, 'jointUpperLimit':h, 'maxJointVelocity':1, 'jointLimitForce':10, 'jointDamping':0.5} for id,l,h in [
                    (8, -0.5, -0.05), # b'base_left_finger_joint
                    (11, 0.05, 0.5), # b'base_right_finger_joint
                    (10, -0.3, 0.1), # b'left_base_tip_joint
                    (13, -0.1, 0.3)] # b'right_base_tip_joint
            } | {i:{'maxJointVelocity':0.5, 'jointLimitForce':100 if i==1 else 50} for i in range(7)} # decrease max force & velocity
        )
        if self.obstacle_pos is not None:
            # create the obstacle object at the required location
            col_id = self.p.createCollisionShape(self.p.GEOM_SPHERE, radius=self.obstacle_size)
            viz_id = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=self.obstacle_size, rgbaColor=[0, 0, 0, 1])
            obs_id = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
            pos_obstacle = np.array(self.p.getBasePositionAndOrientation(self.obj_id)[0]) + self.obstacle_pos

            self.p.resetBasePositionAndOrientation(obs_id, pos_obstacle, [0, 0, 0, 1])


    
    def get_object(self, obj=None):
        if obj == 'cuboid':
            return {"shape":'cuboid', "x":0.06, "y":0.06, "z":0.16}
        elif obj == 'cube':
            return {"shape": 'cube', "unit":0.055}
        elif obj == 'sphere':
            return {"shape":'sphere', "radius":0.055}
        elif obj == 'cylinder':
            return {"shape":'cylinder', "radius":0.032, "z":0.15}
        elif obj == 'paper roll':
            return {"shape":'cylinder', "radius":0.021, "z":0.22}
        else:
            return obj



    def step(self, action):
        fingers = -action[-1], -action[-1], action[-1], action[-1]
        if self.mode == 'joint positions':
            # we want one action per joint (gripper is composed by 4 joints but considered as one)
            assert(len(action) == self.n_actions)
            self.info['closed gripper'] = action[-1]<0
            # add the 3 commands for the 3 last gripper joints
            commands = np.hstack([action[:-1], *fingers])
        elif self.mode == 'inverse kinematic':
            pass
        elif self.mode == 'joint torques':
            # control the gripper in positions
            for id, a, v, f, u, l in zip(self.joint_ids[-4:], fingers, self.maxVelocity[-4:], self.maxForce[-4:], self.upperLimits[-4:], self.lowerLimits[-4:]):
                self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=id, controlMode=self.p.POSITION_CONTROL, targetPosition=l+(a+1)/2*(u-l), maxVelocity=v, force=f)
            commands = action[:-1]

        # apply the commands
        return super().step(commands)

    
    def reset_robot(self):
        for i, in zip(self.joint_ids,):
            p.resetJointState(self.robot_id, i, targetValue=0)


