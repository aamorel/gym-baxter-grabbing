import pybullet as p
import pybullet_data
import numpy as np
import gym
import os
import random
from pathlib import Path
from gym_baxter_grabbing.envs.robot_grasping import RobotGrasping

MAX_FORCE = 100


def setMotors(bodyId, joint_ids, jointPoses):
    """
    Parameters
    ----------
    bodyId : int
    jointPoses : [float] * numDofs
    """

    for i, id in enumerate(joint_ids):

        p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=id, controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i], force=MAX_FORCE, maxVelocity=0.7)


def getJointStates(bodyId, includeFixed=False):
    """
    Parameters
    ----------
    bodyId : int
    includeFixed : bool

    Returns
    -------
    lowerLimits : [ float ] * numDofs

    """

    states = []

    numJoints = p.getNumJoints(bodyId)
    # loop through all joints
    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)
        # print(jointInfo[0], jointInfo[1], jointInfo[2], jointInfo[3], jointInfo[8:10])
        if includeFixed or jointInfo[3] > -1:
            # jointInfo[3] > -1 means that the joint is not fixed
            joint_state = p.getJointState(bodyId, i)[0]
            states.append(joint_state)

    return states


def getJointRanges2(bodyId, includeFixed=False):
    """
    Parameters
    ----------
    bodyId : int
    includeFixed : bool

    Returns
    -------
    lowerLimits : [ float ] * numDofs
    upperLimits : [ float ] * numDofs
    jointRanges : [ float ] * numDofs
    restPoses : [ float ] * numDofs
    """

    lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

    numJoints = p.getNumJoints(bodyId)
    # loop through all joints
    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)
        # print(jointInfo[0], jointInfo[1], jointInfo[2], jointInfo[3], jointInfo[8:10])
        if includeFixed or jointInfo[3] > -1:
            # jointInfo[3] > -1 means that the joint is not fixed
            ll, ul = jointInfo[8:10]
            jr = ul - ll

            # For simplicity, assume resting state == initial state
            rp = p.getJointState(bodyId, i)[0]

            lowerLimits.append(ll)
            upperLimits.append(ul)
            jointRanges.append(jr)
            restPoses.append(rp)

    return lowerLimits, upperLimits, jointRanges, restPoses


class KukaGrasping(RobotGrasping):

    def __init__(self, display=False, obj='cube', random_obj=False, steps_to_roll=1, random_var=None, mode='joint positions',
                 delta_pos=[0, 0], obstacle=False, obstacle_pos=[0, 0, 0], obstacle_size=0.1):
        
        self.obstacle = obstacle
        self.obstacle_pos = obstacle_pos
        self.obstacle_size = obstacle_size
        
        def load_kuka():
            id = self.p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")[0]
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
            table_height=0.8,
            joint_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13],
            contact_ids=[8, 9, 10, 11, 12, 13],
            mode = mode,
            end_effector_id = 6,
            n_actions = 9,
            center_workspace = 0,
            radius = 0.65,
            disable_collision_pair = [[11,13]],
            change_dynamics = { # change joints ranges for gripper
                8:{'lateralFriction':1, 'jointLowerLimit':-0.5, 'jointUpperLimit':-0.05}, # b'base_left_finger_joint
                11:{'lateralFriction':1, 'jointLowerLimit':0.05, 'jointUpperLimit':0.5},  # b'base_right_finger_joint
                10:{'lateralFriction':1, 'jointLowerLimit':-0.3, 'jointUpperLimit':0.1},   # b'left_base_tip_joint
                13:{'lateralFriction':1, 'jointLowerLimit':-0.1, 'jointUpperLimit':0.3}   # b'right_base_tip_joint
            }
        )

        self.joint_ranges = getJointRanges2(self.robot_id)

        self.joint_ranges = np.array(self.joint_ranges)

        

        # base of gripper
        self.joint_ranges[0, 8] = -0.05
        self.joint_ranges[1, 8] = -0.5
        self.joint_ranges[0, 10] = 0.05
        self.joint_ranges[1, 10] = 0.5

        # tip of gripper
        self.joint_ranges[0, 9] = 0.1
        self.joint_ranges[1, 9] = -0.3
        self.joint_ranges[0, 11] = -0.1
        self.joint_ranges[1, 11] = 0.3

        #self.n_joints = len(self.joint_ranges[0])

        #self.joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13]

        #self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,))
    
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

    def setup_world(self, initialSimSteps=100):
        super().setup_world(table_height=0.8)

        objects = p.loadSDF(os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/kuka_with_gripper2.sdf"))
        self.robot_id = objects[0]

        p.resetBasePositionAndOrientation(self.robot_id, [-0.1, -0.5, -0.5], [0., 0., 0., 1.])
        self.n_actions = self.n_joints - 3



        if self.obstacle:
            # create the obstacle object at the required location
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=self.obstacle_size)
            viz_id = p.createVisualShape(p.GEOM_SPHERE, radius=self.obstacle_size,
                                         rgbaColor=[0, 0, 0, 1])
            obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
            pos_obstacle = pos
            pos_obstacle[0] += self.obstacle_pos[0]
            pos_obstacle[1] += self.obstacle_pos[1]
            pos_obstacle[2] += self.obstacle_pos[2]

            p.resetBasePositionAndOrientation(obs_id, pos_obstacle, [0, 0, 0, 1])


        self.end_effector_id = 6

    def step(self, action):
        # we want one action per joint (gripper is composed by 4 joints but considered as one)
        assert(len(action) == self.n_actions)
        self.info['closed gripper'] = action[-1]>0
        # add the 3 commands for the 3 last gripper joints

        commands = np.hstack([action[:-1], -action[-1], -action[-1], action[-1], action[-1]])


        # apply the commands
        return super().step(commands)

    
    def reset_robot(self):
        for i, in zip(self.joint_ids,):
            p.resetJointState(self.robot_id, i, targetValue=0)


