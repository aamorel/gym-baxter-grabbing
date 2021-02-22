from time import sleep
import pybullet as p
import qibullet as q
import numpy as np
import gym
import os
import random
from pathlib import Path


def setUpWorld(physics_client, obj='cube', random_obj=False, initialSimSteps=100):
    """
    Reset the simulation to the beginning and reload all models.

    Parameters
    ----------
    initialSimSteps : int

    Returns
    -------
    baxterId : int
    endEffectorId : int
    obj_to_grab_id : int
    """
    p.resetSimulation()
    p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    # load plane
    p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)

    pepper = q.PepperVirtual()

    rob_or = p.getQuaternionFromEuler([0, 0, 1.57])
    pepper.loadRobot(translation=[0.2, -0.38, -0.8],
                     quaternion=rob_or,
                     physicsClientId=physics_client)  # experimentation

    pepper_id = pepper.getRobotModel()

    # table robot part shapes
    t_body = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.7, 0.7, 0.1])
    t_body_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.7, 0.7, 0.1], rgbaColor=[0.3, 0.3, 0, 1])

    t_legs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.4])
    t_legs_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.4], rgbaColor=[0.3, 0.3, 0, 1])

    body_Mass = 500
    visualShapeId = t_body_v
    link_Masses = [30, 30, 30, 30]

    linkCollisionShapeIndices = [t_legs] * 4

    nlnk = len(link_Masses)
    linkVisualShapeIndices = [t_legs_v] * nlnk
    # link positions wrt the link they are attached to

    linkPositions = [[0.35, 0.35, -0.3], [-0.35, 0.35, -0.3], [0.35, -0.35, -0.3], [-0.35, -0.35, -0.3]]

    linkOrientations = [[0, 0, 0, 1]] * nlnk
    linkInertialFramePositions = [[0, 0, 0]] * nlnk
    # note the orientations are given in quaternions (4 params). There are function to convert of Euler angles and back
    linkInertialFrameOrientations = [[0, 0, 0, 1]] * nlnk
    # indices determine for each link which other link it is attached to
    indices = [0] * nlnk
    # most joint are revolving. The prismatic joints are kept fixed for now
    jointTypes = [p.JOINT_FIXED] * nlnk
    # revolution axis for each revolving joint
    axis = [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]

    # drop the body in the scene at the following body coordinates
    basePosition = [0, 0.4, 0]
    baseOrientation = [0, 0, 0, 1]
    # main function that creates the table
    p.createMultiBody(body_Mass, t_body, visualShapeId, basePosition, baseOrientation,
                      linkMasses=link_Masses,
                      linkCollisionShapeIndices=linkCollisionShapeIndices,
                      linkVisualShapeIndices=linkVisualShapeIndices,
                      linkPositions=linkPositions,
                      linkOrientations=linkOrientations,
                      linkInertialFramePositions=linkInertialFramePositions,
                      linkInertialFrameOrientations=linkInertialFrameOrientations,
                      linkParentIndices=indices,
                      linkJointTypes=jointTypes,
                      linkJointAxis=axis)

    # set gravity
    p.setGravity(0., 0., -9.81)

    # let the world run for a bit
    for _ in range(initialSimSteps):
        p.stepSimulation()

    # create object to grab
    if obj == 'cube':
        square_base = 0.02
        height = 0.02
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[square_base, square_base, height])
        viz_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[square_base, square_base, height], rgbaColor=[1, 0, 0, 1])
        obj_to_grab_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
        pos = [0, -0.2, -0.15]
        if random_obj:
            pos[0] = pos[0] + random.gauss(0, 0.01)
            pos[1] = pos[1] + random.gauss(0, 0.01)
        p.resetBasePositionAndOrientation(obj_to_grab_id, pos, [0, 0, 0, 1])
    if obj == 'cup':
        path = os.path.join(Path(__file__).parent, "cup_urdf.urdf")
        cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 1.57])
        pos = [0, -0.15, -0.05]
        if random_obj:
            pos[0] = pos[0] + random.gauss(0, 0.01)
            pos[1] = pos[1] + random.gauss(0, 0.01)
        obj_to_grab_id = p.loadURDF(path, pos, cubeStartOrientation, globalScaling=0.7)
    
    # change friction  of object
    p.changeDynamics(obj_to_grab_id, -1, lateralFriction=1)

    return pepper_id, obj_to_grab_id, pepper


def getJointRanges(bodyId, joints_id):
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

    # loop through all joints
    for i in joints_id:
        jointInfo = p.getJointInfo(bodyId, i)

        ll, ul = jointInfo[8:10]
        jr = ul - ll

        # For simplicity, assume resting state == initial state
        rp = p.getJointState(bodyId, i)[0]

        lowerLimits.append(ll)
        upperLimits.append(ul)
        jointRanges.append(jr)
        restPoses.append(rp)

    return lowerLimits, upperLimits, jointRanges, restPoses


class PepperGrasping(gym.Env):

    def __init__(self, display=False, obj='cube', random_obj=False):

        self.display = display
        if self.display:
            physics_client = p.connect(p.GUI)
        else:
            physics_client = p.connect(p.DIRECT)
        self.obj = obj

        # set up the world, endEffector is the tip of the left finger
        self.robot_id, self.objectId, self.pepper = setUpWorld(physics_client, obj=self.obj, random_obj=random_obj)

        # self.savefile = 'save_state.bullet'
        self.savestate = p.saveState()

        if self.display:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(2., 180, 0., [0.52, 0.2, np.pi / 4.])
            p.getCameraImage(320, 200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            sleep(1.)

        self.steps_to_roll = 10

        # self.joints = ['HipRoll',
        #                'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll',
        #                'LWristYaw', 'LFinger21', 'LFinger22', 'LFinger23',
        #                'LFinger11', 'LFinger12', 'LFinger13', 'LFinger41', 'LFinger42',
        #                'LFinger43', 'LFinger31', 'LFinger32', 'LFinger33', 'LThumb1', 'LThumb2']

        self.joints = ['HipRoll',
                       'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll',
                       'LWristYaw', 'LHand']
        all_joints = self.pepper.joint_dict
        interesting_joints_idx = [all_joints[joint].getIndex() for joint in self.joints]

        self.endEffectorId = all_joints['LHand'].getIndex()
        self.joint_ranges = getJointRanges(self.robot_id, interesting_joints_idx)
        self.n_joints = len(self.joints)
        self.speeds = [1] * self.n_joints

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,))

    def set_steps_to_roll(self, steps_to_roll):
        self.steps_to_roll = steps_to_roll
            
    def step(self, action):
        """Executes one step of the simulation

        Args:
            action (list): between -1 and 1

        Returns:
            list: observation
            float: reward
            bool: done
            dict: info
        """
        # we want one action per joint
        assert(len(action) == self.n_joints)

        # map the action
        commands = []
        for i, joint_command in enumerate(action):
            percentage_command = (joint_command + 1) / 2  # between 0 and 1
            low = self.joint_ranges[0][i]
            high = self.joint_ranges[1][i]
            command = low + percentage_command * (high - low)
            commands.append(command)

        # apply the commands
        self.pepper.setAngles(self.joints, commands, self.speeds)

        # roll the world (motor control doesn't have to be done every loop)
        for _ in range(self.steps_to_roll):
            p.stepSimulation()

        # get information on target object
        obj = p.getBasePositionAndOrientation(self.objectId)
        obj_pos = list(obj[0])  # x, y, z
        # obj_orientation = p.getEulerFromQuaternion(list(obj[1]))
        obj_orientation = list(obj[1])

        # get information on gripper
        grip = p.getLinkState(self.robot_id, self.endEffectorId)
        grip_pos = list(grip[0])  # x, y, z
        grip_orientation = list(grip[1])

        jointPoses = self.pepper.getAnglesPosition(self.joints)

        observation = [obj_pos, obj_orientation, grip_pos, grip_orientation, jointPoses]

        contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.objectId)
        reward = None
        info = {}
        info['contact_points'] = contact_points
        info['closed gripper'] = True
        if jointPoses[-1] > 0.2:
            info['closed gripper'] = False
        done = False
        return observation, reward, done, info

    def reset(self):
        p.restoreState(self.savestate)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
