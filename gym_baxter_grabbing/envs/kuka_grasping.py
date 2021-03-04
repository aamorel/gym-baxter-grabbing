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

    def __init__(self, display=False, obj='cube', random_obj=False, steps_to_roll=1):
        super().__init__(display=display, obj=obj, random_obj=random_obj, pos_cam=[1.3, 180, -40],
                         gripper_display=True, steps_to_roll=steps_to_roll)

        self.joint_ranges = getJointRanges2(self.robot_id)

        self.joint_ranges = np.array(self.joint_ranges)

        # change joints ranges for gripper

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

        self.n_joints = len(self.joint_ranges[0])

        self.joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13]

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,))

    def setup_world(self, initialSimSteps=100):
    
        p.resetSimulation()
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # load plane
        p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)

        objects = p.loadSDF(os.path.join(pybullet_data.getDataPath(), "kuka_iiwa/kuka_with_gripper2.sdf"))
        kuka_id = objects[0]

        p.resetBasePositionAndOrientation(kuka_id, [-0.1, -0.5, -0.5], [0., 0., 0., 1.])

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
        # note the orientations are given in quaternions (4 params).
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
        if self.obj == 'cube':
            square_base = 0.03
            height = 0.08
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[square_base, square_base, height])
            viz_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[square_base, square_base, height],
                                         rgbaColor=[1, 0, 0, 1])
            obj_to_grab_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
            pos = [0, 0.1, -0.05]
            if self.random_obj:
                pos[0] = pos[0] + random.gauss(0, 0.01)
                pos[1] = pos[1] + random.gauss(0, 0.01)
            p.resetBasePositionAndOrientation(obj_to_grab_id, pos, [0, 0, 0, 1])
        if self.obj == 'cup':
            path = os.path.join(Path(__file__).parent, "cup_urdf.urdf")
            cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 1.57])
            pos = [0, 0.1, -0.05]
            if self.random_obj:
                pos[0] = pos[0] + random.gauss(0, 0.01)
                pos[1] = pos[1] + random.gauss(0, 0.01)
            obj_to_grab_id = p.loadURDF(path, pos, cubeStartOrientation)

        if self.obj == 'deer':
            path = os.path.join(Path(__file__).parent, "deer_urdf.urdf")
            cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
            pos = [0, 0.1, -0.05]
            if self.random_obj:
                pos[0] = pos[0] + random.gauss(0, 0.01)
                pos[1] = pos[1] + random.gauss(0, 0.01)
            obj_to_grab_id = p.loadURDF(path, pos, cubeStartOrientation)
        
        # change friction  of object
        p.changeDynamics(obj_to_grab_id, -1, lateralFriction=1)

        for _ in range(initialSimSteps):
            p.stepSimulation()

        self.obj_id = obj_to_grab_id
        self.robot_id = kuka_id
        self.end_effector_id = 6

    def actuate(self):
        # we want one action per joint (gripper is composed by 4 joints but considered as one)
        assert(len(self.action) == self.n_joints - 3)

        # add the 3 commands for the 3 last gripper joints
        for i in range(3):
            self.action = np.append(self.action, self.action[-1])

        # map the action
        commands = []
        for i, joint_command in enumerate(self.action):
            percentage_command = (joint_command + 1) / 2  # between 0 and 1
            low = self.joint_ranges[0][i]
            high = self.joint_ranges[1][i]

            command = low + percentage_command * (high - low)
            commands.append(command)

        # apply the commands
        setMotors(self.robot_id, self.joints_id, commands)

    def compute_joint_poses(self):

        self.joint_poses = getJointStates(self.robot_id)

    def compute_grip_info(self):
        self.info['closed gripper'] = True
        if self.joint_poses[8] < -0.1 and self.joint_poses[10] > 0.1:
            self.info['closed gripper'] = False

        if self.joint_poses[9] < 0 and self.joint_poses[11] > 0:
            self.info['closed gripper'] = False
