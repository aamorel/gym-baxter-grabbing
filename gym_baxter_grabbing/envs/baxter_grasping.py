import pybullet as p
import numpy as np
import os
import random
from pathlib import Path
import pyquaternion as pyq
import gym_baxter_grabbing.envs.robot_grasping as robot_grasping
import json

MAX_FORCE = 100


def getJointRanges(bodyId, includeFixed=False):
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


def accurateIK(bodyId, end_effector_id, targetPosition, targetOrientation, lowerLimits, upperLimits, jointRanges,
               restPoses,
               useNullSpace=False, maxIter=10, threshold=1e-4):
    """
    Parameters
    ----------
    bodyId : int
    end_effector_id : int
    targetPosition : [float, float, float]
    targetOrientation: Quaternion
    lowerLimits : [float]
    upperLimits : [float]
    jointRanges : [float]
    restPoses : [float]
    useNullSpace : bool
    maxIter : int
    threshold : float

    Returns
    -------
    jointPoses : [float] * numDofs
    """

    if useNullSpace:
        jointPoses = p.calculateInverseKinematics(bodyId, end_effector_id, targetPosition,
                                                  targetOrientation=targetOrientation,
                                                  lowerLimits=lowerLimits, upperLimits=upperLimits,
                                                  jointRanges=jointRanges,
                                                  restPoses=restPoses)
    else:
        jointPoses = p.calculateInverseKinematics(bodyId, end_effector_id, targetPosition,
                                                  targetOrientation=targetOrientation)

    return jointPoses


def setMotors(bodyId, jointPoses):
    """
    Parameters
    ----------
    bodyId : int
    jointPoses : [float] * numDofs
    """
    numJoints = p.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)
        qIndex = jointInfo[3]
        if qIndex > -1:
            p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[qIndex - 7], force=MAX_FORCE, maxVelocity=0.5)


def setMotorsIds(bodyId, joint_ids, jointPoses):
    """
    Parameters
    ----------
    bodyId : int
    jointPoses : [float] * numDofs
    """

    for i, id in enumerate(joint_ids):

        p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=id, controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i], force=MAX_FORCE, maxVelocity=0.5)


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
        if includeFixed or jointInfo[3] > -1:
            # print(jointInfo[0], jointInfo[1], jointInfo[2], jointInfo[3], jointInfo[8:10])
            # jointInfo[3] > -1 means that the joint is not fixed
            joint_state = p.getJointState(bodyId, i)[0]
            states.append(joint_state)

    return states


class BaxterGrasping(robot_grasping.RobotGrasping):

    def __init__(self, display=False, obj='cube', random_obj=False, steps_to_roll=1, mode='joints_space', y_pose=0.12):
        self.y_pos = y_pose
        self.mode = mode
        super().__init__(display=display, obj=obj, random_obj=random_obj, pos_cam=[1.2, 180, -40],
                         gripper_display=True, steps_to_roll=steps_to_roll)
        self.lowerLimits, self.upperLimits, self.jointRanges, self.restPoses = getJointRanges(self.robot_id,
                                                                                              includeFixed=False)
        # much simpler and faster (we want a linear function)
        self.interp_grip = lambda a: (a + 1) * 0.010416

        self.steps_to_roll = 10

        if mode == 'joints_space':
            self.joints_id = [34, 35, 36, 37, 38, 40, 41, 49, 51]
            self.ids_in_ranges = [10, 11, 12, 13, 14, 15, 16, 17, 18]
            self.n_joints = len(self.joints_id)

    def setup_world(self, initialSimSteps=100):

        p.resetSimulation()
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # load plane
        p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)

        # load Baxter
        urdf_flags = p.URDF_USE_SELF_COLLISION   # makes the simulation go crazys
        robot_id = p.loadURDF("baxter_common/baxter_description/urdf/toms_baxter.urdf",
                              useFixedBase=False, flags=urdf_flags)
        p.resetBasePositionAndOrientation(robot_id, [0, -0.8, 0.0], [0., 0., -1., -1.])

        path = os.path.join(Path(__file__).parent, "contact_points_baxter.txt")
        with open(path) as json_file:
            data = json.load(json_file)

        for contact_point in data:
            p.setCollisionFilterPair(robot_id, robot_id, contact_point[0], contact_point[1], 0)

        # table robot part shapes
        t_body = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.7, 0.7, 0.025])
        t_body_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.7, 0.7, 0.025], rgbaColor=[0.3, 0.3, 0, 1])

        t_legs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.35])
        t_legs_v = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.35], rgbaColor=[0.3, 0.3, 0, 1])

        body_Mass = 500
        visualShapeId = t_body_v
        link_Masses = [30, 30, 30, 30]

        linkCollisionShapeIndices = [t_legs] * 4

        nlnk = len(link_Masses)
        linkVisualShapeIndices = [t_legs_v] * nlnk
        # link positions wrt the link they are attached to

        linkPositions = [[0.35, 0.35, -0.375], [-0.35, 0.35, -0.375], [0.35, -0.35, -0.375], [-0.35, -0.35, -0.375]]

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

        # grab relevant joint IDs
        end_effector_id = 48  # (left gripper left finger)

        # set gravity
        p.setGravity(0., 0., -9.81)

        # let the world run for a bit
        for _ in range(initialSimSteps):
            p.stepSimulation()

        # create object to grab
        if self.obj == 'cube':
            w = 0.023
            le = 0.05
            height = 0.108
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[le, w, height])
            viz_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[le, w, height], rgbaColor=[1, 0, 0, 1])
            obj_to_grab_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
            pos = [0, self.y_pos, -0.05]
            if self.random_obj:
                pos[0] = pos[0] + random.gauss(0, 0.01)
                pos[1] = pos[1] + random.gauss(0, 0.01)
            p.resetBasePositionAndOrientation(obj_to_grab_id, pos, [0, 0, 0, 1])
        if self.obj == 'cylinder':
            r = 0.032
            le = 0.15
            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=le)
            viz_id = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=le, rgbaColor=[1, 0, 0, 1])
            obj_to_grab_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
            pos = [0, self.y_pos, -0.1]
            if self.random_obj:
                pos[0] = pos[0] + random.gauss(0, 0.01)
                pos[1] = pos[1] + random.gauss(0, 0.01)
            p.resetBasePositionAndOrientation(obj_to_grab_id, pos, [0, 0, 0, 1])
        if self.obj == 'cup':
            path = os.path.join(Path(__file__).parent, "cup_urdf.urdf")
            cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 1.57])
            pos = [0, self.y_pos, -0.05]
            if self.random_obj:
                pos[0] = pos[0] + random.gauss(0, 0.01)
                pos[1] = pos[1] + random.gauss(0, 0.01)
            obj_to_grab_id = p.loadURDF(path, pos, cubeStartOrientation, globalScaling=2)
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
        self.robot_id = robot_id
        self.end_effector_id = end_effector_id
        self.obj_id = obj_to_grab_id

    def actuate(self):

        if self.mode == 'end_effector_space':
            target_position = self.action[0:3]
            target_orientation = self.action[3:7]
            quat_orientation = pyq.Quaternion(target_orientation)
            quat_orientation = quat_orientation.normalised
            target_gripper = self.action[7]

            jointPoses = accurateIK(self.robot_id, self.end_effector_id, target_position, target_orientation,
                                    self.lowerLimits,
                                    self.upperLimits, self.jointRanges, self.restPoses, useNullSpace=True)
            setMotors(self.robot_id, jointPoses)

            # explicitly control the gripper
            target_gripper_pos = float(self.interp_grip(target_gripper))
            p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=49, controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_gripper_pos, force=MAX_FORCE)
            p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=51, controlMode=p.POSITION_CONTROL,
                                    targetPosition=-target_gripper_pos, force=MAX_FORCE)

        if self.mode == 'joints_space':
            # we want one action per joint (gripper is composed by 2 joints but considered as one)
            assert(len(self.action) == self.n_joints - 1)

            # add the command for the last gripper joint
            for i in range(1):
                self.action = np.append(self.action, self.action[-1])

            # map the action
            commands = []
            for i, joint_command in enumerate(self.action):
                percentage_command = (joint_command + 1) / 2  # between 0 and 1
                if i == 8:
                    percentage_command = 1 - percentage_command
                low = self.lowerLimits[self.ids_in_ranges[i]]
                high = self.upperLimits[self.ids_in_ranges[i]]

                command = low + percentage_command * (high - low)
                commands.append(command)

            # apply the commands
            setMotorsIds(self.robot_id, self.joints_id, commands)

    def compute_joint_poses(self):

        self.joint_poses = getJointStates(self.robot_id)
    
    def compute_self_contact(self):
        
        self_contact_points = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        self.info['self contact_points'] = self_contact_points

    def compute_grip_info(self):
        self.info['closed gripper'] = True
        if self.joint_poses[-2] > 0.0003 or self.joint_poses[-2] < -0.0003:
            self.info['closed gripper'] = False
        if self.joint_poses[-1] > 0.0003 or self.joint_poses[-1] < -0.0003:
            self.info['closed gripper'] = False
