from time import sleep
import pybullet as p
import numpy as np
import gym
from scipy.interpolate import interp1d

MAX_FORCE = 100


def setUpWorld(initialSimSteps=100):
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
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

    # load plane
    p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)

    # load Baxter
    # urdf_flags = p.URDF_USE_SELF_COLLISION    makes the simulation go crazys
    baxterId = p.loadURDF("baxter_common/baxter_description/urdf/toms_baxter.urdf", useFixedBase=True)

    p.resetBasePositionAndOrientation(baxterId, [0, -0.8, 0.0], [0., 0., -1., -1.])

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
                       
    # create object to grab
    square_base = 0.02
    height = 0.08
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[square_base, square_base, height])
    viz_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[square_base, square_base, 0.1], rgbaColor=[1, 0, 0, 1])
    obj_to_grab_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
    p.resetBasePositionAndOrientation(obj_to_grab_id, [0, 0.2, 0.73], [0, 0, 0, 1])

    # grab relevant joint IDs
    endEffectorId = 48  # (left gripper left finger)

    # set gravity
    p.setGravity(0., 0., -9.81)

    # let the world run for a bit
    for _ in range(initialSimSteps):
        p.stepSimulation()

    return baxterId, endEffectorId, obj_to_grab_id


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


def accurateIK(bodyId, endEffectorId, targetPosition, lowerLimits, upperLimits, jointRanges, restPoses,
               useNullSpace=False, maxIter=10, threshold=1e-4):
    """
    Parameters
    ----------
    bodyId : int
    endEffectorId : int
    targetPosition : [float, float, float]
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
        jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition,
                                                  lowerLimits=lowerLimits, upperLimits=upperLimits,
                                                  jointRanges=jointRanges,
                                                  restPoses=restPoses)
    else:
        jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition)

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
                                    targetPosition=jointPoses[qIndex - 7], force=MAX_FORCE)


class Baxter_grabbingEnv(gym.Env):

    def __init__(self, display=False):

        self.display = display
        if self.display:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # set up the world, endEffector is the tip of the left finger
        self.baxterId, self.endEffectorId, self.objectId = setUpWorld()

        self.lowerLimits, self.upperLimits, self.jointRanges, self.restPoses = getJointRanges(self.baxterId,
                                                                                              includeFixed=False)
        if self.display:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.resetDebugVisualizerCamera(2., 180, 0., [0.52, 0.2, np.pi / 4.])
            p.getCameraImage(320, 200, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            sleep(1.)
        
        self.interp_grip = interp1d([-1, 1], [0, 0.020833], bounds_error=False, fill_value='extrapolate')
            
    def step(self, action):
        """Executes one step of the simulation

        Args:
            action (list): size 4, target position of end effector in Cartesian coordinate
                           and target position of gripper

        Returns:
            list: observation
            float: reward
            bool: done
            dict: info
        """
        targetPosition = action
        p.stepSimulation()

        jointPoses = accurateIK(self.baxterId, self.endEffectorId, targetPosition[0:3], self.lowerLimits,
                                self.upperLimits, self.jointRanges, self.restPoses, useNullSpace=True)
        setMotors(self.baxterId, jointPoses)

        # explicitly control the gripper
        target_gripper_pos = float(self.interp_grip(targetPosition[3]))
        p.setJointMotorControl2(bodyIndex=self.baxterId, jointIndex=49, controlMode=p.POSITION_CONTROL,
                                targetPosition=target_gripper_pos, force=MAX_FORCE)
        p.setJointMotorControl2(bodyIndex=self.baxterId, jointIndex=51, controlMode=p.POSITION_CONTROL,
                                targetPosition=-target_gripper_pos, force=MAX_FORCE)

        # get information on target object
        obj = p.getBasePositionAndOrientation(self.objectId)
        obj_pos = list(obj[0])  # x, y, z
        # obj_orientation = p.getEulerFromQuaternion(list(obj[1]))
        obj_orientation = list(obj[1])

        # get information on gripper
        grip = p.getLinkState(self.baxterId, self.endEffectorId)
        grip_pos = list(grip[0])  # x, y, z
        # grip_orientation = p.getEulerFromQuaternion(list(grip[1]))
        grip_orientation = list(grip[1])

        observation = [obj_pos, obj_orientation, grip_pos, grip_orientation]
        reward = None
        info = None
        done = False
        return observation, reward, done, info

    def reset(self):
        self.baxterId, self.endEffectorId, self.objectId = setUpWorld()

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
