import gym
import pybullet as p
from time import sleep
import pybullet_data
#from pybullet_utils import bullet_client
from pathlib import Path
import weakref
import functools
import inspect
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple, Sequence, Callable, Any, Union, Optional

class BulletClient(object):
  """A wrapper for pybullet to manage different clients. copy-pasted from https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_utils/bullet_client.py"""

  def __init__(self, connection_mode=None, hostName=None, options=None):
    self._shapes = {}
    if connection_mode is None:
      self._client = p.connect(p.SHARED_MEMORY, options=options) if options else p.connect(p.SHARED_MEMORY)
      if self._client >= 0:
        return
      else:
        connection_mode = p.DIRECT
    if hostName is None:
        self._client = p.connect(connection_mode, options=options) if options else p.connect(connection_mode)
    else:
        self._client = p.connect(connection_mode, hostName=hostName, options=options) if options else p.connect(connection_mode, hostName=hostName)

  def __getattr__(self, name):
    """Inject the client id into Bullet functions."""
    attribute = getattr(p, name)
    if inspect.isbuiltin(attribute):
      attribute = functools.partial(attribute, physicsClientId=self._client)
    if name=="disconnect":
      self._client = -1
    return attribute


class RobotGrasping(gym.Env):

    def __init_subclass__(cls, *args, **kwargs): # callback trick
        super().__init_subclass__(*args, **kwargs)
        #oldinit = cls.__init__
        #def newinit(self, *args, **kwargs): # this function wraps __init__ in subclasses
            #oldinit(self, *args, **kwargs)
        #cls.__init__ = newinit

    def __init__(self,
        robot: Callable[[], int], # function to load the robot, returning the id
        display: bool = False, # enable/disable display
        obj: str = 'cube', # object to load, available objects are those defined in get_object and in the folder gym-baxter-grabbing/gym_baxter_grabbing/envs/objects/
        pos_cam: npt.ArrayLike = [1.3, 180, -40], # initial positon of the camera
        gripper_display: bool = False, # display the ripper frame
        steps_to_roll: int = 1, # nb time to call p.stepSimulation within one step
        random_var: Optional[float] = 0, # the variance of the positon noise of the object
        delta_pos: npt.ArrayLike = [0, 0], # position displacement of the object
        initial_position_object: npt.ArrayLike = [0., 0., 0.], # initial position of te object during loading, then it will fall
        table_height: Optional[float] = 0.7, # the height of the table
        mode: str = 'joint positions', # the control mode, either 'joint positions', 'joint velocities', 'joint torques', 'inverse kinematic'
        end_effector_id: int = -1, # link id of the end effector
        joint_ids: Optional[npt.ArrayLike] = None, # array of int, ids of joints to control
        n_actions: int = 1, # nb of dimensions of the action space
        center_workspace: Union[int, npt.ArrayLike] = -1, # position of the center of the sphere, supposing the workspace is a sphere (robotic arm) and the robot is not moving, if int, the position of the robot link is used
        radius: float = 1, # radius of the workspace
        contact_ids: npt.ArrayLike = [], # link id (int) of the robot gripper that can have a grasping contact
        disable_collision_pair: npt.ArrayLike = [], # 2D array (-1,2), list of pair of link id (int) to disable collision
        change_dynamics: Dict[int, Dict[str, Any]] = {} # the key is the robot link id, the value is the args passed to p.changeDynamics
    ):
        weakref.finalize(self, self.close) # cleanup
        self.obj = obj.strip()
        self.initial_position_object = initial_position_object
        self.display = display
        self.pos_cam = pos_cam
        self.gripper_display = gripper_display
        self.steps_to_roll = steps_to_roll
        self.random_var = random_var
        self.delta_pos = delta_pos
        self.has_reset_object = False
        self.p = BulletClient(connection_mode=p.GUI if display else p.DIRECT)
        self.physicsClientId = self.p._client
        self.end_effector_id = end_effector_id
        self.mode = mode
        self.n_actions = n_actions
        self.radius = radius
        self.contact_ids = contact_ids

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        #self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        #self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)

        self.plane_id = self.p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True) # load plane with an offset

        # table is about 62.5cm tall and the z position of the table is located at the very bottom, there is link with the ground (so it is static)
        self.table_id = None if table_height is None else self.p.loadURDF("table/table.urdf", basePosition=[0, 0.4, -1+(table_height-0.625)], baseOrientation=[0,0,0,1], useFixedBase=False)

        self.p.setGravity(0., 0., -9.81) # set gravity
        
        self.load_object(self.get_object(self.obj), delta_pos=self.delta_pos)
        self.robot_id = robot()
        self.joint_ids = np.array([i for i in range(self.p.getNumJoints(self.robot_id)) if self.p.getJointInfo(self.robot_id, i)[3]>-1] if joint_ids is None else joint_ids, dtype=int)
        self.n_joints = 8 if mode=='inverse kinematic' else len(self.joint_ids)
        self.reset_robot()
        
        self.center_workspace = np.array(self.p.getLinkState(self.robot_id, center_workspace)[0] if isinstance(center_workspace, int) else center_workspace)
        self.center_workspace = self.p.multiplyTransforms(*self.p.invertTransform(*p.getBasePositionAndOrientation(self.robot_id)), self.center_workspace, [1,0,0,0]) # the pose of center_workspace in the robot frame
        for contact_point in disable_collision_pair:
            self.p.setCollisionFilterPair(self.robot_id, self.robot_id, contact_point[0], contact_point[1], enableCollision=0)
        
        self.lowerLimits, self.upperLimits, self.maxForce, self.maxVelocity = np.zeros(self.n_joints), np.zeros(self.n_joints), np.zeros(self.n_joints), np.zeros(self.n_joints)
        for i, id in enumerate(self.joint_ids):
            self.lowerLimits[i], self.upperLimits[i], self.maxForce[i], self.maxVelocity[i] = self.p.getJointInfo(self.robot_id, id)[8:12]
        
        for id, args in change_dynamics.items(): # change dynamics
            self.p.changeDynamics(self.robot_id, linkIndex=id, **args)
            if id in self.joint_ids: # update limits if needed
                index = np.nonzero(self.joint_ids==id)[0][0]
                if 'jointLowerLimit' in args and 'jointUpperLimit' in args:
                    self.lowerLimits[index] = args['jointLowerLimit']
                    self.upperLimits[index] = args['jointUpperLimit']
                if 'maxJointVelocity' in args:
                    self.maxVelocity[index] = args['maxJointVelocity']
                if 'jointLimitForce' in args:
                    self.maxForce[index] = args['jointLimitForce']
        self.maxForce = np.where(self.maxForce<=0, 100, self.maxForce)# replace bad values
        self.maxVelocity = np.where(self.maxVelocity<=0, 1, self.maxVelocity)
        
        self.jointRanges = self.upperLimits-self.lowerLimits
        self.restPoses = [s[0] for s in self.p.getJointStates(self.robot_id, self.joint_ids)]
        
        if self.display: # set the camera
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)
            self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
            self.p.resetDebugVisualizerCamera(self.pos_cam[0], self.pos_cam[1], self.pos_cam[2], [0, 0, 0])
            self.p.getCameraImage(320, 200, renderer=self.p.ER_BULLET_HARDWARE_OPENGL)

            if self.gripper_display:
                self.line_width = 4
                self.lines = [self.p.addUserDebugLine([0, 0, 0], end, color, lineWidth=self.line_width,
                                                 parentObjectUniqueId=self.robot_id,
                                                 parentLinkIndex=self.end_effector_id)
                                                 for end, color in zip(np.eye(3)*0.2, np.eye(3))]
        # let the world run for a bit
        for _ in range(100): self.p.stepSimulation()
        
        
        self.save_state = self.p.saveState()
        self.action_space = gym.spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        self.observation_space = gym.spaces.Box(-1., 1., shape=(13+len(self.joint_ids)*2,), dtype='float32') # TODO: add image if asked
        self.info = {'contact object table':[], 'applied joint motor torques':np.zeros(self.n_joints), 'joint positions':np.zeros(self.n_joints), 'joint velocities':np.zeros(self.n_joints)}


    def step(self, action=None): # actions are in [-1,1]
        if self.mode in {'joint positions', 'inverse kinematic'} and action is not None:
            for id, a, v ,f, u, l in zip(self.joint_ids, action, self.maxVelocity, self.maxForce, self.upperLimits, self.lowerLimits):
                self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=id, controlMode=self.p.POSITION_CONTROL, targetPosition=l+(a+1)/2*(u-l), maxVelocity=v, force=f)
        elif self.mode in {'joint torques'} and action is not None: # much harder and might not be transferable because requires very accurate URDF, center of mass, frictions...
            self.p.setJointMotorControlArray(bodyIndex=self.robot_id, jointIndices=self.joint_ids, controlMode=self.p.TORQUE_CONTROL, forces=action*self.maxForce)
        elif self.mode in {'joint velocities'} and action is not None:
            self.p.setJointMotorControlArray(bodyIndex=self.robot_id, jointIndices=self.joint_ids, controlMode=self.p.VELOCITY_CONTROL, forces=self.maxForce, targetVelocities=actions*self.maxVelocity)

        # roll the world (motor control doesn't have to be done every loop)
        for _ in range(self.steps_to_roll): self.p.stepSimulation()

        # get information on gripper
        self.info['end effector position'], self.info['end effector xyzw'] = self.p.getLinkState(self.robot_id, self.end_effector_id)[:2]
        
        self.info['contact object robot'] = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.robot_id)
        self.info['contact object plane'] = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.plane_id)
        self.info['contact object table'] = self.p.getContactPoints(bodyA=self.obj_id, bodyB=self.table_id) if self.table_id is not None else []
        self.info['contact robot robot'] = self.p.getContactPoints(bodyA=self.robot_id,bodyB=self.robot_id)
        self.info['touch'] = False
        for c in self.info['contact object robot']:
            if c[4] in self.contact_ids:
                self.info['touch'] = True
                break
        #self.compute_self_contact()

        if self.display and self.gripper_display:
            self.lines = [self.p.addUserDebugLine([0, 0, 0], end, color, lineWidth=self.line_width,
                                             replaceItemUniqueId=id,
                                             parentObjectUniqueId=self.robot_id,
                                             parentLinkIndex=self.end_effector_id)
											 for end, color, id in zip(np.eye(3)*0.2, np.eye(3), self.lines)]

        reward = False
        for c in self.info['contact object robot']:
            if c[8]<-0.005: # if contactDistance is negative, there is a penetration, this is bad
                reward = False
                break
            elif c[4] in self.contact_ids: reward = True # the object must touch the gripper
        reward = len(self.info['contact object table'] + self.info['contact object plane'])==0 and reward
        done = False
        # observations are normalized, thus it is not mean to be handled by humans, check info for human readable datas
        return self.get_obs(), reward, done, self.info
    
    def get_object(self, obj=None):
        """ return a dict containing informations of the primitive shape or a str (urdf file) """
        return obj
        
    def load_object(self, obj=None, delta_pos=[0,0]):
        pos = [self.initial_position_object[0]+delta_pos[0],
               self.initial_position_object[1]+delta_pos[1],
               self.initial_position_object[2]]
        if self.random_var:
            pos[0] += random.gauss(0, self.random_var)
            pos[1] += random.gauss(0, self.random_var)

        # create object to grab
        if isinstance(obj, dict):
            if "shape" not in obj.keys(): raise ValueError("'shape' as a key doesn't exist in obj")
            elif obj["shape"] == 'cuboid':
                infoShape =  {"shapeType":self.p.GEOM_BOX, "halfExtents":[obj["x"]/2, obj["y"]/2, obj[z]/2]}
                obj_to_grab_id = self.p.createMultiBody(baseMass=1, baseCollisionShapeIndex=self.p.createCollisionShape(**infoShape), baseVisualShapeIndex=self.p.createVisualShape(**infoShape, rgbaColor=[1, 0, 0, 1]))
                self.p.resetBasePositionAndOrientation(obj_to_grab_id, pos, [0, 0, 0, 1])
                
            elif obj["shape"] == 'cube':
                obj_to_grab_id = self.p.loadURDF("cube_small.urdf", pos, globalScaling=obj["unit"]/0.05) # cube_small is a 5cm 0.1kg cube
            elif obj["shape"] == 'sphere':
                obj_to_grab_id = self.p.loadURDF("sphere_small.urdf", pos, globalScaling=obj["radius"]/0.06) # sphere_small is a 6cm diameter 0.1kg sphere
                self.p.changeDynamics(obj_to_grab_id, -1, rollingFriction=1e-5, spinningFriction=1e-5) # allow the sphere to roll

            elif obj["shape"] == 'cylinder':
                infoShape =  {"shapeType":self.p.GEOM_BOX, "radius": obj["radius"]}
                col_id = self.p.createCollisionShape(**infoShape, height=obj["z"])
                viz_id = self.p.createVisualShape(**infoShape, length=obj["z"], rgbaColor=[1, 0, 0, 1])
                obj_to_grab_id = self.p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
                self.p.resetBasePositionAndOrientation(obj_to_grab_id, pos, [0, 0, 0, 1])
            self.p.changeDynamics(obj_to_grab_id, -1, lateralFriction=0.7)
        elif isinstance(obj, str):
            urdf = Path(__file__).parent/"objects"/obj/f"{obj}.urdf"
            if not urdf.exists(): raise ValueError(str(urdf) + " doesn't exist")
            try:
                obj_to_grab_id = self.p.loadURDF(str(urdf), pos) # the scale is set in the urdf file
            except self.p.error as e:
                raise self.p.error(f"{e}: "+path)
        #elif obj is None:
            #pass # do not load any object
        else:
            raise ValueError("Unrecognized object: "+obj)
            
        #self.p.changeDynamics(obj_to_grab_id, -1, collisionMargin=0.04)
        self.obj_id = obj_to_grab_id

    def reset(self, delta_pos=[0,0], yaw=0): # it is not a transform: rotate and translate
        assert not self.has_reset_object, "you can not remove/change the object and restore a state: use either reset() or reset_object(), not both"
        self.p.restoreState(self.save_state)
        pos, qua = self.p.getBasePositionAndOrientation(self.obj_id)
        pos = [pos[0]+delta_pos[0], pos[1]+delta_pos[1], pos[2]]
        if self.random_var:
            pos[0] += random.gauss(0, self.random_var)
            pos[1] += random.gauss(0, self.random_var)
        _, qua = self.p.multiplyTransforms([0,0,0], [0, 0, np.sin(yaw/2), np.cos(yaw/2)], [0,0,0], qua) # apply yaw rotation
        self.p.resetBasePositionAndOrientation(self.obj_id, pos, qua)
        return np.maximum(np.minimum(self.get_obs(),1),-1)
        
    def get_obs(self):
        obj_pose = self.p.getBasePositionAndOrientation(self.obj_id)
        # we do not normalize the velocity, supposing the object is not moving that fast
        # we do not express the velocity in the robot frame, supoosing the robot is not moving
        obj_vel = self.p.getBaseVelocity(self.obj_id)
        self.info['object position'], self.info['object xyzw'] = obj_pose
        self.info['object linear velocity'], self.info['object angular velocity'] = obj_vel
        jointStates = self.p.getJointStates(self.robot_id, self.joint_ids)
        pos, vel = [0]*self.n_joints, [0]*self.n_joints
        for i, state, u, l, v in zip(range(self.n_joints), jointStates, self.upperLimits, self.lowerLimits, self.maxVelocity):
            pos[i] = 2*(state[0]-u)/(u-l) + 1 # set between -1 and 1
            vel[i] = state[1]/v # set between -1 and 1
            self.info['joint positions'][i], self.info['joint velocities'][i], _, self.info['applied joint motor torques'][i] = state
        absolute_center = p.multiplyTransforms(*self.p.getBasePositionAndOrientation(self.robot_id), *self.center_workspace) # the pose of center_workspace in the world
        obj_pos, obj_or = self.p.multiplyTransforms(*self.p.invertTransform(*absolute_center), *obj_pose) # the object pose in the center_workspace frame
        observation = np.hstack([np.array(obj_pos)/self.radius, obj_or, *obj_vel, pos, vel])
        return observation#np.maximum(np.minimum(obs,1),-1)

            
    def reset_object(self, obj=None, delta_pos=[0,0]): # TODO: delete, useless
        if obj == self.obj and not self.has_reset_object:
            self.reset(delta_pos=delta_pos)
        else:
            self.has_reset_object = True
            self.reset_robot()
            self.p.removeBody(self.obj_id)
            self.load_object(self.get_object(obj), delta_pos=delta_pos)
            for _ in range(100): self.p.stepSimulation() # let the object fall
    
    def reset_robot(self):
        pass
            

    def render(self, mode='human'):
        pass

    def close(self):
        if self.physicsClientId >=0:
            self.p.disconnect()
            self.physicsClientId = -1
        
    def get_state(self):
        return None
