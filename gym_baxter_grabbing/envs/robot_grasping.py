import gym
import pybullet as p
from time import sleep
import pybullet_data
from pathlib import Path


class RobotGrasping(gym.Env):

    def __init__(self, display=False, obj='cube', random_obj=False, pos_cam=[1.3, 180, -40],
                 gripper_display=False, steps_to_roll=1, random_var=0.01, delta_pos=[0, 0]):
        assert isinstance(obj, str), "obj must be a str"
        self.obj = obj.strip()
        self.display = display
        self.random_obj = random_obj
        self.pos_cam = pos_cam
        self.gripper_display = gripper_display
        self.steps_to_roll = steps_to_roll
        self.random_var = random_var
        self.delta_pos = delta_pos
        if self.display:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
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

    def setup_world(self, table_height=None, initial_position=[0, 0, 0], obj=None):
        p.resetSimulation()
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        # load plane with an offset
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=True)

        if table_height is not None:
        # table is about 62.5cm tall and the z position of the table is located at the very bottom, I don't know why it floats
            self.table_id = p.loadURDF("table/table.urdf", basePosition=[0, 0.4, -1+(table_height-0.625)], baseOrientation=[0,0,0,1], useFixedBase=False)

        # set gravity
        p.setGravity(0., 0., -9.81)
            
        pos = initial_position
        if self.random_obj:
            pos[0] = pos[0] + random.gauss(0, self.random_var)
            pos[1] = pos[1] + random.gauss(0, self.random_var)

        # create object to grab
        if isinstance(obj, dict):
            if "shape" not in obj.keys(): raise ValueError("'shape' as a key doesn't exist in obj")
            elif obj["shape"] == 'cuboid':
                infoShape =  {"shapeType":p.GEOM_BOX, "halfExtents":[obj["x"]/2, obj["y"]/2, obj[z]/2]}
                obj_to_grab_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=p.createCollisionShape(**infoShape), baseVisualShapeIndex=p.createVisualShape(**infoShape, rgbaColor=[1, 0, 0, 1]))
                p.resetBasePositionAndOrientation(obj_to_grab_id, pos, [0, 0, 0, 1])
                
            elif obj["shape"] == 'cube':
                obj_to_grab_id = p.loadURDF("cube_small.urdf", pos, globalScaling=obj["unit"]/0.05) # cube_small is a 5cm cube
            elif obj["shape"] == 'sphere':
                obj_to_grab_id = p.loadURDF("sphere_small.urdf", pos, globalScaling=obj["radius"]/0.06) # sphere_small is a 6cm diameter sphere
                p.changeDynamics(obj_to_grab_id, -1, rollingFriction=1e-6, spinningFriction=1e-6) # allow the sphere to roll

            elif obj["shape"] == 'cylinder':
                infoShape =  {"shapeType":p.GEOM_BOX, "radius": obj["radius"]}
                col_id = p.createCollisionShape(**infoShape, height=obj["z"])
                viz_id = p.createVisualShape(**infoShape, length=obj["z"], rgbaColor=[1, 0, 0, 1])
                obj_to_grab_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=viz_id)
                p.resetBasePositionAndOrientation(obj_to_grab_id, pos, [0, 0, 0, 1])
            p.changeDynamics(obj_to_grab_id, -1, lateralFriction=0.7)
        elif isinstance(obj, str):
            urdf = Path(__file__).parent/"objects"/obj/f"{obj}.urdf"
            if not urdf.exists(): raise ValueError(str(urdf) + " doesn't exist")
            try:
                obj_to_grab_id = p.loadURDF(str(urdf), pos) # the scale is set in the urdf file
            except p.error as e:
                raise p.error(f"{e}: "+path)
        elif obj is None:
            pass # do not load any object
        else:
            raise ValueError("Unrecognized object: "+self.obj)
            
        p.changeDynamics(obj_to_grab_id, -1, collisionMargin=0.04)
        self.obj_id = obj_to_grab_id
        
        # let the world run for a bit
        for _ in range(240):
            p.stepSimulation()

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
        
        self.info['contact_points'] = p.getContactPoints(bodyA=self.robot_id, bodyB=self.obj_id)
        self.info['contact object plane'] = p.getContactPoints(bodyA=self.obj_id, bodyB=self.plane_id)
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

    def reset(self, delta_pos=[0,0]):
        p.restoreState(self.save_state)
        pos, qua = p.getBasePositionAndOrientation(self.obj_id)
        pos = [pos[0]+delta_pos[0], pos[1]+delta_pos[1], pos[2]]
        if self.random_obj:
            pos[0] += random.gauss(0, self.random_var)
            pos[1] += random.gauss(0, self.random_var)
        p.resetBasePositionAndOrientation(self.obj_id, pos, qua)

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
    def get_action(self):
        return None
