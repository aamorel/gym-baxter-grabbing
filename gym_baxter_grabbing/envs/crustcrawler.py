import pybullet as p
import numpy as np
import os
import random
from pathlib import Path
import pyquaternion as pyq
from gym_baxter_grabbing.envs.robot_grasping import RobotGrasping
import xml.etree.ElementTree as ET
from gym_baxter_grabbing.envs.xacro import _process




class CrustCrawler(RobotGrasping):

    def __init__(self, display=False, obj='cube', random_obj=False, delta_pos=[0, 0],
                 steps_to_roll=1, mode='joints_space', random_var=0.01,
                 limit_scale=0.3):
        

        self.mode = mode

        if (not (isinstance(limit_scale, int) or isinstance(limit_scale, float))) or limit_scale<0:
            raise NameError(f"The limit_scale value must be a positive scalar")
                    
        cwd = Path(__file__).parent
        urdf = Path(cwd/f"robots/generated/crustcrawler{np.round(limit_scale, 2)}.urdf")
        urdf.parent.mkdir(exist_ok=True)
        if not urdf.is_file(): # create the file if doesn't exist
            _process(cwd/"robots/crustcrawler_description/urdf/crustcrawler_scale.xacro", dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1, mappings={"limit_scale":str(limit_scale)})) # convert xacro to urdf
        self.crustcrawler_urdf_file = str(urdf)
            
        super().__init__(display=display, obj=obj, random_obj=random_obj, pos_cam=[1.2, 180, -40],
                         gripper_display=False, steps_to_roll=steps_to_roll, random_var=random_var, delta_pos=delta_pos)
        
        


    def setup_world(self):
		# create object to grab
        if self.obj == 'cuboid':
            obj = {"shape":'cuboid', "x":0.046, "y":0.1, "z":0.216}
        elif self.obj == 'cube':
            obj = {"shape": 'cube', "unit":0.055}
        elif self.obj == 'sphere':
            obj = {"shape":'sphere', "radius":0.055}
        elif self.obj == 'cylinder':
            obj = {"shape":'cylinder', "radius":0.032, "z":0.15}
        elif self.obj == 'paper roll':
            obj = {"shape":'cylinder', "radius":0.021, "z":0.22}
        else:
            obj = self.obj

        h = 0.76
        super().setup_world(table_height=h, initial_position=[self.delta_pos[0], 0.5+self.delta_pos[1], 0], obj=obj)

        urdf_flags = p.URDF_USE_SELF_COLLISION   # makes the simulation go crazys
        # z offset to make baxter touch the floor z=1 is about -.074830m in pybullet
        robot_id = p.loadURDF(self.crustcrawler_urdf_file, basePosition=[0, 0, h-1+1e-3], baseOrientation=[0,0,-1,-1], useFixedBase=False, flags=urdf_flags)
        self.robot_id = robot_id
        self.end_effector_id = 15
        self.joints_id = [i for i in range(p.getNumJoints(self.robot_id)) if p.getJointInfo(self.robot_id, i)[3]>-1]
        self.n_joints = len(self.joints_id)
        
        lowerLimits, upperLimits = np.array([p.getJointInfo(self.robot_id, i)[8:10] for i in self.joints_id]).T
        self.jointRanges = (upperLimits - lowerLimits).tolist()
        self.lowerLimits, self.upperLimits = lowerLimits.tolist(), upperLimits.tolist()
        self.restPoses = p.getJointStates(self.robot_id, self.joints_id) # the rest pose is the initial state
        self.maxVelocity = [p.getJointInfo(self.robot_id, i)[11] for i in self.joints_id]
        self.maxForce = [p.getJointInfo(self.robot_id, i)[10] for i in self.joints_id]

        for contact_point in [[13,14], [11,14], [11,13]]:
            p.setCollisionFilterPair(robot_id, robot_id, contact_point[0], contact_point[1], enableCollision=0)


        #for i,v in zip([34, 35, 36, 37, 38, 40, 41,  12, 13, 14, 15, 16, 18, 19], [-0.08, -1.0, -1.19, 1.94, 0.67, 1.03, -0.50,  0.08, -1.0,  1.19, 1.94, -0.67, 1.03, 0.50]):
            #p.resetJointState(robot_id, i, targetValue=v) # put baxter in untuck position
            
        #for i in [25, 26, 27, 28, 29, 30, 47, 48, 49, 50, 51, 52]: # add collision margin to the gripper
            #p.changeDynamics(robot_id, i, collisionMargin=0.04)
        #for i in [28, 30, 50, 52]: # add friction to the finger tips
            #p.changeDynamics(robot_id, i, lateralFriction=1)
        
        # set maximum velocity and force, doesn't work
        for i, id in enumerate(self.joints_id):#range(p.getNumJoints(robot_id)):
            #if p.getJointInfo(robot_id, i)[3]>-1:
            p.changeDynamics(robot_id, id, maxJointVelocity=self.maxVelocity[i], jointLimitForce=self.maxForce[i])
        
        

        
    def actuate(self):

        if isinstance(self.action, dict):
            assert {"cartesian", "quaternion", "gripper close"} <= set(self.action.keys())
            target_position = self.action["cartesian"]#self.action[0:3]
            target_orientation = self.action["quaternion"]#self.action[3:7]
            quat_orientation = pyq.Quaternion(target_orientation)
            quat_orientation = quat_orientation.normalised
            self.info['closed gripper'] = closed = self.action["gripper close"] # Bool

            commands = np.zeros(self.n_joints)
            commands[:self.n_joints-2] = np.array(p.calculateInverseKinematics(self.robot_id, self.end_effector_id, target_position, targetOrientation=target_orientation, lowerLimits=self.lowerLimits, upperLimits=self.upperLimits, jointRanges=self.jointRanges, restPoses=self.restPoses))[self.joints_id[-3:]]
            commands[-3:] = [self.lowerLimits[-2], self.upperLimits[-1]] if closed else [self.upperLimits[-2], self.lowerLimits[-1]] # add gripper
            commands = commands.tolist()


        elif isinstance(self.action, (list, tuple, np.ndarray)):
            # we want one action per joint (gripper is composed by 2 joints but considered as one)
            assert len(self.action)==self.n_joints-1 #and np.max(self.action)<=1 and np.min(self.action)>=-1
            self.info['closed gripper'] = self.action[-1]>0
            # add the command for the last gripper joint
            self.action = np.append(self.action, -self.action[-1])

            # map the action
            commands = [(u-l)/2*(a-1)+u for a,u,l in zip(self.action, self.upperLimits, self.lowerLimits)]

        else:
            raise ValueError(f"self.action is neither a dict(en effector position) or an array(joint positions)")
        p.setJointMotorControlArray(bodyIndex=self.robot_id, jointIndices=self.joints_id, controlMode=p.POSITION_CONTROL, targetPositions=commands)

    def compute_joint_poses(self):
        self.joint_poses = [state[0] for state in p.getJointStates(self.robot_id, self.joints_id)]
    
    def compute_self_contact(self):
        self.info['contact object table'] = p.getContactPoints(bodyA=self.obj_id, bodyB=self.table_id)
        self.info['self contact_points'] = p.getContactPoints(bodyA=self.robot_id, bodyB=self.robot_id)
        self.info['contact robot table'] =  p.getContactPoints(bodyA=self.robot_id, bodyB=self.table_id)
        if len(self.info['self contact_points'])>0:
            print(self.info['self contact_points'])

    def compute_grip_info(self):
        pass


    def get_action(self):
        positions = [0]*(self.n_joints-1)
        for i,j in enumerate(self.joints_id[:-1]):
            pos = p.getJointState(self.robot_id, j)[0]
            low = self.lowerLimits[i]
            high = self.upperLimits[i]
            positions[i] = 2*(pos-high)/(high-low) + 1
        return positions

if __name__ == "__main__": # testing
    env = CrustCrawler(display=True)
    env.step([0,0,0,0,0,0,1])
    while True:
        p.stepSimulation()#o, r, eo, inf = env.step(None)
	
