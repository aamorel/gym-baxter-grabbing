import numpy as np
import os
import random
from pathlib import Path
import pyquaternion as pyq
from gym_baxter_grabbing.envs.robot_grasping import RobotGrasping
import xml.etree.ElementTree as ET
from gym_baxter_grabbing.envs.xacro import _process




class CrustCrawler(RobotGrasping):

    def __init__(self,
        display=False,
        obj='cube',
        random_obj=False,
        delta_pos=[0, 0],
        steps_to_roll=1,
        mode='joint positions',
        random_var=0,
        limit_scale=0.3,
        reset_random_initial_state=None,
        object_position=[0, 0.4, 0],
        object_xyzw=[0,0,0,1],
        joint_positions=None
    ):
        
        if (not (isinstance(limit_scale, int) or isinstance(limit_scale, float))) or limit_scale<0:
            raise NameError(f"The limit_scale value must be a positive scalar")
                    
        cwd = Path(__file__).parent
        urdf = Path(cwd/f"robots/generated/crustcrawler{np.round(limit_scale, 2)}.urdf")
        urdf.parent.mkdir(exist_ok=True)
        if not urdf.is_file(): # create the file if doesn't exist
            _process(cwd/"robots/crustcrawler_description/urdf/crustcrawler_scale.xacro", dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1, mappings={"limit_scale":str(limit_scale)})) # convert xacro to urdf
        
        h = 0.76
        super().__init__(
            robot=lambda : self.p.loadURDF(str(urdf), basePosition=[0, 0, h-1+1e-3], baseOrientation=[0,0,-1,-1], useFixedBase=False, flags=self.p.URDF_USE_SELF_COLLISION),
            display=display,
            obj=obj,
            pos_cam=[1.2, 180, -40],
            gripper_display=False,
            steps_to_roll=steps_to_roll,
            random_var=random_var,
            delta_pos=delta_pos,
            reset_random_initial_state=reset_random_initial_object_pose,
            object_position=object_position,
            object_xyzw=object_xyzw,
            joint_positions=joint_positions,
            table_height=h,
            mode=mode,
            end_effector_id=15,
            n_control_gripper=2,
            n_actions=7,
            center_workspace=0,
            radius=0.65,
            contact_ids=[12, 13, 14],
            disable_collision_pair=[[11,14], [11,13]],
            allowed_collision_pair=[[13,14]], # allow the two fingers to collide
            change_dynamics = {i:{'lateralFriction':1} for i in (13, 14)}
        )

        
    def get_object(self, obj=None):
        if obj == 'cuboid':
            return {"shape":'cuboid', "x":0.046, "y":0.1, "z":0.216}
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

        if self.mode == 'inverse kinematic':
            #assert {"position", "quaternion", "gripper close"} <= set(self.action.keys())
            target_position = action[0:3]
            q = pyq.Quaternion(action[3:7]).normalised # wxyz
            target_orientation = [q[3], q[0], q[1], q[2]] # xyzw
            gripper = action[7] # [-1(open), 1(close)]
            self.info['closed gripper'] = gripper>0

            commands = np.zeros(self.n_joints)
            commands[:-2] = self.p.calculateInverseKinematics(self.robot_id, self.end_effector_id, target_position, targetOrientation=target_orientation)[:-2]
            #commands[:-2] = np.array(self.p.calculateInverseKinematics(self.robot_id, self.end_effector_id, target_position, targetOrientation=target_orientation, lowerLimits=self.lowerLimits, upperLimits=self.upperLimits, jointRanges=self.jointRanges, restPoses=self.restPoses))[self.joints_id[:-2]]
            commands[-2:] = [(u-l)/2*(a-1)+u for a,u,l in zip((gripper, -gripper), self.upperLimits, self.lowerLimits)] # add gripper
            commands = commands.tolist()


        elif self.mode == 'joint positions':
            # we want one action per joint (gripper is composed by 2 joints but considered as one)
            assert len(action)==self.n_actions #and np.max(self.action)<=1 and np.min(self.action)>=-1
            self.info['closed gripper'] = action[-1]<0
            # add the command for the last gripper joint
            commands = np.append(action, -action[-1])
        
        elif self.mode == 'joint torques':
            # control the two fingers in positions
            for id, a, v, f, u, l in zip(self.joint_ids[-2:], (action[-1], -action[-1]), self.maxVelocity[-2:], self.maxForce[-2:], self.upperLimits[-2:], self.lowerLimits[-2:]):
                self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=id, controlMode=self.p.POSITION_CONTROL, targetPosition=l+(a+1)/2*(u-l), maxVelocity=v, force=f)
            commands = action[:-1]

        return super().step(commands)


        
    def reset_robot(self):
        for i, in zip(self.joint_ids):
            self.p.resetJointState(self.robot_id, i, targetValue=0)


    def get_joint_state(self):
        positions = [2*(s[0]-u)/(u-l) + 1 for s,u,l in zip(self.p.getJointStates(self.robot_id, self.joint_ids[:-1]), self.upperLimits[:-1], self.lowerLimits[:-1])]
        return positions, self.p.getLinkState(self.robot_id, self.end_effector_id)

if __name__ == "__main__": # testing
    env = CrustCrawler(display=True)
    env.step([0,0,0,0,0,0,1])
    while True:
        env.p.stepSimulation()#o, r, eo, inf = env.step(None)
	
