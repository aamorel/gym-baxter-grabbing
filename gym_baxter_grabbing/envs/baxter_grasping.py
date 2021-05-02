



import pybullet as p
import numpy as np
import os
import random
from pathlib import Path
import pyquaternion as pyq
from gym_baxter_grabbing.envs.robot_grasping import RobotGrasping
import json
import xml.etree.ElementTree as ET
from .xacro import _process



def accurateIK(bodyId, end_effector_id, targetPosition, targetOrientation, lowerLimits, upperLimits, jointRanges,
               restPoses,
               useNullSpace=False, maxIter=10, threshold=1e-4):

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



class BaxterGrasping(RobotGrasping):

    def __init__(self, display=False, obj='cube', delta_pos=[0, 0],
                 steps_to_roll=1, mode='joint positions', y_pose=0.12, random_var=None,
                 finger="extended_narrow", slot=3, tip="basic_soft", grasp="inner", limit_scale=0.13):
        
        self.y_pos = y_pose
        self.mode = mode
        if mode == 'joint positions':
            self.joints_id = [34, 35, 36, 37, 38, 40, 41, 49, 51]
            #self.ids_in_ranges = [10, 11, 12, 13, 14, 15, 16, 17, 18]
            self.n_joints = len(self.joints_id)
            
        obj = obj.strip()
        cwd = Path(__file__).parent
        if obj not in {"cuboid", "cube", "sphere", "cylinder", "paper roll"}:
            baxtertag = ET.parse(cwd/'objects'/obj/f"{obj}.urdf").findall("baxter")
            if len(baxtertag)>0: # if the baxter tag is specified in the urdf of the object like <baxter finger="extended_wide" slot="2" tip="basic_soft"/>, arguments in init will be overwritten
                baxtertag = baxtertag[0]
                finger = baxtertag.get('finger') or finger
                slot = int(baxtertag.get('slot') or slot)
                tip = baxtertag.get('tip') or tip
                grasp = baxtertag.get('grasp') or grasp
                limit_scale = float(baxtertag.get('limit_scale') or limit_scale)
            else: baxtertag = None
        else: baxtertag = None
        if finger not in {"extended_narrow", "extended_wide", "standard_narrow", "standard_wide"}:
            raise NameError(f"The finger value {f'in the baxter tag in {obj} ' if baxtertag else ''}must be either: extended_narrow, extended_wide, standard_narrow, standard_wide")
        elif tip not in {"basic_hard", "basic_soft", "paddle", "half_round"}:
            raise NameError(f"The tip value {f'in the baxter tag in {obj} ' if baxtertag else ''}must be either: basic_hard, basic_soft, paddle, half_round")
        elif grasp not in {"inner", "outer"}:
            raise NameError(f"The grasp value {f'in the baxter tag in {obj} ' if baxtertag else ''}must be either: inner, outer")
        elif not 0<slot<5:
            raise NameError(f"The slot value {f'in the baxter tag in {obj} ' if baxtertag else ''}must be either: 1, 2, 3, 4")
        elif (not (isinstance(limit_scale, int) or isinstance(limit_scale, float))) or limit_scale<0:
            raise NameError(f"The limit_scale value {f'in the baxter tag in {obj} ' if baxtertag else ''}must be a positive scalar")
                    
                    
        urdf = Path(cwd/f"robots/generated/{finger}_{slot}_{tip}_{grasp}_{np.round(limit_scale, 2)}.urdf")
        urdf.parent.mkdir(exist_ok=True)
        if not urdf.is_file(): # create the file if doesn't exist
            _process(cwd/"robots/baxter_description/urdf/baxter_symmetric.xacro", dict(output=urdf, just_deps=False, xacro_ns=True, verbosity=1, mappings={'finger':finger, "slot":str(slot), 'tip':tip+"_tip", "grasp":grasp, "limit_scale":str(limit_scale)})) # convert xacro to urdf
        
        super().__init__(
            robot=lambda: self.p.loadURDF(fileName=str(urdf), basePosition=[0, -0.8, -.074830], baseOrientation=[0,0,-1,-1], useFixedBase=False, flags=self.p.URDF_USE_SELF_COLLISION),
            display=display,
            obj=obj,
            pos_cam=[1.2, 180, -40],
            gripper_display=False,
            steps_to_roll=steps_to_roll,
            random_var=random_var,
            delta_pos=delta_pos,
            initial_position_object=[0, 0.12, 0],
            table_height=0.76,
            mode = mode,
            end_effector_id = 48,
            joint_ids = [34, 35, 36, 37, 38, 40, 41, 49, 51],
            n_actions = 8,
            center_workspace = 34,
            radius = 1.2,
            contact_ids=[47, 48, 49, 50, 51, 52],
            disable_collision_pair = [[49, 51], [38, 53], [27, 29], [16, 31], [1, 10], [1, 7], [1, 5], [0, 10], [0, 7], [0, 5], [0, 1], [40, 53], [37, 54], [34, 36], [18, 31], [15, 32], [12, 14], [35, 2], [34, 2], [14, 2], [13, 2], [12, 2], [2, 7], [1, 2], [0, 2], [41, 53], [36, 2], [34, 54], [54, 2], [50, 55], [38, 54], [1, 53], [1, 38], [1, 37], [16, 32], [19,31], [49,52], [50,51], [50,52]],
            change_dynamics = {i:{'collisionMargin':0.04} for i in (25, 26, 27, 29, 47, 48, 49, 51)} | {i:{'collisionMargin':0.04, 'lateralFriction':1} for i in (28, 30, 50, 52)}
        )


    def get_object(self, obj=None):
        # create object to grab
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
        

        
    def reset_robot(self):
        for i,v in zip([34, 35, 36, 37, 38, 40, 41,  12, 13, 14, 15, 16, 18, 19], [-0.08, -1.0, -1.19, 1.94, 0.67, 1.03, -0.50,  0.08, -1.0,  1.19, 1.94, -0.67, 1.03, 0.50]):
            self.p.resetJointState(self.robot_id, i, targetValue=v) # put baxter in untuck position

        

    def step(self, action):

        if self.mode == 'inverse kinematic':
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
            target_gripper_pos = (target_gripper + 1) * 0.010416
            self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=49, controlMode=self.p.POSITION_CONTROL,
                                    targetPosition=target_gripper_pos, force=MAX_FORCE)
            self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=51, controlMode=self.p.POSITION_CONTROL,
                                    targetPosition=-target_gripper_pos, force=MAX_FORCE)

        if self.mode == 'joint positions':
            # we want one action per joint (gripper is composed by 2 joints but considered as one)
            assert(len(action) == self.n_actions)
            self.info['closed gripper'] = action>0
            # add the command for the last gripper joint
            #self.action = np.append(self.action, -self.action[-1])
            commands = np.append(action, -action[-1])
            #commands = [l+(a+1)/2*(u-l) for a,u,l in zip(self.action, self.upperLimits, self.lowerLimits)]
            #for id, command, v ,f in zip(self.joints_id, commands, self.maxVelocity, self.maxForce):
                #self.p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=id, controlMode=self.p.POSITION_CONTROL, targetPosition=command, maxVelocity=v, force=f)
        return super().step(commands)



    def compute_grip_info(self):
        self.info['closed gripper'] = True
        if self.joint_poses[-2] > 0.0003 or self.joint_poses[-2] < -0.0003:
            self.info['closed gripper'] = False
        if self.joint_poses[-1] > 0.0003 or self.joint_poses[-1] < -0.0003:
            self.info['closed gripper'] = False

    def get_state(self):

        positions = [2*(s[0]-u)/(u-l) + 1 for s,u,l in zip(self.p.getJointStates(self.robot_id, self.joints_id[:-1]), self.upperLimits[:-1], self.lowerLimits[:-1])]
        return positions, self.p.getLinkState(self.robot_id, self.end_effector_id)


