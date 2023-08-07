import gymnasium as gym
from gymnasium import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random


MAX_EPISODE_LEN = 20000

class PandaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.step_counter = 0
        p.connect(p.DIRECT)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])
        self.action_space = spaces.Box(np.array([-1]*3), np.array([1]*3))
        self.observation_space = spaces.Box(np.array([-1,-1,-1,-1,-1,-1,-1,-1]), np.array([1,1,1,1,1,1,1,1]))
        self.T_reach_interval = 1


    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([0.,-math.pi,math.pi/2.])

        dv = 1/240
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = 0.08
        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [currentPosition[0] + dx,
                       currentPosition[1] + dy,
                       currentPosition[2] + dz]

        jointPoses = p.calculateInverseKinematics(self.pandaUid,11,newPosition, orientation)[0:7]

        p.setJointMotorControlArray(self.pandaUid, list(range(7))+[9,10], p.POSITION_CONTROL, list(jointPoses)+2*[fingers])

        p.setRealTimeSimulation(1)
        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        #vel_robot = p.getLinkState(self.pandaUid, 11, computeLinkVelocity=1)[6]
        #print(math.sqrt(vel_robot[0]**2+vel_robot[1]**2+vel_robot[2]**2))

        # plug in the reward_function
        reward=self.reward_function()
        truncated = bool(reward)


        # plug in the termination condition
        terminated = False
        if self.step_counter >= MAX_EPISODE_LEN:
            terminated = True
            print(self.step_counter)
        else:
            self.step_counter += 1 

        #info = {'object_position': state_object}

        self.observation = list(state_robot)+list(state_object) + [(self.step_counter-0.5*MAX_EPISODE_LEN)/(0.5*MAX_EPISODE_LEN), (self.T_reach-0.5*MAX_EPISODE_LEN)/(0.5*MAX_EPISODE_LEN)]  
        #if terminated or truncated:
        #print('step: ', np.array(self.observation).astype(np.float32), self.step_counter)

        return np.array(self.observation).astype(np.float32), reward, terminated, truncated, {}

    def reset(self, seed=None):
        
        self.step_counter = 0                
        self.T_reach = 1000
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0) # we will enable rendering after we loaded everything
        urdfRootPath=pybullet_data.getDataPath()
        p.setGravity(0,0,-9.81)

        planeUid = p.loadURDF(os.path.join(urdfRootPath,"plane.urdf"), basePosition=[0,0,-0.65])

        rest_poses = [0,-0.215,0,-2.57,0,2.356,2.356,0.08,0.08]
        self.pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)
        for i in range(7):
            p.resetJointState(self.pandaUid,i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid,10, 0.08)
        tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

        #trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

        self.state_object=[random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.01]
        self.objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=self.state_object)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (p.getJointState(self.pandaUid,9)[0], p.getJointState(self.pandaUid, 10)[0])
        self.observation = list(state_robot) + list(self.state_object)+[-1,(self.T_reach-0.5*MAX_EPISODE_LEN)/(0.5*MAX_EPISODE_LEN)]
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
        #print('reset: ', np.array(self.observation).astype(np.float32))
        return np.array(self.observation).astype(np.float32), {}

    def render(self, mode='human'):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                            distance=.7,
                                                            yaw=90,
                                                            pitch=-70,
                                                            roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(960) /720,
                                                     nearVal=0.1,
                                                     farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                              height=720,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720,960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def reward_function(self):
        # The reward function is here
        currentPosition = np.array(p.getLinkState(self.pandaUid,11)[0])
        target = np.array(self.state_object)


        distance = math.sqrt((target[0]-currentPosition[0])**2+(target[1]-currentPosition[1])**2+(target[2]-currentPosition[2])**2)

        if (distance <= 0.25) and (self.step_counter*(1-self.T_reach_interval)<self.T_reach<self.step_counter*(1+self.T_reach_interval)):
            if self.T_reach_interval >=0.2:
                self.T_reach_interval -=0.1
            return 1
        elif distance <= 0.25:
            return 0.5
        else:
            return 0

    def close(self):
        p.disconnect()