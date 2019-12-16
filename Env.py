import airsim
import numpy as np
from SpatialPyramidPooling import SpatialPyramidPooling
from scipy.stats import entropy
from math import exp
import time
import datetime as dt
import math
from PIL import Image
from keras.models import load_model
from AdaptedModel import resnet8
import pandas as pd

class AirSimEnv:
    def __init__(self, past_k_size=5,navigate_model_path = None, resolution = 128, bandwidth_file_path = None):
        self.connect()

        self.forward = airsim.DrivetrainType.ForwardOnly
        self.yaw = airsim.YawMode(False, 0)

        self.resolution = resolution

        self.past_k_size = past_k_size
        self.mem_idx = 0
        self.throughput_memory = np.zeros(self.past_k_size, dtype=np.float32)
        #self.strength_memory = np.zeros(self.past_k_size, dtype=np.float32)
        if navigate_model_path is not None:
            self.navigate_model = load_model(navigate_model_path,custom_objects={"SpatialPyramidPooling":SpatialPyramidPooling})
        else:
            self.navigate_model = resnet8(None,None,3)

        if bandwidth_file_path is not None:
            self.bandwidth_file_path = bandwidth_file_path

    def connect(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def store_transition(self, throughput,):
        index = self.mem_idx % self.past_k_size
        self.throughput_memory[index] = throughput
        #self.strength_memory[index] = strength
        self.mem_idx += 1


    def get_image(self,):
        while True:  # 有的时候会失败，这时候重试就好
            try:
                image_request = airsim.ImageRequest(0,airsim.ImageType.Scene, False, False)
                image_response = self.client.simGetImages([image_request, ])[0]

                image1d = np.frombuffer(image_response.image_data_uint8,dtype=np.uint8)
                image_rgba = image1d.reshape(image_response.height,image_response.width, 3)
                image = Image.fromarray(image_rgba)
                image.thumbnail((self.resolution, self.resolution))
                break
            except:
                print('Error when capturing image, retrying...')
        return image

    def generator_bandwidth(self,):
        dataset = pd.read_csv(self.bandwidth_file_path, sep='\t')
        for i in range(len(dataset)):
            yield dataset.loc[i, 'throughput_bpms']
            if i > len(dataset):
                raise StopIteration

    def control(self, steering=0, speed=1):
        yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)[2] + steering
        vx = math.cos(yaw) * speed
        vy = math.sin(yaw) * speed
        self.client.moveByVelocityZAsync(vx, vy, -2, 0.1, self.forward, self.yaw).join()


    def get_drone_state(self):
        return self.client.getMultirotorState()


    def get_roads(self, include_corners=True):
        # 获得地图上的路的信息
        lines = [
            [[-128, -121], [-128, 119]],
            [[-120, -129], [120, -129]],
            [[-120, 127], [120, 127]],
            [[128, -121], [128, 119]],
            [[0, -121], [0, 119]],
            [[-120, 0], [120, 0]],
            [[80, -124], [80, -5]],
        ]
        if include_corners:  # 路的拐弯
            for x0, x1 in [[-128, -120], [0, -8], [0, 8], [120, 128]]:
                corners = [
                    [[x0, -121], [x1, -129]],
                    [[x0, -8], [x1, 0]],
                    [[x0, 8], [x1, 0]],
                    [[x0, 119], [x1, 127]],
                ]
                lines += corners
            for x0, x1 in [[80, 75], [80, 85]]:
                corners = [
                    [[x0, -124], [x1, -129]],
                    [[x0, -5], [x1, 0]],
                ]
                lines += corners
        roads = [(np.array(p), np.array(q)) for p, q in lines]
        return roads


    def get_start_pose(self, random=True, verbose=True):
        # 在路上选择一个起始位置和方向
        if not random:  # 固定选择默认的起始位置
            position = np.array([0., 0.])
            yaw = 0.
        else:  # 随机选择一个位置
            if not hasattr(self, 'roads_without_corners'):
                self.roads_without_corners = self.get_roads(
                    include_corners=False)

            # 计算位置
            road_index = np.random.choice(len(self.roads_without_corners))
            p, q = self.roads_without_corners[road_index]
            t = np.random.uniform(0.3, 0.7)
            position = t * p + (1. - t) * q

            # 计算朝向
            if np.isclose(p[0], q[0]):  # 与 Y 轴平行
                yaws = [0.5 * math.pi, -0.5 * math.pi]
            elif np.isclose(p[1], q[1]):  # 与 X 轴平行
                yaws = [0., math.pi]
            yaw = np.random.choice(yaws)

        if verbose:
            print('起始位置 = {}, 方向 = {}'.format(position, yaw))

        position = airsim.Vector3r(position[0], position[1], -0.6)
        orientation = airsim.to_quaternion(pitch=0., roll=0., yaw=yaw)
        pose = airsim.Pose(position, orientation)
        return pose


    def reset(self, explore_start=False,
              max_epoch_time=None, verbose=True):
        if verbose:
            print('开始新回合')

        # 起始探索
        start_pose = self.get_start_pose(random=explore_start)

        # 再次设置初始位置
        if verbose:
            print('设置初始位置')
        self.client.simSetVehiclePose(start_pose, True)

        # 回合开始时间和预期结束时间
        self.start_time = dt.datetime.now()
        self.end_time = None
        if max_epoch_time:
            self.expected_end_time = self.start_time + \
                                     dt.timedelta(seconds=max_epoch_time)
        else:
            self.expected_end_time = None


    def interpret_action(self, action):
        if action == 0:
            self.resolution = 32
        elif action == 1:
            self.resolution = 64
        elif action == 2:
            self.resolution = 128
        elif action == 3:
            self.resolution = 256
        else:
            self.resolution = 320


    def get_reward(self, energy, complexity, resolution, entropy):
        # 计算奖励，并评估回合是否结束

        collision_info = self.client.simGetCollisionInfo()  # 碰撞信息
        if collision_info.has_collided:  # 如果发生了碰撞，没有奖励，回合结束
            self.end_time = dt.datetime.now()
            return 0.0, True, {'message': 'collided'}

        drone_state = self.client.getMultirotorState()
        drone_point = drone_state.kinematics_estimated.position.to_numpy_array()  # 获取无人机位置信息

        if not hasattr(self, 'roads'):
            self.roads = self.get_roads()

        # 计算位置到各条路的最小距离
        distance = float('+inf')
        for p, q in self.roads:
            # 点到线段的最小距离
            frac = np.dot(drone_point[:2] - p, q - p) / np.dot(q - p, q - p)
            clipped_frac = np.clip(frac, 0., 1.)
            closest = p + clipped_frac * (q - p)
            dist = np.linalg.norm(drone_point[:2] - closest)
            distance = min(dist, distance)  # 更新最小距离

        #reward = math.exp(-1.2 * distance)  # 基于距离的奖励函数
        #reward =  energy + complexity + resolution + entropy
        print(energy,' ',complexity[0][0],' ',resolution,' ',entropy)
        scaled_complexity = 4.8 * complexity[0][0] - 64
        k1 = 1.0
        k2 = 3.0
        k3 = 1.0
        k4 = 6.0
        #print(k2*exp(k3*((4.8*complexity[0][0]-64)-resolution)))
        reward = 500.0 - k1*energy - k2*exp(k3*(scaled_complexity-resolution)) - k4*entropy
        #reward = 100
        if distance > 2:  # 偏离路面太远，回合结束
            self.end_time = dt.datetime.now()
            return reward, True, {'distance': distance}

        # 判断是否超时
        now = dt.datetime.now()
        if self.expected_end_time is not None and now > \
                self.expected_end_time:
            self.end_time = now
            info = {'start_time': self.start_time,
                    'end_time': self.end_time}
            return reward, True, info  # 回合超时结束

        return reward, False, {}


    def step(self, last_action, action, max_chunk_time):
        x=0
        entropy_ = 0
        complexity = 0
        collision = 0
        #根据action选取不同的image_buf
        self.interpret_action(action)
        image_buf = np.zeros((1, self.resolution, self.resolution, 3))

        #载入带宽数据集，计算时延，根据时延判断是否卸载
        gen_bandwidth = self.generator_bandwidth()

        self.store_transition(throughput=next(gen_bandwidth), ) #存储带宽
        start_time=dt.datetime.now()
        while(dt.datetime.now()-start_time).seconds<max_chunk_time:
            image_buf[0] = self.get_image()
            model_output = self.navigate_model.predict([image_buf])
            steering = float(model_output[0][0])
            #print(steering)
            collision += model_output[1][0][1]
            entropy_ += entropy(model_output[1][0])
            complexity += float(model_output[2][0])
            print(steering,' ',collision,' ',entropy_,' ',complexity)
            x=x+1
        #TODO 获取能耗仿真
        ave_entropy = entropy_ / x
        ave_energy = 3

        ave_collision = np.expand_dims([collision / x], axis=2)
        ave_complexity = np.expand_dims([complexity / x], axis=2)
        last_action = np.expand_dims([last_action], axis=2)
        throughput_memory = np.expand_dims(self.throughput_memory, axis=2)
        reward, done, info = self.get_reward(ave_energy,ave_complexity,self.resolution,ave_entropy)
        print('reward: %.2f' % reward)
        observation = [ave_collision,ave_complexity,last_action,throughput_memory]

        return observation, reward, done, info
