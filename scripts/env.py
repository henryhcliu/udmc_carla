import atexit
import logging
import os
import random
import sys
import time
from collections import deque

import numpy as np

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
except IndexError:
    pass

from tqdm import tqdm

try:
    import queue  # python3
except ImportError:
    import Queue as queue  # python2

import copy
import math

import carla
import cv2
import joblib
from carla_birdeye_view import (BirdViewCropType, BirdViewProducer,
                                PixelDimensions)
from others_agent import OthersAgent
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

DT_ = None  # [s] delta time step, = 1/FPS_in_server
NO_RENDERING = False
BEV_RES_X = 200
RES_X = 1280
RES_Y = 720

START_TIME = 3

DISPLAY_METHOD= 'spec' # 'pygame' or 'spec'

if DISPLAY_METHOD == 'pygame':
    import pygame
    pygame.init()

    def get_font():
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)


    def display_image(surface, image_array, blend=False):
        image_surface = pygame.surfarray.make_surface(image_array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))


def draw_waypoints(world, waypoints, z=0.5, color=(255, 0, 0)):
    color = carla.Color(r=color[0], g=color[1], b=color[2], a=255)
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z)
        # world.debug.draw_points(begin, size=0.05, color=color, life_time=0.1)


class PIDAccelerationController():
    """
    PIDAccelerationController implements acceleration control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)
        self.last_control = [0,0]

    def run_step(self, target_acc, debug=False):
        """
        Execute one step of acceleration control to reach a given target speed.

            :param target_acceleration: target acceleration in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_acc = self.get_acc()

        if debug:
            print('Current acceleration = {}'.format(current_acc))

        # print('err', current_acc, target_acc, target_acc-current_acc)
        return self._pid_control(target_acc, current_acc)

    def _pid_control(self, target_acc, current_acc):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """
        error = target_acc - current_acc
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        acc_pred = (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie)
        acc_pred = acc_pred # normalize the control signal
        
        delta_acc = acc_pred - self.last_control[0]
        threashold = 3/20
        if delta_acc > threashold:
            acc_pred = self.last_control[0] + threashold
            print("acc_pred:", acc_pred)
        elif delta_acc < -threashold:
            acc_pred = self.last_control[0] - threashold
            print("acc_pred:", acc_pred)
        self.last_control[0] = acc_pred
        
        return np.clip(acc_pred, -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt

    def get_acc(self):
        # direction flag, 1: forward, -1: backward
        flag = 1

        yaw = np.radians(self._vehicle.get_transform().rotation.yaw)
        ax = self._vehicle.get_acceleration().x
        ay = self._vehicle.get_acceleration().y 
        acc_yaw = math.atan2(ay, ax)
        error = acc_yaw - yaw
        if error > math.pi:
            error -= 2 * math.pi
        elif error < -math.pi:
            error += 2 * math.pi
        error = math.fabs(error)
        if error > math.pi / 2:
            flag = -1

        return flag * np.sqrt(ax**2+ay**2)*0.1


class Env:
    def __init__(self, host="localhost", port=2000, map_id='05', birdeye_view=False, 
                 display_method='spec',recording=True, dt=0.05, is_Leaderboard=False) -> None:
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.steer_ratio = 1/0.7 # 70.0/180.0*math.pi
        self.is_Leaderboard = is_Leaderboard

        # Gaussian Process Regression init
        # # kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        # kernel = 1.0 * RBF(length_scale=1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(
        #     noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))
        # self.gp = GaussianProcessRegressor(kernel=kernel)
        self.inferMethod = 'gpr'
        if self.inferMethod == 'gpr':
            self.gpr = joblib.load('gpr.pkl')
        elif self.inferMethod == 'lstm':
            self.lstm = joblib.load('lstm.pkl')
        else:
            print("Now, no infer method is selected and the prediction of the other vehicles' trajectory is not available!")
        
        self.gp_times = np.arange(-15, 0).reshape(-1, 1)

        self.gpr_infer_times = []
        self.lstm_infer_times = []
        self.computing_times = []

        self.dir_name = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())).split('.')[0]

        if not self.is_Leaderboard:
            # Carla init
            map_name = 'Town'+map_id
            if str(self.map).split('/')[-1][:-1] == map_name:
                logging.info('Already load map {}, skipped'.format(map_name))
            else:
                self.world = self.client.load_world(map_name)

            self.blueprint_library = self.world.get_blueprint_library()
            self.ego_vehicle_type = self.blueprint_library.filter("model3")[0]
            self.ego_vehicle = None

        else:
            # find the ego vehicle with the "role_name" as "hero"
            self.ego_vehicle = None
            for actor in self.world.get_actors():
                if actor.attributes.get('role_name') == 'hero':
                    self.ego_vehicle = actor
                    break

        global DT_, DISPLAY_METHOD, HFC_HZ
        DT_ = dt
        HFC_HZ = 1
        DISPLAY_METHOD = display_method
        # exit with cleaning all actors
        atexit.register(self.clean)

        self.original_settings = self.world.get_settings()
        # world in sync mode
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=NO_RENDERING,  # False for debug
            synchronous_mode=True,
            fixed_delta_seconds=DT_/HFC_HZ))
        self.world.tick()
        print('world in sync mode')
        # birdeye view setting informationget_obsget_obs
        self.birdview_on = birdeye_view
        if birdeye_view:
            self.font_res_x = BEV_RES_X
            PG_RES_X = BEV_RES_X + RES_X
            self.birdview_producer = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=BEV_RES_X, height=RES_Y),
                pixels_per_meter=6,
                crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
            )
        else:
            self.font_res_x = 0
            PG_RES_X = RES_X

        # pygame setting
        if DISPLAY_METHOD == 'pygame':
            self.pg_display = pygame.display.set_mode(
                (PG_RES_X, RES_Y),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()
            if self.recording:
                self.videoWriter = cv2.VideoWriter('mapping.mp4', cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),\
                                24, (BEV_RES_X+RES_X, RES_Y)) # fps
        elif DISPLAY_METHOD == 'spec':
            self.spectator = self.world.get_spectator()
        else:
            raise ValueError('display_method should be pygame or spec')
            exit()

        # update if spawn vehicles
        self.spawn_others_bool = False
        self.other_vehicles_auto = False

        self.recording = recording


    def reset(self, ego_transform=None):
        """
        initial environment
        ego_vehicle can be tranformed on the position by setting ego_transform

        ego_transform: carla.Transform
        """
        self.If_record_sv_history = False
        self.If_record_pede_history = False
        self.actor_list = []
        self.other_vehicles_list = []
        self.other_vehicles_queue_list = []
        if self.If_record_sv_history:
            self.other_vehicles_history_list = []
        self.pedestrian_list = []
        self.pedestrian_queue_list = []
        if self.If_record_pede_history:
            self.pedestrian_history_list = []

        # choose a random point for generation
        if ego_transform is None:
            spawn_point = random.choice(
                self.world.get_map().get_spawn_points())
        else:
            spawn_point = ego_transform

        if self.ego_vehicle is None:
            self.ego_vehicle = self.world.spawn_actor(
                self.ego_vehicle_type, spawn_point)
            # # change the max steering angle of the vehicle [FAILED]
            # physics_control = self.ego_vehicle.get_physics_control()
            # physics_control.wheels[0].max_steer_angle = 36 # set max steer angle to 36 deg
            # physics_control.wheels[1].max_steer_angle = 36 # set max steer angle to 36 deg
            # self.ego_vehicle.apply_physics_control(physics_control)
            # # print current max steer angle
            # print('max steer angle: ', self.ego_vehicle.get_physics_control().wheels[0].max_steer_angle, \
            #       self.ego_vehicle.get_physics_control().wheels[1].max_steer_angle)
            self.actor_list.append(self.ego_vehicle)
        else:
            self.ego_vehicle.set_transform(spawn_point)
        
        self._acc_controller = PIDAccelerationController(self.ego_vehicle, K_P=0.34, K_I=0.01, K_D=0.001, dt=DT_/HFC_HZ) # original K_P=0.28, K_I=0.01, K_D=0.001

        if self.ego_vehicle is not None:
            # setting camera information
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '{}'.format(RES_X))
            camera_bp.set_attribute('image_size_y', '{}'.format(RES_Y))
            ifFPV = True
            if ifFPV:
                camera_bp.set_attribute('fov', '110')
                camera_pos = carla.Transform(carla.Location(x=-6, z=3))
                attached_obj = self.ego_vehicle
            else:
                camera_pos = carla.Transform(carla.Location(z=0), carla.Rotation(pitch=0))
                attached_obj = self.spectator
            # camera_bp.set_attribute('fov', '110')
            # camera_pos = carla.Transform(carla.Location(x=-6, z=3))
            
            self.camera = self.world.spawn_actor(
                camera_bp,
                camera_pos,
                attach_to=attached_obj)
            self.camera.image_size_x = 720
            self.camera.image_size_x = 480
            if self.recording:
                self.actor_list.append(self.camera)
                # self.image_queue = queue.Queue()
                self.camera.listen(lambda image: image.save_to_disk('images/' + str(self.dir_name) + '/%06d.png' % image.frame_number))

        self.time = 0

        self.world.tick()
        if DISPLAY_METHOD == 'pygame':
            self.clock.tick()

    def spawn_other_vehicles(self, transform_list, auto_mode=False):
        bp_lib = self.world.get_blueprint_library()
        bp_types = []
        self.spawn_other_agents = []
        self.other_vehicles_auto = auto_mode

        # bp_types.extend(bp_lib.filter('vehicle.nissan.*'))
        # bp_types.extend(bp_lib.filter('vehicle.audi.*'))
        # bp_types.extend(bp_lib.filter('vehicle.bmw.*'))
        # bp_types.extend(bp_lib.filter('vehicle.chevrolet.*'))
        # bp_types.extend(bp_lib.filter('vehicle.dodge.*'))
        # bp_types.extend(bp_lib.filter('vehicle.ford.*'))
        bp_types.extend(bp_lib.filter('vehicle.tesla.model3'))

        for vehicle_i in range(len(transform_list)):
            bp = random.choice(bp_types)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)

            transform = transform_list[vehicle_i]
            try:
                vehicle = self.world.spawn_actor(bp, transform)
            except:
                print('spawn other vehicle '+str(vehicle_i)+' failed!')
                continue
            # self.world.tick()
            if vehicle is not None:
                if auto_mode:
                    vehicle.set_autopilot(True)
                    # set the desired speed for autopilot
                    # vehicle.set_target_velocity(carla.Vector3D(x=25, y=0, z=0))
                    self.other_vehicles_list.append(vehicle)
                    self.other_vehicles_queue_list.append(queue.Queue(maxsize=15))
                    if self.If_record_sv_history:
                        self.other_vehicles_history_list.append([])

                else:
                    agent = OthersAgent(vehicle)
                    self.spawn_other_agents.append(agent)
                    self.other_vehicles_list.append(vehicle)
                    self.other_vehicles_queue_list.append(queue.Queue(maxsize=15))
                    if self.If_record_sv_history:
                        self.other_vehicles_history_list.append([])
            else:
                pass
        # tick the world
        self.world.tick()
               
    def spawn_pedestrians(self, transform_list, destination_transform_list):
        bp_lib = self.world.get_blueprint_library()
        # self.pedestrian_list = []
        self.pedestrian_ai_controller_list = []
        self.pedestrian_control_list = []
        for i, transform, destination_transform in zip(range(len(transform_list)), transform_list, destination_transform_list):
            try:
                bp = random.choice(bp_lib.filter('walker.pedestrian.*'))
                pedestrian = self.world.spawn_actor(bp, transform)
                location = pedestrian.get_location()
                print('pedestrian', i, 'spawned at', location.x, location.y, location.z)
                self.pedestrian_list.append(pedestrian)
                self.pedestrian_queue_list.append(queue.Queue(maxsize=15))
                if self.If_record_pede_history:
                    self.pedestrian_history_list.append([])
                pedestrian_controller = pedestrian.get_control()
                if i == 0:
                    pedestrian_controller.speed = 0.1
                else:
                    pedestrian_controller.speed = 0.045
                pedestrian_controller.direction.y = destination_transform.location.y - transform.location.y
                pedestrian_controller.direction.x = destination_transform.location.x - transform.location.x
                pedestrian_controller.use_lanes = False
                pedestrian_controller.aim_for_destination = True
                self.pedestrian_control_list.append(pedestrian_controller)
                # Apply the controller to the pedestrian
                pedestrian.apply_control(pedestrian_controller)
                # self.pedestrian_ai_controller_list.append(self.world.spawn_actor(bp_lib.find('controller.ai.walker'),
                #                                                   carla.Transform(), pedestrian))
                # self.pedestrian_ai_controller_list[i].start()
                # self.pedestrian_ai_controller_list[i].go_to_location(destination_transform.location)
                self.world.tick()
                
                
                # self.world.tick()
            except:
                print('spawn pedestrian failed!')
                continue
        # self.world.tick()

    def update_other_agents(self):
        for agent in self.spawn_other_agents:
            control = agent.run_step()
            if control is None:
                control = carla.VehicleControl()

            if isinstance(control, carla.VehicleControl):
                agent._vehicle.apply_control(control)
                steer_, throttle_, brake_ = control.steer, control.throttle, control.brake
            else:
                if self.time >= START_TIME:  # starting time
                    steer_, throttle_, brake_ = action
                else:
                    steer_ = 0
                    throttle_ = 0.5
                    brake_ = 0

                agent._vehicle.apply_control(carla.VehicleControl(throttle=float(
                    throttle_), steer=float(steer_*20), brake=float(brake_)))
    def infer_vehicle_states(self, gp, latest_states, num_inf=10):
        latest_times = np.arange(-latest_states.shape[0], 0).reshape(-1, 1)
        inferred_states = []
        
        for i in range(1, num_inf+1):
            next_time = np.array([[latest_times[-1][0] + i]])
            next_state = gp.predict(next_time)
            inferred_states.append(next_state.flatten())
        
        return np.array(inferred_states)

    def get_pedes(self, cur_state=None, dist_threshold=10):
        pedes = []
        heights = []
        pedes_infer = []
        # add the current states of the pedestrians to the state queue for trajectory prediction:
        for sv in self.pedestrian_list:
            sv_state = sv.get_transform()
            sv_state_gp = [sv_state.location.x, -sv_state.location.y, -sv_state.rotation.yaw/180*3.14]
            sv_vel = sv.get_velocity()
            sv_state_gp.append(sv_vel.x)
            sv_state_gp.append(-sv_vel.y)
            # judge if the queue is full, if so, pop the first element
            if self.pedestrian_queue_list[self.pedestrian_list.index(sv)].full():
                self.pedestrian_queue_list[self.pedestrian_list.index(sv)].get()
            # put the current state into the queue
            self.pedestrian_queue_list[self.pedestrian_list.index(sv)].put(sv_state_gp)
            if self.If_record_pede_history and (self.time > 5) and (self.time<25):
                self.pedestrian_history_list[self.pedestrian_list.index(sv)].append(sv_state_gp)
     
        for pede in self.pedestrian_list:
            location = pede.get_location()
            transform = pede.get_transform()
            phi = transform.rotation.yaw * np.pi / 180
            if cur_state is None:
                pedes.append([location.x, -location.y, -phi])
            else:
                dist = np.sqrt((location.x-cur_state[0])**2 + (-location.y-cur_state[1])**2)
                if dist < dist_threshold:
                    pedes.append([location.x, -location.y, -phi])
                    pede_hist=self.pedestrian_queue_list[self.pedestrian_list.index(pede)].queue
                    pede_hist = np.array(pede_hist)
                    # if the queue is not full, fill it with the oldest state from the queue
                    if len(pede_hist) < 15:
                        empty_num = 15 - len(pede_hist)
                        for i in range(empty_num):
                            pede_hist = np.insert(pede_hist, 0, pede_hist[0], axis=0)
                    # self.gp.fit(self.gp_times, obs_hist)
                    # obs_infer.append(self.infer_vehicle_states(self.gp, obs_hist, num_inf=10))
                    pede_hist = pede_hist.reshape(-1, 1).T
                    # normalize the x, y of the SVs history states
                    pede_hist = self.norm_pre_states(pede_hist)
                    # infer the x, y of the SVs history states\
                    # tick = time.time()
                    if self.inferMethod == "gpr":
                        pede_infer_i = self.gpr.predict(pede_hist)
                    elif self.inferMethod == "lstm":
                        pede_hist = pede_hist.reshape(1, 15, 5)
                        pede_infer_i = self.lstm.predict(pede_hist)
                    else:
                        # just repeat the last state for 10 times
                        last_state = pede_hist[0,-5:-2]
                        pede_infer_i = np.tile(last_state, 10)
                        pede_infer_i = pede_infer_i.reshape(1, 30)
                    # tock = time.time()
                    # infer_time_once.append(tock-tick)
                        # print("lstm infer time: ", tock-tick)
                    pede_infer_i = np.array(pede_infer_i)
                    # restore the x, y of the SVs history states
                    pede_infer_i = self.restore_pre_states(pede_infer_i)
                    pede_infer_i = pede_infer_i.reshape(-1, 3)
                    pedes_infer.append(pede_infer_i)
                    heights.append(location.z)
        return pedes, heights, pedes_infer

    def get_obs(self, cur_state=None, dist_threshold=15):
        obs = []
        heights = [] # the height of the SVs, for the bounding box drawing in Carla
        obs_infer = []
        infer_time_once = []
        # add the current states of the SVs to the state queue for trajectory prediction:
        for sv in self.other_vehicles_list:
            sv_state = sv.get_transform()
            sv_state_gp = [sv_state.location.x, -sv_state.location.y, -sv_state.rotation.yaw/180*3.14]
            sv_vel = sv.get_velocity()
            sv_state_gp.append(sv_vel.x)
            sv_state_gp.append(-sv_vel.y)
            # judge if the queue is full, if so, pop the first element
            if self.other_vehicles_queue_list[self.other_vehicles_list.index(sv)].full():
                self.other_vehicles_queue_list[self.other_vehicles_list.index(sv)].get()
            # put the current state into the queue
            self.other_vehicles_queue_list[self.other_vehicles_list.index(sv)].put(sv_state_gp)
            if self.If_record_sv_history and (self.time > 5) and (self.time<25):
                self.other_vehicles_history_list[self.other_vehicles_list.index(sv)].append(sv_state_gp)
        # get the SVs states within the distance threshold
        for car in self.other_vehicles_list:
            location = car.get_location()
            transform = car.get_transform()
            phi = transform.rotation.yaw * np.pi / 180
            if cur_state is None:
                obs.append([location.x, -location.y, -phi])
            else:
                dist = np.sqrt((location.x-cur_state[0])**2 + (-location.y-cur_state[1])**2)
                if dist < dist_threshold:
                    obs.append([location.x, -location.y, -phi])
                    obs_hist=self.other_vehicles_queue_list[self.other_vehicles_list.index(car)].queue
                    obs_hist = np.array(obs_hist)
                    # if the queue is not full, fill it with the oldest state from the queue
                    if len(obs_hist) < 15:
                        empty_num = 15 - len(obs_hist)
                        for i in range(empty_num):
                            obs_hist = np.insert(obs_hist, 0, obs_hist[0], axis=0)
                    # self.gp.fit(self.gp_times, obs_hist)
                    # obs_infer.append(self.infer_vehicle_states(self.gp, obs_hist, num_inf=10))
                    obs_hist = obs_hist.reshape(-1, 1).T
                    # normalize the x, y of the SVs history states
                    obs_hist = self.norm_pre_states(obs_hist)
                    # infer the x, y of the SVs history states\
                    tick = time.time()
                    if self.inferMethod == "gpr":
                        obs_infer_i = self.gpr.predict(obs_hist)
                    elif self.inferMethod == "lstm":
                        obs_hist = obs_hist.reshape(1, 15, 5)
                        obs_infer_i = self.lstm.predict(obs_hist)
                    else:
                        # just repeat the last state for 10 times
                        last_state = obs_hist[0,-5:-2]
                        obs_infer_i = np.tile(last_state, 10)
                        obs_infer_i = obs_infer_i.reshape(1, 30)
                    tock = time.time()
                    infer_time_once.append(tock-tick)
                        # print("lstm infer time: ", tock-tick)
                    obs_infer_i = np.array(obs_infer_i)
                    # restore the x, y of the SVs history states
                    obs_infer_i = self.restore_pre_states(obs_infer_i)
                    obs_infer_i = obs_infer_i.reshape(-1, 3)
                    obs_infer.append(obs_infer_i)
                    heights.append(location.z)
        infer_time_sum = np.sum(infer_time_once)
        if self.inferMethod == "gpr":
            self.gpr_infer_times.append(infer_time_sum)
            # print("gpr infer time: ", tock-tick)
        elif self.inferMethod == "lstm":
            self.lstm_infer_times.append(infer_time_sum)
        return obs, heights, obs_infer

    def norm_pre_states(self, history_data):
        self.extract_val = copy.deepcopy(history_data[0][0:2])
        for i in range(15):
            history_data[0][i*5:i*5+2] = (history_data[0][i*5:i*5+2] - self.extract_val)*10
        return history_data
    
    def restore_pre_states(self, predict_data):
        for i in range(10):
            predict_data[0][i*3:i*3+2] = (predict_data[0][i*3:i*3+2]/10 + self.extract_val)
        return predict_data

    # def get_state(self):
    #     """
    #     get the vehicle's state, TODO to fit our dynamic model
    #     """
    #     self.location = self.ego_vehicle.get_location()
    #     self.location_ = np.array(
    #         [self.location.x, self.location.y, self.location.z])

    #     self.transform = self.ego_vehicle.get_transform()
    #     phi = self.transform.rotation.yaw * np.pi / 180

    #     self.velocity = self.ego_vehicle.get_velocity()
    #     vx = self.velocity.x
    #     vy = self.velocity.y

    #     beta_candidate = np.arctan2(
    #         vy, vx) - phi + np.pi*np.array([-2, -1, 0, 1, 2])
    #     local_diff = np.abs(beta_candidate - 0)
    #     min_index = np.argmin(local_diff)
    #     beta = beta_candidate[min_index]

    #     # state = [self.velocity.x, self.velocity.y, self.yaw, self.angular_velocity.z]
    #     state = [
    #         self.location.x,  # x
    #         self.location.y,  # y
    #         np.sqrt(vx**2 + vy**2),  # v
    #         phi,  # phi
    #         beta,  # beta
    #     ]

    #     return np.array(state)

    def step(self, state, transform_mode=False):
        """
        excute one step of simulation

        state: if state is action, it's type is carla.VehicleControl or [steer_, throttle_, brake_]
               other is the transform data(carla.Transform or [x,y,z,roll,pitch,yaw])
        """
        if not transform_mode:
            steer_ = state[1]*self.steer_ratio # transform the steering angle from rad to -1~1
            print("steer_:", steer_)
            delta_steer = steer_ - self._acc_controller.last_control[1]
            threashold = 4.71/20*self.steer_ratio # 9.42; 4.71
            if delta_steer > threashold:
                steer_ = self._acc_controller.last_control[1] + threashold
                print("last_steer:", self._acc_controller.last_control[1])
                print("steer_processed:", steer_)
            elif delta_steer < -threashold:
                steer_ = self._acc_controller.last_control[1] - threashold
                print("last_steer:", self._acc_controller.last_control[1])
                print("steer_processed:", steer_)
            self._acc_controller.last_control[1] = steer_
            

            for i in range(HFC_HZ):
                throttle_tmp = self._acc_controller.run_step(state[0])

                if throttle_tmp >= 0:
                    throttle_ = throttle_tmp
                    brake_ = 0
                    reverse_ = False
                else:
                    throttle_ = abs(throttle_tmp)
                    brake_ = 0
                    reverse_ = True
                if not self.is_Leaderboard:
                    self.ego_vehicle.apply_control(carla.VehicleControl(
                            throttle=float(throttle_), steer=float(-steer_), brake=float(brake_), reverse=reverse_))
                    # for debug
                    # for pede in self.pedestrian_list:
                    #     # pede.apply_control(self.pedestrian_control_list[self.pedestrian_list.index(pede)])
                    #     print('pedestrian', self.pedestrian_list.index(pede), 'is moving from ', pede.get_location().x, pede.get_location().y, 'to', self.pedestrian_control_list[self.pedestrian_list.index(pede)].direction.x, self.pedestrian_control_list[self.pedestrian_list.index(pede)].direction.y)
                    

                    self.world.tick()
            
        else:
            steer_ = state[1]
            accelerate_ = state[0]
            new_state = state[2]
            height = new_state[1]
            transform = new_state[0]
            self.ego_vehicle.set_transform(carla.Transform(carla.Location(
                x=transform[0], y=-transform[1], z=height), carla.Rotation(yaw=-np.degrees(transform[2]))))

        if self.spawn_others_bool:
            if not self.other_vehicles_auto:
                self.update_other_agents()

        if DISPLAY_METHOD == 'pygame':
            self.clock.tick()
        # self.world.tick()
        self.time += DT_

        # image_rgb = self.image_queue.get()
        # image_array = np.reshape(np.frombuffer(image_rgb.raw_data, dtype=np.dtype('uint8')),
        #                          (image_rgb.height, image_rgb.width, 4))
        # image_array = image_array[:, :, :3]
        # image_array = image_array[:, :, ::-1]

        if self.birdview_on:
            birdview = self.birdview_producer.produce(
                agent_vehicle=self.ego_vehicle  # carla.Actor (spawned vehicle)
            )
            rgb = BirdViewProducer.as_rgb(birdview)
            image_array = np.concatenate((rgb, image_array), axis=1)

            if self.recording:
                self.videoWriter.write(image_array[:,:,::-1])            

        if not transform_mode:
            vel = self.ego_vehicle.get_velocity()
            vel = np.sqrt(vel.x**2 + vel.y**2)
        else:
            vel = transform[3]

        if DISPLAY_METHOD == "pygame":
            display_image(self.pg_display, image_array)
            self.pg_display.blit(
                self.font.render(
                    'Velocity = {0:.2f} m/s'.format(vel), True, (255, 255, 255)),
                (self.font_res_x+8, 10))

            # pygame text display
            v_offset = 25
            bar_h_offset = self.font_res_x+75
            bar_width = 100
            if not transform_mode:
                display_item = {"steering": steer_,
                                "throttle": throttle_, "brake": brake_}
                for key, value in display_item.items():
                    rect_border = pygame.Rect(
                        (bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(self.pg_display, (255, 255, 255), rect_border, 1)
                    if key == "steering":
                        rect = pygame.Rect((bar_h_offset + (1+value) * (bar_width)/2, v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect(
                            (bar_h_offset + value * (bar_width), v_offset + 8), (6, 6))
                    pygame.draw.rect(self.pg_display, (255, 255, 255), rect)
                    self.pg_display.blit(self.font.render(
                        key, True, (255, 255, 255)), (self.font_res_x+8, v_offset+3))

                    v_offset += 18
            else:
                self.pg_display.blit(
                self.font.render(
                    'Accelerating = {0:.2f} m/s2'.format(accelerate_), True, (255, 255, 255)),
                (self.font_res_x+8, 28))

                self.pg_display.blit(
                self.font.render(
                    'Steering = {0:.2f} °'.format(-steer_*180/np.pi), True, (255, 255, 255)),
                (self.font_res_x+8, 46))
            
            pygame.display.flip()

            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        elif DISPLAY_METHOD == "spec":
            transform = self.ego_vehicle.get_transform()
            yaw = transform.rotation.yaw
            # BEV view
            self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30), carla.Rotation(pitch=-90))) # original z=70
            # BEV with rotation
            # self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=40), carla.Rotation(pitch=-90, roll=yaw, yaw=0)))
            # First person view
            # spectator_transform = transform
            # spectator_transform.location.z += 3
            # spectator_transform.location.x -= 6*math.cos(math.radians(spectator_transform.rotation.yaw))
            # spectator_transform.location.y -= 6*math.sin(math.radians(spectator_transform.rotation.yaw))
            # self.spectator.set_transform(spectator_transform)                                                 
        return throttle_, steer_, reverse_
    def clean(self):
        """
        restore carla's settings and destroy all created actors when the program exits
        """


        # if self.recording:
        #     dir_name = str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())).split('.')[0]
        #     if not os.path.exists('image/%s' % dir_name):
        #         os.mkdir('image/%s' % dir_name)
            
        #     while(not self.image_queue.empty()):
        #         image = self.image_queue.get()
        #         print('image saving: ', image.frame_number)
        #         image.save_to_disk('image/%s/%08d.jpg' % (dir_name, image.frame_number))

        self.world.apply_settings(self.original_settings)
        logging.info('destroying actors')
        self.client.apply_batch([carla.command.DestroyActor(x)
                                for x in self.actor_list])
        self.client.apply_batch([carla.command.DestroyActor(x)
                                for x in self.other_vehicles_list])
        logging.info('done')
        

    def clean_all_actors(self):
        actors = self.world.get_actors().filter("*veh*")
        for actor in actors:
            actor.destroy()
        actors = self.world.get_actors().filter("*ped*")
        for actor in actors:
            actor.destroy()
        actors = self.world.get_actors().filter("*ai*")
        for actor in actors:
            actor.destroy()
        actors = self.world.get_actors().filter("*cam*")
        for actor in actors:
            actor.destroy()
        self.world.tick()


if __name__ == '__main__':
    log_level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    env = Env()
    env.reset()

    for _ in tqdm(range(500)):
        action = np.random.rand(3)
        action[0] = (action[0]-0.5) * 2
        action[1] = 1
        action[2] = 0

        env.step(action)
        if DISPLAY_METHOD == "pygame":
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                print(event.type)
                pygame.quit()
                exit()
    
    if DISPLAY_METHOD == "pygame":
        pygame.quit()
