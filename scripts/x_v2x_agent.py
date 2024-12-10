#!/usr/bin/env python

import os
import sys

try:
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'official'))
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'utils'))
except IndexError:
    print('Error: cannot add the path to sys')

import copy
import carla
import math
import numpy as np
import interpolate as itp
import carla_utils as ca_u
from enum import Enum
from collections import deque
from official.basic_agent import BasicAgent
from official.misc import get_trafficlight_trigger_location, is_within_distance
from official.global_route_planner import GlobalRoutePlanner
from official.local_planner import LocalPlanner

import matplotlib.pyplot as plt
import time

class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to other.
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class Xagent(BasicAgent):
    def __init__(self, env, model, dt=0.1, target_speed=30) -> None:
        '''
        vehicle: carla
        model: kinematic/dynamic model
        '''
        self._env = env
        self._vehicle = env.ego_vehicle
        self._model = model
        
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._base_min_distance = 3.0
        # self._waypoints_queue = deque(maxlen=100000)
        self._d_dist = 0.4
        self._sample_resolution = 2.0
        # state of lane_change
        self._a_opt = np.array([0.0]*self._model.horizon)
        self._delta_opt = np.array([0.0]*self._model.horizon)
        self._dt = dt

        self._next_states = None
        self._last_traffic_light = None
        self._last_traffic_waypoint = None

        self._model.solver_basis(Q=np.diag([10, 10, 10, 1, 0.1]), Rd=np.diag([1.0, 1000.0]))
        self.Q_origin = copy.deepcopy(self._model.Q)
        self._log_data = []
        self._simu_time = 0
        
        self._global_planner = GlobalRoutePlanner(self._map, self._sample_resolution)
        # Change parameters according to the dictionary
        opt_dict = {}
        opt_dict['target_speed'] = target_speed
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'max_brake' in opt_dict:
            self._max_steering = opt_dict['max_brake']
        self._local_planner = LocalPlanner(self._vehicle, opt_dict)
        self._route_assigned = False
        self.dist_move = 0.2
        self.dist_step = 1.5

    def plan_route(self, start_location, end_location):
        self._route = self.trace_route(start_location.location, end_location.location)
        for i in self._route:
            self._local_planner._waypoints_queue.append(i)

    def calc_ref_trajectory_in_T_step(self, node, ref_path, sp):
        """
        calc referent trajectory in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param node: current information
        :param ref_path: reference path: [x, y, yaw]
        :param sp: speed profile (designed speed strategy)
        :return: reference trajectory
        """
        T = self._model.horizon
        z_ref = np.zeros((4, T + 1))
        length = ref_path.length
        ind, _ = ref_path.nearest_index(node)

        z_ref[0, 0] = ref_path.cx[ind]
        z_ref[1, 0] = ref_path.cy[ind]
        z_ref[2, 0] = sp[ind]
        z_ref[3, 0] = ref_path.cyaw[ind]

        dist_move = copy.copy(self.dist_move)

        for i in range(1, T + 1):
            dist_move += self.dist_step *abs(self._model.get_v()) * self._dt
            ind_move = int(round(dist_move / self._d_dist))
            index = min(ind + ind_move, length - 1)

            z_ref[0, i] = ref_path.cx[index]
            z_ref[1, i] = ref_path.cy[index]
            z_ref[2, i] = sp[index]
            z_ref[3, i] = ref_path.cyaw[index]

        return z_ref, ind

    def rotate(self, x, y, theta, ratio=1.75):
        return np.array([(x * np.cos(theta) - y * np.sin(theta)) * ratio, (x * np.sin(theta) + y * np.cos(theta)) * ratio])
    
    def lat_dis_wp_ev(self, wp, ev):
        '''
        calculate the lateral distance between the waypoint and the ego vehicle
        '''
        wp_loc = np.array([wp.transform.location.x, wp.transform.location.y])
        ev_loc = np.array([ev.get_location().x, ev.get_location().y])
        wp_yaw = wp.transform.rotation.yaw
        wp_loc = self.rotate(wp_loc[0], wp_loc[1], np.deg2rad(wp_yaw))
        ev_loc = self.rotate(ev_loc[0], ev_loc[1], np.deg2rad(wp_yaw))
        return np.abs(wp_loc[1] - ev_loc[1])
    
    def run_step(self,lv=None):
        self._simu_time += self._dt
        state, height = self._model.get_state_carla() # return the car's state and height
        current_state = np.array(ca_u.carla_vector_to_rh_vector(state[0:2], state[2], state[3:]))

        # - Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vehicle_speed = self._model.get_v()
        self._min_distance = self._base_min_distance + 0.5 * vehicle_speed

        # - Get waypoints
        if len(self._local_planner._waypoints_queue) == 0:
            raise Exception("No waypoints to follow")
        else:
            carla_wp, _ = np.array(self._local_planner._waypoints_queue).T
            waypoints = []
            v = math.sqrt(current_state[3]**2+current_state[4]**2)
            waypoints.append(
                [current_state[0], current_state[1], v, current_state[2]])
            cnt = 0

            # delete the same waypoints to solve the problem of spline interpolation(NaN)
            last_state = None
            for wp in carla_wp:
                if cnt > 10:
                    break
                cnt += 1
                t = wp.transform
                ref_state = ca_u.carla_vector_to_rh_vector(
                    [t.location.x, t.location.y], t.rotation.yaw)
                if last_state is not None:
                    if (ref_state[0]-last_state[0])**2+(ref_state[1]-last_state[1])**2 < 0.001:
                        # print('same waypoints!!! continue')
                        continue
                waypoints.append([ref_state[0], ref_state[1],
                                 self._model.target_v, ref_state[2]])
                last_state = [ref_state[0],ref_state[1]]

            waypoints = np.array(waypoints).T

        num_waypoint_removed = 0

        for waypoint, _ in self._local_planner._waypoints_queue:
            if len(self._local_planner._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                moved_wp = self._local_planner._waypoints_queue.popleft()
            if veh_location.distance(self._local_planner._waypoints_queue[0][0].transform.location) > 10:
                self._local_planner._waypoints_queue.appendleft(moved_wp)

        # if the distance from the vehicle to the first waypoint is farther than 10 m, add the last moved waypoint to the queue
        
        # - Interplote the waypoints
        # t_0 = time.time()
        cx, cy, cyaw, ck, s = itp.calc_spline_course_carla(
            waypoints[0], waypoints[1], waypoints[3][0], ds=self._d_dist)
        sp = itp.calc_speed_profile(cx, cy, cyaw, self._model.target_v)
        # print('Interpolation Time cost: ', time.time()-t_0)

        ref_path = itp.PATH(cx, cy, cyaw, ck)
        z_ref, target_ind = self.calc_ref_trajectory_in_T_step(
            [current_state[0], current_state[1], v, current_state[2]], ref_path, sp)
        ref_traj = np.array([z_ref[0], z_ref[1], z_ref[3], z_ref[2], [
                            0]*len(z_ref[0]), [0]*len(z_ref[0])])[:, :self._model.horizon]

        if self._next_states is None:
            self._next_states = np.zeros(
                (self._model.n_states, self._model.horizon+1)).T
        
        cur_v = self._model.get_v()
        self._next_states[:,3] = cur_v
        current_state[3:] = self._next_states[0][3:]
        u0 = np.array([self._a_opt, self._delta_opt]).reshape(-1, 2).T

        if self._next_states is None:
            self._next_states = np.zeros(
                (self._model.n_states, self._model.horizon+1)).T

        # current_state[3:] = self._next_states[1][3:]
        # u0 = np.array([self._a_opt, self._delta_opt] *
        #               self._model.horizon).reshape(-1, 2).T

        '''
        For debug the obstacle center
        '''
        # obc = [current_state[0], current_state[1], current_state[2]]
        # center1, center2 = self._model.get_obs_centers(obc, carla=True)
        # tmp1_location = carla.Location(x=center1[0], y=center1[1], z=2)
        # tmp2_location = carla.Location(x=center2[0], y=center2[1], z=2)
        # self._world.debug.draw_point(tmp1_location, size=0.5,  life_time=0.1)
        # self._world.debug.draw_point(tmp2_location, size=0.5,  life_time=0.1)
    
        # - Obstacle constraints
        obs, heights, obs_infer, vehicle_sizes = self._env.get_obs(current_state)
        # print('vehicle_sizes:', vehicle_sizes)
        obs_infer = list(obs_infer)
        for i in range(len(obs_infer)):
            obs_infer[i] = obs_infer[i].tolist()
        temp_obs = copy.copy(obs)
        if len(temp_obs) > 0:
            for i in range(len(temp_obs)):
                centers = self._model.get_obs_centers(temp_obs[i], carla=True)
                # self._world.debug.draw_point(carla.Location(x=temp_obs[i][0], y=temp_obs[i][1], z=2), size=0.2,  life_time=0.1)
                # self._world.debug.draw_point(carla.Location(x=centers[0][0], y=-centers[0][1], z=3), size=0.2,  life_time=0.1)
                # self._world.debug.draw_point(carla.Location(x=centers[1][0], y=-centers[1][1], z=3), size=0.2,  life_time=0.1)
                bb = carla.BoundingBox(carla.Location(x=temp_obs[i][0], y=-temp_obs[i][1], z=heights[i]+1.0),\
                                       carla.Vector3D(vehicle_sizes[i][0], vehicle_sizes[i][1], vehicle_sizes[i][2]))
                self._world.debug.draw_box(bb, carla.Rotation(yaw=np.degrees(-temp_obs[i][2])), thickness=0.03, life_time=2*self._dt)
                for j in range(len(obs_infer[i])):
                    self._world.debug.draw_point(carla.Location(x=obs_infer[i][j][0], y=-obs_infer[i][j][1], z=heights[i]+1.5), size=0.06,  life_time=2*self._dt)
                # self._world.debug.draw_point(carla.Location(x=centers[0][0], y=-centers[0][1], z=heights[i]), size=0.2,  life_time=2*self._dt)

        # - Pedestrian constraints
        pedes, pede_heights, pedes_infer = self._env.get_pedes(current_state)
        pedes_infer = list(pedes_infer)
        for i in range(len(pedes_infer)):
            pedes_infer[i] = pedes_infer[i].tolist()
        temp_pedes = copy.copy(pedes)
        if len(temp_pedes) > 0:
            for i in range(len(temp_pedes)):
                centers = self._model.get_obs_centers(temp_pedes[i], carla=True)
                # self._world.debug.draw_point(carla.Location(x=temp_pedes[i][0], y=temp_pedes[i][1], z=2), size=0.2,  life_time=0.1)
                # self._world.debug.draw_point(carla.Location(x=centers[0][0], y=-centers[0][1], z=3), size=0.2,  life_time=0.1)
                # self._world.debug.draw_point(carla.Location(x=centers[1][0], y=-centers[1][1], z=3), size=0.2,  life_time=0.1)
                bb = carla.BoundingBox(carla.Location(x=temp_pedes[i][0], y=-temp_pedes[i][1], z=pede_heights[i]),\
                                       carla.Vector3D(0.6, 0.6, 1.8))
                self._world.debug.draw_box(bb, carla.Rotation(yaw=np.degrees(-temp_pedes[i][2])), thickness=0.03, life_time=2*self._dt)
                for j in range(len(pedes_infer[i])):
                    self._world.debug.draw_point(carla.Location(x=pedes_infer[i][j][0], y=-pedes_infer[i][j][1], z=pede_heights[i]+1.5), size=0.06,  life_time=2*self._dt)
                # self._world.debug.draw_point(carla.Location(x=centers[0][0], y=-centers[0][1], z=heights[i]), size=0.2,  life_time=2*self._dt)
        apf_obs = apf_pede = apf_nc_road = apf_c_road = apf_traffic = apf_ttc = 0

        self._model.solver_add_cost()
        
   
        # - Lane constraints
        # left_lane_type:  wp.left_lane_marking.type
        # right_lane_type: wp.right_lane_marking.type
        lane_location = self._vehicle.get_location()
        lane_location.z += 2
        # ego_location
        # self._world.debug.draw_point(lane_location, size=0.2,  life_time=0.05)
        wp = self._map.get_waypoint(lane_location, project_to_road=True)
        # wapoint location
        tmp_location = wp.transform.location
        tmp_location.z += 2
        # self._world.debug.draw_point(tmp_location, size=0.2, color=carla.Color(r=0,g=200,b=0,a=0), life_time=self._dt)

        # - Transfer to right coordinate from left coordinate
        wp_yaw = -np.radians(wp.transform.rotation.yaw)
        wp_y = np.sin(-wp_yaw)*wp.transform.location.x + np.cos(-wp_yaw)*-wp.transform.location.y  # rotate along the reverce direction of wp_yaw
        wp_x = np.cos(-wp_yaw)*wp.transform.location.x - np.sin(-wp_yaw)*-wp.transform.location.y
        left_bound = wp_y + 1.75
        right_bound = wp_y - 1.75
        
        lcrossable_list = [carla.LaneMarkingType.Broken]
        rcrossable_list = [carla.LaneMarkingType.Broken]
        lncrossable_list = [carla.LaneMarkingType.Solid, carla.LaneMarkingType.BrokenSolid, carla.LaneMarkingType.SolidSolid , carla.LaneMarkingType.BrokenBroken, carla.LaneMarkingType.SolidBroken]
        rncrossable_list = [carla.LaneMarkingType.Solid, carla.LaneMarkingType.SolidBroken, carla.LaneMarkingType.SolidSolid, carla.LaneMarkingType.BrokenBroken, carla.LaneMarkingType.BrokenSolid]
        # lc_bool = lnc_bool = rc_bool = rnc_bool = False
        self._model.Q = self.Q_origin
        vpf_offset = 0.25
        
        # if the vehicle will turn left or right, set nc_road_pf
        if self._local_planner._waypoints_queue[0][1].value == 2 or self._local_planner._waypoints_queue[3][1].value == 2:
            # if rhe vehicle has crossed the road lane, set ordinary PF
            if self.lat_dis_wp_ev(self._local_planner._waypoints_queue[1][0], self._env.ego_vehicle) < 1.7:
                self._model.solver_add_nc_road_pf([[left_bound, 1]], wp_yaw, carla=True)
                apf_nc_road += self._model.nc_road_pf([[left_bound, 1]], ref_traj, wp_yaw)
            elif wp.left_lane_marking.type in lcrossable_list:
                # lc_bool = True
                # pass
                self._model.solver_add_c_road_pf([left_bound], wp_yaw, carla=True)
                apf_c_road += self._model.c_road_pf([left_bound], ref_traj, wp_yaw)
            elif wp.left_lane_marking.type in lncrossable_list:
                # lnc_bool = True
                self._model.solver_add_nc_road_pf([[left_bound, 1]], wp_yaw, carla=True)
                apf_nc_road += self._model.nc_road_pf([[left_bound, 1]], ref_traj, wp_yaw)
            elif wp.left_lane_marking.type == carla.LaneMarkingType.NONE:
                # get the waypoint the vehicle is following
                waypoint_now = self._local_planner._waypoints_queue[0][0]
                waypoint_now = [waypoint_now.transform.location.x, waypoint_now.transform.location.y, waypoint_now.transform.rotation.yaw]
                # draw the waypoint
                self._world.debug.draw_point(carla.Location(x=waypoint_now[0], y=waypoint_now[1], z=2), size=0.1,  life_time=0.1, color=carla.Color(0,0,255))
                wp_ref = carla.Transform(carla.Location(x=waypoint_now[0], y=waypoint_now[1], z=0.05), carla.Rotation(yaw=np.degrees(waypoint_now[2])))
                wp_ref_yaw = -np.radians(wp_ref.rotation.yaw)
                wp_y = np.sin(-wp_ref_yaw)*wp_ref.location.x + np.cos(-wp_ref_yaw)*-wp_ref.location.y  # rotate along the reverce direction of wp_yaw
                wp_x = np.cos(-wp_ref_yaw)*wp_ref.location.x - np.sin(-wp_ref_yaw)*-wp_ref.location.y
                left_bound = wp_y + 1.75
                right_bound = wp_y - 1.75
                # increase the position tracking cost
                self._model.Q[0, 0] = 10
                self._model.Q[1, 1] = 10
                # set a relaxed virtual PF for the unknown lane
                self._model.solver_add_nc_road_pf([[left_bound+vpf_offset, -1]], wp_yaw, carla=True)
                apf_nc_road += self._model.nc_road_pf([[left_bound+vpf_offset , -1]], ref_traj, wp_yaw)
        elif wp.left_lane_marking.type in lcrossable_list:
            # lc_bool = True
            # pass
            self._model.solver_add_c_road_pf([left_bound], wp_yaw, carla=True)
            apf_c_road += self._model.c_road_pf([left_bound], ref_traj, wp_yaw)
        elif wp.left_lane_marking.type in lncrossable_list:
            # lnc_bool = True
            self._model.solver_add_nc_road_pf([[left_bound, 1]], wp_yaw, carla=True)
            apf_nc_road += self._model.nc_road_pf([[left_bound, 1]], ref_traj, wp_yaw)
        # Don't need to add the PF for the unknown lane if there is not a intersection
        # elif wp.left_lane_marking.type == carla.LaneMarkingType.NONE:
        #     # get the waypoint the vehicle is following
        #     waypoint_now = self._waypoints_queue[0][0]
        #     waypoint_now = [waypoint_now.transform.location.x, waypoint_now.transform.location.y, waypoint_now.transform.rotation.yaw]
        #     # draw the waypoint
        #     self._world.debug.draw_point(carla.Location(x=waypoint_now[0], y=waypoint_now[1], z=2), size=0.2,  life_time=0.1)
        #     wp_ref = carla.Transform(carla.Location(x=waypoint_now[0], y=waypoint_now[1], z=0.05), carla.Rotation(yaw=np.degrees(waypoint_now[2])))
        #     wp_ref_yaw = -np.radians(wp_ref.rotation.yaw)
        #     wp_y = np.sin(-wp_ref_yaw)*wp_ref.location.x + np.cos(-wp_ref_yaw)*-wp_ref.location.y  # rotate along the reverce direction of wp_yaw
        #     wp_x = np.cos(-wp_ref_yaw)*wp_ref.location.x - np.sin(-wp_ref_yaw)*-wp_ref.location.y
        #     left_bound = wp_y + 1.75
        #     right_bound = wp_y - 1.75
        #     # increase the position tracking cost
        #     self._model.Q[0, 0] = 10
        #     self._model.Q[1, 1] = 10
        #     # set a relaxed virtual PF for the unknown lane
        #     self._model.solver_add_nc_road_pf([[left_bound+vpf_offset, -1]], wp_yaw, carla=True)
        #     apf_nc_road += self._model.nc_road_pf([[left_bound+vpf_offset, -1]], ref_traj, wp_yaw)
        # if the vehicle will turn left or right, set nc_road_pf
        if self._local_planner._waypoints_queue[0][1].value == 1 or self._local_planner._waypoints_queue[3][1].value == 1:
            # if the vehicle has crossed the road lane, set ordinary PF
            if self.lat_dis_wp_ev(self._local_planner._waypoints_queue[0][0], self._env.ego_vehicle) < 1.70:
                self._model.solver_add_nc_road_pf([[right_bound, -1]], wp_yaw, carla=True)
                apf_nc_road += self._model.nc_road_pf([[right_bound, -1]], ref_traj, wp_yaw)
            elif wp.right_lane_marking.type in rcrossable_list:
            # rc_bool = True
                # pass
                self._model.solver_add_c_road_pf([right_bound], wp_yaw, carla=True)
                apf_c_road += self._model.c_road_pf([right_bound], ref_traj, wp_yaw)
            elif wp.right_lane_marking.type in rncrossable_list:
                # rnc_bool = True
                self._model.solver_add_nc_road_pf([[right_bound, -1]], wp_yaw, carla=True)
                apf_nc_road += self._model.nc_road_pf([[right_bound, -1]], ref_traj, wp_yaw)
            elif wp.right_lane_marking.type == carla.LaneMarkingType.NONE:
                # get the waypoint the vehicle is following
                # waypoint_now = ref_traj[:,0]
                waypoint_now = self._local_planner._waypoints_queue[0][0]
                waypoint_now = [waypoint_now.transform.location.x, waypoint_now.transform.location.y, waypoint_now.transform.rotation.yaw]
                # draw the waypoint with blue color
                self._world.debug.draw_point(carla.Location(x=waypoint_now[0], y=waypoint_now[1], z=2), size=0.1,  life_time=0.1, color=carla.Color(0,0,255))
                wp_ref = carla.Transform(carla.Location(x=waypoint_now[0], y=waypoint_now[1], z=0.05), carla.Rotation(yaw=np.degrees(waypoint_now[2])))
                wp_ref_yaw = -np.radians(wp_ref.rotation.yaw)
                wp_y = np.sin(-wp_ref_yaw)*wp_ref.location.x + np.cos(-wp_ref_yaw)*-wp_ref.location.y  # rotate along the reverce direction of wp_yaw
                wp_x = np.cos(-wp_ref_yaw)*wp_ref.location.x - np.sin(-wp_ref_yaw)*-wp_ref.location.y
                left_bound = wp_y + 1.75
                right_bound = wp_y - 1.75
                # increase the position tracking cost
                self._model.Q[0, 0] = 10
                self._model.Q[1, 1] = 10
                # set a relaxed virtual PF for the unknown lane
                self._model.solver_add_nc_road_pf([[right_bound-vpf_offset, -1]], wp_yaw, carla=True)
                apf_nc_road += self._model.nc_road_pf([[right_bound-vpf_offset , -1]], ref_traj, wp_yaw)

        elif wp.right_lane_marking.type in rcrossable_list:
            # rc_bool = True
            # pass
            self._model.solver_add_c_road_pf([right_bound], wp_yaw, carla=True)
            apf_c_road += self._model.c_road_pf([right_bound], ref_traj, wp_yaw)
        elif wp.right_lane_marking.type in rncrossable_list:
            # rnc_bool = True
            self._model.solver_add_nc_road_pf([[right_bound, -1]], wp_yaw, carla=True)
            apf_nc_road += self._model.nc_road_pf([[right_bound, -1]], ref_traj, wp_yaw)
        # Don't need to add the PF for the unknown lane if there is not a intersection
        # elif wp.right_lane_marking.type == carla.LaneMarkingType.NONE:
        #     # get the waypoint the vehicle is following
        #     # waypoint_now = ref_traj[:,0]
        #     waypoint_now = self._waypoints_queue[0][0]
        #     waypoint_now = [waypoint_now.transform.location.x, waypoint_now.transform.location.y, waypoint_now.transform.rotation.yaw]
        #     # draw the waypoint
        #     self._world.debug.draw_point(carla.Location(x=waypoint_now[0], y=waypoint_now[1], z=2), size=0.2,  life_time=0.1)
        #     wp_ref = carla.Transform(carla.Location(x=waypoint_now[0], y=waypoint_now[1], z=0.05), carla.Rotation(yaw=np.degrees(waypoint_now[2])))
        #     wp_ref_yaw = -np.radians(wp_ref.rotation.yaw)
        #     wp_y = np.sin(-wp_ref_yaw)*wp_ref.location.x + np.cos(-wp_ref_yaw)*-wp_ref.location.y  # rotate along the reverce direction of wp_yaw
        #     wp_x = np.cos(-wp_ref_yaw)*wp_ref.location.x - np.sin(-wp_ref_yaw)*-wp_ref.location.y
        #     left_bound = wp_y + 1.75
        #     right_bound = wp_y - 1.75
        #     # increase the position tracking cost
        #     self._model.Q[0, 0] = 10
        #     self._model.Q[1, 1] = 10
        #     # set a relaxed virtual PF for the unknown lane
        #     self._model.solver_add_nc_road_pf([[right_bound-vpf_offset, -1]], wp_yaw, carla=True)
        #     apf_nc_road += self._model.nc_road_pf([[right_bound-vpf_offset , -1]], ref_traj, wp_yaw)

        # add overtaking velocity dynamic penalty
        # wp_ref = [self._waypoints_queue[0][0].transform.location.x, self._waypoints_queue[0][0].transform.location.y, self._waypoints_queue[0][0].transform.rotation.yaw]
        # wp_ref_yaw = -np.radians(wp_ref[2])
        # wp_y = np.sin(-wp_ref_yaw)*wp_ref[0] + np.cos(-wp_ref_yaw)*-wp_ref[1]  # rotate along the reverce direction of wp_yaw
        # ev_y = np.sin(-wp_ref_yaw)*current_state[0] + np.cos(-wp_ref_yaw)*-current_state[1]
        # if wp_y - ev_y > 1.75:
        #     self._model.Q[3, 3] = 10
        # else:
        #     self._model.Q[3, 3] = self.Q_origin[3,3]

        # DEBUG
        # ego_y = np.sin(-wp_yaw)*lane_location.x + np.cos(-wp_yaw)*-lane_location.y
        # print('left_lane_type: %10s, l_dis: %4.2f, lc: %r, lnc: %r; right_lane_type: %10s, r_dis: %4.2f, vel: %4.2f, rc: %r, rnc: %r' 
        #       % (wp.left_lane_marking.type, left_bound-ego_y, lc_bool, lnc_bool,  wp.right_lane_marking.type, right_bound-ego_y, cur_v, rc_bool, rnc_bool))

        # - Traffic light constraints
        tl_bool, tl_waypoint = self.check_traffic_light()
        if tl_bool:
            tl_x = np.cos(-wp_yaw)*tl_waypoint.transform.location.x - np.sin(-wp_yaw)*-tl_waypoint.transform.location.y
            self._model.solver_add_single_tr_lgt_pf_carla(wp_y, wp_yaw, tl_x)
            apf_traffic += self._model.traffic_pf(ref_traj, wp_y, wp_yaw, tl_x)
        
        # - Solve the MPC problem
        # tick = time.time()
        # add the cost of the SVs' apf
        
        if len(obs) > 0:
            self._model.solver_add_soft_obs(obs_infer, sizes=vehicle_sizes,carla=True)
        # self._model.solver_add_hard_obs(obs, carla=True)
        # time_1 = time.time()-tick
        
        # added in revisoin version 1.0
        if len(pedes) > 0:
            self._model.solver_add_soft_obs(pedes_infer, carla=True, pede=True)

        if lv is not None:
            self._model.solver_add_soft_ttc(lv)
            apf_ttc += self._model.soft_ttc_apf(lv, ref_traj)
        
        # Get the cost of vehicle apf
        apf_obs += self._model.soft_obs_apf(obs, ref_traj)
        # Get the cost of pedes apf
        apf_pede += self._model.soft_obs_apf(pedes, ref_traj, pede=True)

        self._model.solver_add_bounds()  #  u00 = u0[:,0])
        
        tick = time.time()
        state = self._model.solve_MPC(
            ref_traj.T, current_state, self._next_states, u0)
        time_2 = time.time()-tick
        # print('Optimization Solving Time cost: ', time_2)
        self._env.computing_times.append(time_2)
        
        ca_u.draw_planned_trj(self._world, ref_traj[:2, :].T, height+0.5)
        ca_u.draw_planned_trj(self._world, state[2][:, :2], height+0.5, color=(0, 233, 222))

        self._next_states = state[2]

        next_state = state[2][1]
        self._a_opt = state[0]
        self._delta_opt = state[1]
        
        next_state = self._model.predict(current_state, (self._a_opt[0], self._delta_opt[0]))
        self._model.set_state(next_state)

        dist_error = math.hypot(
            next_state[0] - ref_traj[0, 1], next_state[1] - ref_traj[1, 1] - self.dist_move)
        yaw_error = abs(next_state[2] - ref_traj[2, 1])
        vel_error = abs(state[2][0][3] - ref_traj[3, 0])
        acc = self._a_opt[0] # carla_acc # self._a_opt[0]
        steer = self._delta_opt[0] # carla_steer # self._delta_opt[0]
        
        cost_time = state[-1]
        # print([self._simu_time, dist_error, yaw_error, vel_error, acc, steer, apf_obs, apf_nc_road, apf_c_road, apf_traffic, cost_time])

        self._log_data.append([self._simu_time, dist_error, yaw_error, vel_error, acc, steer, apf_obs, apf_ttc, apf_pede, apf_nc_road, apf_c_road, apf_traffic, cost_time])
        return self._a_opt[0], self._delta_opt[0], (next_state, height+0.05)

    def trace_route(self, start_location, end_location):
        return self._global_planner.trace_route(start_location, end_location)
    
    def check_traffic_light(self, traffic_distance=15.0):
        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
                self._last_traffic_waypoint = None
            else:
                return (True, self._last_traffic_waypoint)

        traffic_light, object_waypoint = self.get_traffic_light(traffic_distance, 'opposite')
        if traffic_light:
            self._last_traffic_light = traffic_light
            self._last_traffic_waypoint = object_waypoint
            return (True, object_waypoint)
        else:
            return (False, None)
    
    def get_traffic_light(self, traffic_distance=5.0, type='closest'):
        '''
        Get the red traffic light with different types
        Parameters:
            lights_list: list of traffic lights
            traffic_distance: the distance to the traffic light
            type: 'closest' or 'opposite'
        '''
        lights_list = self._world.get_actors().filter('*traffic_light*')

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        if type == 'closest':
            min_dist = 100000
            closest_light = None
            closest_waypoint = None

        for traffic_light in lights_list:
            if type == 'opposite':
                object_location = get_trafficlight_trigger_location(traffic_light)
                object_waypoint = self._map.get_waypoint(object_location)

                if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                    continue

                ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
                wp_dir = object_waypoint.transform.get_forward_vector()
                dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

                if traffic_light.state != carla.TrafficLightState.Red:
                    continue

                if dot_ve_wp < 0:
                    continue
                
                if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), traffic_distance, [0, 90]):
                    return traffic_light, object_waypoint
            elif type == 'closest':
                if traffic_light.state != carla.TrafficLightState.Red:
                    continue

                object_location = get_trafficlight_trigger_location(traffic_light)
                object_waypoint = self._map.get_waypoint(object_location)

                if object_location.distance(ego_vehicle_location) < min_dist:
                    min_dist = object_location.distance(ego_vehicle_location)
                    closest_light = traffic_light
                    closest_waypoint = object_waypoint
        
        if type == 'closest':    
            return closest_light, closest_waypoint
        else:
            return None, None
