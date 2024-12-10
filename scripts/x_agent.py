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
    pass

import random
import carla
import math
import numpy as np
import interpolate as itp
import carla_utils as ca_u
from enum import Enum
from collections import deque
from basic_agent import BasicAgent


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
    def __init__(self, env, model, dt=0.1) -> None:
        '''
        vehicle: carla
        model: kinematic/dynamic model
        '''
        self._env = env
        self._vehicle = env.ego_vehicle
        self._model = model
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._sampling_radius = 0.5  # for searching next point in conjuction
        self._base_min_distance = 7.0

        self._waypoints_queue = deque(maxlen=100)
        self._min_waypoint_queue_length = 50
        self._d_dist = 0.5
        # state of lane_change
        self._delta_opt, self._a_opt = 0.0, 0.0
        self._dt = dt

        self._next_states = None

        self._model.solver_basis(Q=np.diag([10, 10, 5, 10, 0.01]))
        self._log_data = []
        self._simu_time = 0
        self.getStraightLaneWaypoints()

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

        dist_move = 0.0

        for i in range(1, T + 1):
            dist_move += abs(self._model.get_v()) * self._dt

            ind_move = int(round(dist_move / self._d_dist))
            index = min(ind+i + ind_move, length - 1)

            z_ref[0, i] = ref_path.cx[index]
            z_ref[1, i] = ref_path.cy[index]
            z_ref[2, i] = sp[index]
            z_ref[3, i] = ref_path.cyaw[index]

        return z_ref, ind

    def run_step(self):
        self._simu_time += self._dt

        if len(self._waypoints_queue) < self._min_waypoint_queue_length:
            self._compute_next_waypoints(k=self._min_waypoint_queue_length)

        state, height = self._model.get_state_carla() # return the car's state and height
        current_state = np.array(ca_u.carla_vector_to_rh_vector(state[0:2], state[2], state[3:]))

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vehicle_speed = self._model.get_v()
        self._min_distance = self._base_min_distance + 0.5 * vehicle_speed

        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:
            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            else:
                min_distance = self._min_distance

            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # get waypoints
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            carla_wp, _ = np.array(self._waypoints_queue).T
            waypoints = []
            v = math.sqrt(current_state[3]**2+current_state[4]**2)
            waypoints.append(
                [current_state[0], current_state[1], v, current_state[2]])
            for wp in carla_wp:
                t = wp.transform
                ref_state = ca_u.carla_vector_to_rh_vector(
                    [t.location.x, t.location.y], [t.rotation.yaw])
                waypoints.append([ref_state[0], ref_state[1],
                                 self._model.target_v, ref_state[2]])

            waypoints = np.array(waypoints).T

        cx, cy, cyaw, ck, s = itp.calc_spline_course_carla(
            waypoints[0], waypoints[1], waypoints[3][0], ds=self._d_dist)
        sp = itp.calc_speed_profile(cx, cy, cyaw, self._model.target_v)

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
        u0 = np.array([self._a_opt, self._delta_opt] *
                      self._model.horizon).reshape(-1, 2).T

        obs = self._env.get_obs()
        self._model.solver_add_cost()
        self._model.solver_add_soft_obs(obs, 400)
        self._model.solver_add_bounds()

        state = self._model.solve_MPC(ref_traj.T, current_state, self._next_states, u0)
        

        ca_u.draw_planned_trj(self._world, ref_traj[:2, :].T, height+0.5)
        ca_u.draw_planned_trj(self._world, state[2][:, :2], height+0.5, color=(0, 233, 222))

        self._next_states = state[2]

        next_state = state[2][1]
        self._a_opt = state[0]
        self._delta_opt = state[1]
        
        next_state = self._model.predict(current_state, (self._a_opt, self._delta_opt))
        self._model.set_state(next_state)

        dist_error = math.hypot(
            next_state[0] - ref_traj[0, 1], next_state[1] - ref_traj[1, 1])
        yaw_error = abs(next_state[2] - ref_traj[2, 1])
        acc = state[0]
        steer = state[1]
        vel = state[2][0][3]

        cost_time = state[-1]
        print([self._simu_time, dist_error, yaw_error, acc, steer, vel, cost_time])
        self._log_data.append([self._simu_time, dist_error, yaw_error, acc, steer, vel, cost_time])
        return self._a_opt, self._delta_opt, (next_state, height+0.05)

    def getLeftLaneWaypoints(self):
        current_waypoint = self._world.get_map().get_waypoint(self._vehicle.get_location())
        try:
            left_turn = current_waypoint.left_lane_marking.lane_change
            left_lane = current_waypoint.get_left_lane().next(
                self._looking_ahead)[-1]
            # print(left_turn, left_lane.lane_type, current_waypoint.lane_id, current_waypoint.lane_id*left_lane.lane_id)
            if (left_turn == carla.LaneChange.Left or left_turn == carla.LaneChange.Both) \
                    and current_waypoint.lane_id*left_lane.lane_id > 0:

                self.target_waypoint, self.target_road_option = (
                    left_lane, RoadOption.CHANGELANELEFT)
                self._waypoints_queue.append(
                    (self.target_waypoint, self.target_road_option))
            assert not len(self._waypoints_queue) == 0
        except:
            self.getStraightLaneWaypoints()
        # self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def getRightLaneWaypoints(self):
        current_waypoint = self._world.get_map().get_waypoint(self._vehicle.get_location())
        try:
            right_turn = current_waypoint.right_lane_marking.lane_change
            right_lane = current_waypoint.get_right_lane().next(
                self._looking_ahead)[-1]
            if right_turn == carla.LaneChange.Both:
                self.target_waypoint, self.target_road_option = (
                    right_lane, RoadOption.CHANGELANERIGHT)
                self._waypoints_queue.append(
                    (self.target_waypoint, self.target_road_option))
            assert not len(self._waypoints_queue) == 0
        except:
            self.getStraightLaneWaypoints()
        # self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def getStraightLaneWaypoints(self):
        current_waypoint = self._world.get_map().get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (
            current_waypoint, RoadOption.LANEFOLLOW)
        self._waypoints_queue.append(
            (self.target_waypoint, self.target_road_option))

    def _retrieve_options(self, list_waypoints, current_waypoint):
        """
        Compute the type of connection between the current active waypoint and the multiple waypoints present in
        list_waypoints. The result is encoded as a list of RoadOption enums.

        :param list_waypoints: list with the possible target waypoints in case of multiple options
        :param current_waypoint: current active waypoint
        :return: list of RoadOption enums representing the type of connection from the active waypoint to each
                candidate in list_waypoints
        """
        options = []
        for next_waypoint in list_waypoints:
            # this is needed because something we are linking to
            # the beggining of an intersection, therefore the
            # variation in angle is small
            next_next_waypoint = next_waypoint.next(3.0)[0]
            link = self._compute_connection(
                current_waypoint, next_next_waypoint)
            options.append(link)

        return options

    def _compute_connection(self, current_waypoint, next_waypoint, threshold=35):
        """
        Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
        (next_waypoint).

        :param current_waypoint: active waypoint
        :param next_waypoint: target waypoint
        :return: the type of topological connection encoded as a RoadOption enum:
                RoadOption.STRAIGHT
                RoadOption.LEFT
                RoadOption.RIGHT
        """
        n = next_waypoint.transform.rotation.yaw
        n = n % 360.0

        c = current_waypoint.transform.rotation.yaw
        c = c % 360.0

        diff_angle = (n - c) % 180.0
        if diff_angle < threshold or diff_angle > (180 - threshold):
            return RoadOption.STRAIGHT
        elif diff_angle > 90.0:
            return RoadOption.LEFT
        else:
            return RoadOption.RIGHT

    def _compute_next_waypoints(self, k=1):
        """
        Add new waypoints to the trajectory queue.

        :param k: how many waypoints to compute
        :return:
        """
        # check we do not overflow the queue
        available_entries = self._waypoints_queue.maxlen - \
            len(self._waypoints_queue)
        k = min(available_entries, k)
        for _ in range(k):
            last_waypoint = self._waypoints_queue[-1][0]
            next_waypoints = list(last_waypoint.next(self._sampling_radius))

            if len(next_waypoints) == 0:
                break
            elif len(next_waypoints) == 1:
                # only one option available ==> lanefollowing
                next_waypoint = next_waypoints[0]
                road_option = RoadOption.LANEFOLLOW
            else:
                # random choice between the possible options
                road_options_list = self._retrieve_options(
                    next_waypoints, last_waypoint)
                road_option = random.choice(road_options_list)
                next_waypoint = next_waypoints[road_options_list.index(
                    road_option)]

            self._waypoints_queue.append((next_waypoint, road_option))
