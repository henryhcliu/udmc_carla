import carla
import random
import numpy as np
import math
import time
from enum import Enum
import os
import sys

try:
    sys.path.append(os.path.join(
                    os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))), 'official'))
except IndexError:
    pass

from official.basic_agent import BasicAgent
from official.local_planner import RoadOption
from official.behavior_types import Cautious, Aggressive, Normal
from official.controller import VehiclePIDController
from official.misc import get_speed, positive, is_within_distance, compute_distance

from collections import deque

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

class OthersAgent(BasicAgent):
    def __init__(self, vehicle, lane_change=False):
        super(OthersAgent, self).__init__(vehicle)
        self._keep_time_thresh = 10
        self._target_destination_tuple = None
        self._destination_lane = None
        self._target_destination_index = -1
        self._waypoints_queue = deque(maxlen=10000)
        self._base_min_distance = 3.0
        self._min_waypoint_queue_length = 100
        self._sampling_radius = 2.0
        self._lane_change = lane_change

        dt = 1.0 / 20.0
        args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.2, 'dt': dt}
        args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': dt}
        offset = 0
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                        args_lateral=args_lateral_dict,
                                        args_longitudinal=args_longitudinal_dict,
                                        offset=offset,
                                        max_throttle=self._max_throt,
                                        max_brake=self._max_brake,
                                        max_steering=self._max_steer)
        self._cur_time = time.time()
        self._speed = get_speed(self._vehicle)
        self._location = self._vehicle.get_transform().location
        # self._looking_ahead = int(2*self._speed)+5
        self.target_waypoint = self._location
        self.getStraightLaneWaypoints()

    def set_target_speed(self, speed):
        self._target_speed = speed

    def dist2Waypoint(self, waypoint):
        vehicle_transform = self._vehicle.get_transform()
        vehicle_x = vehicle_transform.location.x
        vehicle_y = vehicle_transform.location.y
        waypoint_x = waypoint.transform.location.x
        waypoint_y = waypoint.transform.location.y
        return math.sqrt((vehicle_x - waypoint_x)**2 + (vehicle_y - waypoint_y)**2)
    
    def run_step(self):
        self._speed = get_speed(self._vehicle) / 3.6
        self._looking_ahead = int(2*self._speed)+1
        # add probability to change lane
        if self._lane_change:
            lane_direct = random.randint(-10,10)
            if lane_direct != 0 and \
                time.time()-self._cur_time > self._keep_time_thresh and \
                not self.target_waypoint.is_junction:
                self._cur_time = time.time()
                self._waypoints_queue.clear()
                if lane_direct > 0: # left
                    self.getLeftLaneWaypoints()
                else:
                    self.getRightLaneWaypoints()

        if len(self._waypoints_queue) < self._min_waypoint_queue_length:
            self._compute_next_waypoints(k=self._min_waypoint_queue_length)

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vehicle_speed = self._speed
        self._min_distance = self._base_min_distance + 0.5 *vehicle_speed

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
                
        # Get the target waypoint and move using the PID controllers. Stop if no target waypoint
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

            return control

    def getLeftLaneWaypoints(self):
        current_waypoint = self._world.get_map().get_waypoint(self._vehicle.get_location())
        try:
            left_turn = current_waypoint.left_lane_marking.lane_change
            left_lane = current_waypoint.get_left_lane().next(self._looking_ahead)[-1]
            # print(left_turn, left_lane.lane_type, current_waypoint.lane_id, current_waypoint.lane_id*left_lane.lane_id)
            if (left_turn == carla.LaneChange.Left or left_turn == carla.LaneChange.Both) \
                and current_waypoint.lane_id*left_lane.lane_id>0:

                self.target_waypoint, self.target_road_option = (left_lane, RoadOption.CHANGELANELEFT)
                self._waypoints_queue.append((self.target_waypoint, self.target_road_option))
            assert not len(self._waypoints_queue) == 0
        except:
            self.getStraightLaneWaypoints()
        # self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def getRightLaneWaypoints(self):
        current_waypoint = self._world.get_map().get_waypoint(self._vehicle.get_location())
        try:
            right_turn = current_waypoint.right_lane_marking.lane_change
            right_lane = current_waypoint.get_right_lane().next(self._looking_ahead)[-1]
            if right_turn == carla.LaneChange.Both:
                self.target_waypoint, self.target_road_option = (right_lane, RoadOption.CHANGELANERIGHT)
                self._waypoints_queue.append((self.target_waypoint, self.target_road_option))
            assert not len(self._waypoints_queue) == 0
        except:
            self.getStraightLaneWaypoints()
        # self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def getStraightLaneWaypoints(self):
        current_waypoint = self._world.get_map().get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

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
            link = self._compute_connection(current_waypoint, next_next_waypoint)
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
        available_entries = self._waypoints_queue.maxlen - len(self._waypoints_queue)
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

    