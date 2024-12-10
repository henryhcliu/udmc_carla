#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module is named Unified Decision-Making and Control (UDMC) agent to control the ego vehicle. For more details, please refer to the following paper: 
"A Unified Decision-Making and Control Framework for Urban Autonomous Driving with Motion Prediction of Traffic Participants" by H. Liu, et al. from HKUST.
"""

from __future__ import print_function

import carla
# from agents.navigation.basic_agent import BasicAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from scripts.env import * # the support functions for the UDMC agent
from scripts.x_v2x_agent_enhanced_bd import Xagent # the wrapper for the UDMC agent
from scripts.official.behavior_agent import BehaviorAgent
from scripts.vehicle_obs import Vehicle # the core class for the UDMC algorithm
from scripts.official.local_planner import RoadOption

def get_entry_point():
    return 'UDMCAgent'

class UDMCAgent(AutonomousAgent):

    """
    UDMC autonomous agent to control the ego vehicle
    """

    _agent = None
    _route_assigned = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        self._route_assigned = False
        self._agent = None
        self._env = Env(display_method=None, recording=False, is_Leaderboard=True)
        self.navigation_agent = None

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},
        ] # This sensor is not required for the agent, but it is required for the leaderboard

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        if self.navigation_agent is None:
            self._env.ego_vehicle_search()
            if self._env.ego_vehicle is None:
                logging.error("No ego vehicle found in the scene!")
                exit()
            self.navigation_agent = BehaviorAgent(self._env.ego_vehicle, behavior='normal')
            self._env.reset()
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False
        vehicle_list_all = self._env.world.get_actors().filter('vehicle.*')
        def dist(v): return v.get_location().distance(ego_vehicle_loc)

        if not self._agent:
            dynamic_model = Vehicle(actor=self._env.ego_vehicle, horizon=20, target_v=20, max_iter=40)
            self._agent = Xagent(self._env, dynamic_model, dt=0.05)
            return control

        if not self._route_assigned:
            if self._global_plan:
                plan = []

                prev = None
                for transform, _ in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    if  prev:
                        route_segment = self._agent.trace_route(prev.transform.location, wp.transform.location)
                        plan.extend(route_segment)

                    prev = wp

                # add a waypoint (along the last waypoint with a distance of 10 m) at the end of the plan to avoid the vehicle stopping at the last waypoint
                # min_dist = 10
                # last_wp = plan[-1][0]
                # last_wp_loc = last_wp.transform.location
                # last_wp_rot = last_wp.transform.rotation
                # last_wp_loc.x += min_dist * np.cos(np.radians(last_wp_rot.yaw))
                # last_wp_loc.y += min_dist * np.sin(np.radians(last_wp_rot.yaw))
                # wp = CarlaDataProvider.get_map().get_waypoint(last_wp_loc)
                # route_segment = self._agent.trace_route(prev.transform.location, wp.transform.location)
                # plan.extend(route_segment)
            
                self._agent._local_planner.set_global_plan(plan)
                # add 10 extra waypoints (as the extention of the original route) to the plan
                for _ in range(30):
                    delta_dist = 3
                    last_wp = self._agent._local_planner._waypoints_queue[-1][0]
                    last_wp_loc = last_wp.transform.location
                    last_wp_rot = last_wp.transform.rotation
                    last_wp_loc.x += delta_dist * np.cos(np.radians(last_wp_rot.yaw))
                    last_wp_loc.y += delta_dist * np.sin(np.radians(last_wp_rot.yaw))
                    wp = CarlaDataProvider.get_map().get_waypoint(last_wp_loc)
                    wp_plan = [wp, RoadOption.LANEFOLLOW]
                    self._agent._local_planner._waypoints_queue.append(wp_plan)
                self._route_assigned = True
            # # get current transform of the ego vehicle
            # ego_vehicle_transform = self._env.ego_vehicle.get_transform()
            # # get the destination transform
            # destination = self._env.world.get_map().get_spawn_points()[0]
            # self._agent.plan_route()

        else:
            ego_vehicle_loc = self._env.ego_vehicle.get_location()
            vehicle_list = [v for v in vehicle_list_all if dist(v) < 45 and v.id != self.navigation_agent._vehicle.id]
            lv_state, lv, distance = self.navigation_agent._vehicle_obstacle_detected(vehicle_list, 15, up_angle_th=45)  
            new_state = self._agent.run_step(lv)
            carla_thr, carla_steer, carla_reverse = self._env.step(new_state, transform_mode=False)
            control.throttle = carla_thr
            control.steer = -0.7*carla_steer
            # print("steer: ", control.steer)
            control.reverse = carla_reverse

        return control
