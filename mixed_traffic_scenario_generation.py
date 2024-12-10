import logging
import random
import time

import carla
import numpy as np

from utils.carla_utils import spawn_vehicles_around_ego_vehicles


class MixedTrafficScenarioGeneration:
    def __init__(self):
        self.client, self.world = self.connect_to_server()
        self.set_synchronous_mode(self.world)
        # change the town to Town03
        map_name = 'Town03'
        if str(self.world.get_map).split('/')[-1][:-1] == map_name:
            logging.info('Already load map {}, skipped'.format(map_name))
        else:
            self.world = self.client.load_world(map_name)
        self.ped_IDs = []
        self.veh_IDs = []
    def connect_to_server(self):
        # Connect to the CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        return client, world

    def set_synchronous_mode(self, world, synchronous_mode=True, fixed_delta_seconds=0.05):
        settings = world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = fixed_delta_seconds
        world.apply_settings(settings)

    def pede_generation(self, world, init_state, des_state):
        # Create a pedestrian
        blueprint_library = world.get_blueprint_library()
        # randomly choose a pedestrian blueprint
        pedestrian_bp = random.choice(blueprint_library.filter('walker.pedestrian.*'))
        # Get possible spawn points for the pedestrian
        spawn_point = init_state
        pedestrian = world.spawn_actor(pedestrian_bp, spawn_point)
        self.ped_IDs.append(pedestrian)
        # Use controller.ai.walker to control the pedestrian
        pedestrian_controller = pedestrian.get_control()
        pedestrian_controller.speed = 1.0
        pedestrian_controller.direction.y = des_state.location.y - init_state.location.y
        pedestrian_controller.direction.x = des_state.location.x - init_state.location.x
        pedestrian_controller.use_lanes = False
        pedestrian_controller.aim_for_destination = True
        # Apply the controller to the pedestrian
        pedestrian.apply_control(pedestrian_controller)

        # attach an walker AI controller to the pedestrian
        pedestrian_ai_controller = world.spawn_actor(blueprint_library.find('controller.ai.walker'),
                                                        carla.Transform(), pedestrian)
        # set speed for the pedestrian AI controller
        # pedestrian_ai_controller.set_max_speed(2.0)
        pedestrian_ai_controller.start()
        pedestrian_ai_controller.go_to_location(des_state.location)


    def veh_generation(self, world, init_state, des_state):
        # Create a vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        spawn_point = init_state
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        self.veh_IDs.append(vehicle)
        # set the vehicle's destination
        # vehicle.set_destination(des_state)
        vehicle.set_autopilot(True)

if __name__ == '__main__':
    try:
        scenario = MixedTrafficScenarioGeneration()
        # destroy all the pedestrian and vehicle actors in the world
        for actor in scenario.world.get_actors():
            if actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker'):
                actor.destroy()
        pede_init_states = [[177.8,75.2,180], [152.4,43.3,-90]] # [x,y,yaw]
        pede_des_states = [[155.8,75.8,180], [152.1,70.34,-90]] # [x,y,yaw]

        for i in range(len(pede_init_states)):
            init_state = carla.Transform(carla.Location(x=pede_init_states[i][0], y=pede_init_states[i][1], z=0.5),
                                        carla.Rotation(yaw=pede_init_states[i][2]))
            des_state = carla.Transform(carla.Location(x=pede_init_states[i][0], y=pede_init_states[i][1], z=0.5),
                                        carla.Rotation(yaw=pede_init_states[i][2]))
            scenario.pede_generation(scenario.world, init_state, des_state)

        veh_init_states = [[170.7,94.3,-85]] # [x,y,yaw]
        veh_des_states = [[126.4,62.5,0]] # [x,y,yaw]
        
        for i in range(len(veh_init_states)):
            init_state = carla.Transform(carla.Location(x=veh_init_states[i][0], y=veh_init_states[i][1], z=0.5),
                                        carla.Rotation(yaw=veh_init_states[i][2]))
            des_state = carla.Transform(carla.Location(x=veh_init_states[i][0], y=veh_init_states[i][1], z=0.5),
                                        carla.Rotation(yaw=veh_init_states[i][2]))
            scenario.veh_generation(scenario.world, init_state, des_state)

        # # generate surrounding vehicles around the 1st ego vehicle
        # veh_spawnpoints = spawn_vehicles_around_ego_vehicles(scenario.veh_IDs[0], scenario.world)

        scenario.spectator = scenario.world.get_spectator()

        while True:
            scenario.world.tick()
            # time.sleep(0.05)
            # set the spectator to the overlook of the 1st pedestrian
            veh_location = scenario.veh_IDs[0].get_location()
            scenario.spectator.set_transform(carla.Transform(veh_location + carla.Location(z=30),
                                                                carla.Rotation(pitch=-90)))
            # if the position of the pedestrian is close to the destination, remove it
            if np.linalg.norm(np.array([veh_location.x, veh_location.y]) - np.array(veh_des_states[0][:2])) < 2:
                scenario.ped_IDs[0].destroy()
                scenario.ped_IDs.pop(0)
            # if there is no pedestrian, generate a new one
            if len(scenario.ped_IDs) == 0:
                break
    # destroy all the actors
    finally:
        for vehicle in scenario.veh_IDs:
            vehicle.destroy()
        for pedestrian in scenario.ped_IDs:
            pedestrian.destroy()
