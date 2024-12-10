import sys
import carla
import pygame
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT, K_ESCAPE, KEYDOWN, QUIT
import time

def setup_pygame():
    pygame.init()
    pygame.display.set_caption('Carla Keyboard Control')
    width, height = 640, 480
    return pygame.display.set_mode((width, height))

def process_events():
    for event in pygame.event.get():
        if event.type == QUIT:
            return False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                return False
    return True

def main():
    actor_list = []
    cont_mode = 'teleop' # 'teleop' or 'set_location'

    try:
        # Initialize pygame and create a window
        scrween = setup_pygame()

        # Connect to the simulator
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        # Get the world
        world = client.get_world()
        if world.get_map().name != 'Carla/Maps/Town03':
            world = client.load_world('Town03')

        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        # Get the blueprint library
        blueprint_library = world.get_blueprint_library()

        # Spawn a vehicle
        vehicle_bp = blueprint_library.filter('model3')[0]
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)

        # Set the control parameters
        control = carla.VehicleControl()
        if cont_mode == 'teleop':
            while process_events():
                # Get keyboard input
                keys = pygame.key.get_pressed()

                # Set control parameters based on keyboard input
                if keys[K_UP]:
                    if control.reverse:
                        control.reverse = False
                    control.throttle = 1.0
                elif keys[K_DOWN]:
                    control.reverse = True
                    control.throttle = 1.0
                else:
                    control.throttle = 0.0

                if keys[K_LEFT]:
                    control.steer = -1.0
                elif keys[K_RIGHT]:
                    control.steer = 1.0
                else:
                    control.steer = 0.0

                # Apply the control to the vehicle
                vehicle.apply_control(control)
                world.tick()
                # sleep for 0.05 seconds
                # time.sleep(0.05)
                # print the vehicle's location and yaw angle
                print("Location: ", vehicle.get_location(), "Yaw: ", vehicle.get_transform().rotation.yaw)
                # set spectator to be above the position of the vehicle
                spectator = world.get_spectator()
                spectator.set_transform(carla.Transform(vehicle.get_location() + carla.Location(z=50), carla.Rotation(pitch=-90)))

        elif cont_mode == 'set_location':

            # control the vehicle with transform location and yaw angle
            # the x location transform is from 155 to 165 with 0.1 increment
            # start_candidates in Town03 = [(155,11.78,0), (-47,-192,0)]
            # start candidates in Town05 = [(-205,-95,0)]
            # end candidates in Town05 = [(6,-95,0)]
            x_vector = [-205 + 1.0 * i for i in range(200)]
            # the y location transform is always 11.78
            y_vector = [-95+3.5 for i in range(200)]
            # the yaw angle is always 0
            yaw_vector = [0 for i in range(200)]
            for i in range(200):
                time.sleep(0.1)
                # set the vehicle's location and yaw angle from x_vector, y_vector and yaw_vector
                vehicle.set_transform(carla.Transform(carla.Location(x_vector[i], y_vector[i], 0.02), carla.Rotation(0, 0, yaw_vector[i])))
                # set the spectator to be above the vehicle
                spectator = world.get_spectator()
                spectator.set_transform(carla.Transform(vehicle.get_location() + carla.Location(z=50), carla.Rotation(pitch=-90)))


    finally:
        print('Destroying actors')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
