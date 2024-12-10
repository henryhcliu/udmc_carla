import carla
import math
import numpy as np

def draw_planned_trj(world, x_trj, car_z, color=(200,0,0)):
    color = carla.Color(r=color[0],g=color[1],b=color[2],a=0)
    length = x_trj.shape[0]
    xx = x_trj[:,0]
    yy = -x_trj[:,1]
    for i in range(1, length):
        begin = carla.Location(float(xx[i-1]), float(yy[i-1]), float(car_z+1))
        end = carla.Location(float(xx[i]), float(yy[i]), float(car_z+1))
        # world.debug.draw_line(begin=begin, end=end, thickness=0.08, color=color, life_time=0.1)

def carla_vector_to_rh_vector(position, yaw, velocity=None):
    """
    Convert a carla location to a right-hand system
    position: x, y, z
    yaw: yaw(degree)
    velocity: vx, vy, omega
    """
    x = position[0]
    y = -position[1]
    yaw = -np.radians(yaw)

    if velocity is not None:
        vx = velocity[0]
        vy = -velocity[1]
        omega = -velocity[2]
        
        return [x, y, yaw, vx, vy, omega]
    
    return [x, y, yaw, 0, 0, 0]

def spawn_vehicles_around_ego_vehicles(ego_vehicle_transform, max_dis, min_dis, spawn_points, numbers_of_vehicles):
    """
    parameters:
    ego_vehicle :: your target vehicle
    max_dis :: the distance max limitation between ego-vehicle and other free-vehicles
    min_dis :: the distance min limitation
    spawn_points :: the available spawn points in current map
    numbers_of_vehicles :: the number of free-vehicles around ego-vehicle that you need
    """
    np.random.shuffle(spawn_points) # shuffle all the spawn points
    ego_location = ego_vehicle_transform.location
    accessible_points = []
    for spawn_point in spawn_points:
        dis = math.sqrt((ego_location.x-spawn_point.location.x)**2 + (ego_location.y-spawn_point.location.y)**2)
        # it also can include z-coordinate,but it is unnecessary
        if dis < max_dis and dis > min_dis and (spawn_point not in accessible_points):
            accessible_points.append(spawn_point)

    transform_list = [] # keep the spawned vehicle in vehicle_list, because we need to link them with traffic_manager
    if len(accessible_points) < numbers_of_vehicles:
        # if your radius is relatively small,the satisfied points may be insufficient
        numbers_of_vehicles = len(accessible_points)

    for i in range(numbers_of_vehicles): # generate the free vehicle
        point = accessible_points[i]
        transform_list.append(point)

    print("spawned vehicles number: ", len(transform_list))
    return transform_list