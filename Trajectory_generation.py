#!/usr/bin/env python


from Optimization import OptPath_patient, OptPath_patient2
import math
import numpy as np
import random
from shapely.geometry import Polygon, Point


def define_obstacles(env):
    ''' This function is to define line segments for each obstacle to be used in the trajectory optimization
    and also obstacle polygons to be used for sampling start and end points that are out of obstacles.'''

    obstacles = []
    r_patient = 0.1 # Margin to avoid obstacles


    for obj in env.objects:
        m_box = [math.tan(obj.conf.z), math.tan(obj.conf.z + math.pi / 2), math.tan(obj.conf.z), math.tan(obj.conf.z + math.pi / 2)]
        dobs = math.sqrt(obj.length**2+obj.width**2)
        corners = np.asarray(obj.polygon.exterior.coords)
        b = [corners[0][1] - m_box[0] * corners[0][0], corners[1][1] - m_box[1] * corners[1][0],
                    corners[2][1] - m_box[2] * corners[2][0], corners[3][1] - m_box[3] * corners[3][0]]
        center_pose = [obj.conf.x, obj.conf.y]
        obstacles.append([m_box, b, center_pose, Polygon(corners)])

    for wall in env.walls:
        wall_c = [(wall[0][0]+wall[0][2])/2, (wall[0][1]+wall[0][3])/2]
        wall_d = [2*r_patient + np.sqrt((wall[0][0]-wall[0][2])**2+(wall[0][1]-wall[0][3])**2),2*r_patient + 0.6]
        wall_angle = np.arctan2((wall[0][1]-wall[0][3]), (wall[0][0]-wall[0][2])) + 0.001
        m_box = [math.tan(wall_angle), math.tan(wall_angle + math.pi / 2), math.tan(wall_angle), math.tan(wall_angle + math.pi / 2)]
        dobs = math.sqrt(wall_d[0]**2+wall_d[1]**2)
        corners = [[wall_c[0] - (wall_d[1]/2) * math.sin(wall_angle) - (wall_d[0]/2) * math.cos(wall_angle), wall_c[1] + (wall_d[1]/2) * math.cos(wall_angle) - (wall_d[0]/2) * math.sin(wall_angle)],
                   [wall_c[0] + (wall_d[1]/2) * math.sin(wall_angle) - (wall_d[0]/2) * math.cos(wall_angle), wall_c[1] - (wall_d[1]/2) * math.cos(wall_angle) - (wall_d[0]/2) * math.sin(wall_angle)],
                   [wall_c[0] + (wall_d[1]/2) * math.sin(wall_angle) + (wall_d[0]/2) * math.cos(wall_angle), wall_c[1] - (wall_d[1]/2) * math.cos(wall_angle) + (wall_d[0]/2) * math.sin(wall_angle)],
                   [wall_c[0] - (wall_d[1]/2) * math.sin(wall_angle) + (wall_d[0]/2) * math.cos(wall_angle), wall_c[1] + (wall_d[1]/2) * math.cos(wall_angle) + (wall_d[0]/2) * math.sin(wall_angle)]]
        b = [corners[0][1] - m_box[0] * corners[0][0], corners[1][1] - m_box[1] * corners[1][0],
                    corners[2][1] - m_box[2] * corners[2][0], corners[3][1] - m_box[3] * corners[3][0]]
        center_pose = [wall_c[0], wall_c[1]]
        obstacles.append([m_box, b, center_pose, Polygon(corners)])

    return obstacles

def sample_point(env, obj, obstacles):
    ''' This function samples a point around the target object. It can be a sitting zone for sittable furniture,
    a reaching zone for reachable objects such as bathroom sink, or just inside an area like main entrance door. '''

    x_min, x_max, y_min, y_max = [0, 10, 0, 10] # Sampling range
    found = False
    while not found:
        x = random.uniform(x_min,x_max)
        y = random.uniform(y_min,y_max)
        point = Point([x,y])

        # Check if the sampled point is in the sitting zone of the target object
        is_in_sample_zone = False
        if point.within(env.sample_zones[obj]):
            is_in_sample_zone = True

        # Check if the sampled point is out of all the obstacles in the room
        is_out_of_obstacle = True
        for obs in obstacles:
            if point.within(obs[3]):
                is_out_of_obstacle = False

        if is_in_sample_zone == True and is_out_of_obstacle == True:
            found = True
            point = [x,y, 0, 0, 0]

    return point

def is_near_sitting_object(state, env, obj):
    ''' This function determines if a point is near the sitting zone of a target object. '''

    is_near = False
    if obj in ['Bed', 'Chair', 'Toilet', 'Sofa', 'Couch']:
        if state.within(env.sample_zones[obj]):
            is_near = True
    return is_near

def generagte_trajectory(start, end, env, obstacles, v_max, w_max, num_points):
    ''' This is the main function that generates trajectories given a scenario. It samples the start and end point.
    Then, using optimization, we find an optimal path between these 2 points. Finally, for each point in the trajectory,
    we find a corresponding activity based on the distance to the target object. '''

    found = 0
    while found == 0:

        # Sample points near the start and end locations
        patient_s = sample_point(env, start, obstacles)
        patient_g = sample_point(env, end, obstacles)
        scenario = {'start': patient_s, 'end': patient_g, 'v_max': v_max, 'w_max': w_max}
        print("senario: ", scenario)

        # Find a trajectory between sampled points
        cost, predicted_patient_traj, status = OptPath_patient2(scenario['start'], scenario['end'],  [scenario['v_max'], scenario['w_max']] , obstacles, num_points, assistive_device=False)

        # If the optimization was successful, find the type of activity for each point on the trajectory and add it to the resturning path
        if status == 2 :
            found = 1
            traj = []
            print("Finding activity types...")
            for i in range(len(predicted_patient_traj)):
                if is_near_sitting_object(Point(predicted_patient_traj[i]), env, start) :
                    traj.append([predicted_patient_traj[i], 'sit_to_stand'])
                elif is_near_sitting_object(Point(predicted_patient_traj[i]), env, end):
                    traj.append([predicted_patient_traj[i], 'stand_to_sit'])
                else:
                    traj.append([predicted_patient_traj[i], 'walking'])
    return traj
