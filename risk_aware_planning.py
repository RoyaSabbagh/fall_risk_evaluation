#!/usr/bin/env python


from Optimization import OptPath_patient, OptPath_patient2
from Fall_risk_assesment import Environment_Image, FallRiskAssesment
import math
import cv2
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap




def sample_point(obj, sitting_zones, objects, obstacles):
    if obj in ['Chair-Patient', 'Bed', 'Chair-Visitor', 'Toilet', 'Sofa', 'Couch']:
        x_min, x_max, y_min, y_max = [0, 10, 0, 10]
        found = False
        l = 0.4
        zone = copy.deepcopy(sitting_zones[obj])
        if 1 in zone[5]:
            zone[4] += l
            zone[0] += float(l)/2*np.cos(zone[2])
            zone[1] += float(l)/2*np.sin(zone[2])
        if 2 in zone[5]:
            zone[3] += l
            zone[0] -= float(l)/2*np.sin(zone[2])
            zone[1] += float(l)/2*np.cos(zone[2])
        if 3 in zone[5]:
            zone[4] += l
            zone[0] -= float(l)/2*np.cos(zone[2])
            zone[1] -= float(l)/2*np.sin(zone[2])
        if 4 in zone[5]:
            zone[3] += l
            zone[0] += float(l)/2*np.sin(zone[2])
            zone[1] -= float(l)/2*np.cos(zone[2])
        corners_sitting = Polygon([[zone[0]-float(zone[4])/2*np.cos(zone[2])-float(zone[3])/2*np.sin(zone[2]),zone[1]-float(zone[4])/2*np.sin(zone[2])+float(zone[3])/2*np.cos(zone[2])],
                                    [zone[0]-float(zone[4])/2*np.cos(zone[2])+float(zone[3])/2*np.sin(zone[2]),zone[1]-float(zone[4])/2*np.sin(zone[2])-float(zone[3])/2*np.cos(zone[2])],
                                    [zone[0]+float(zone[4])/2*np.cos(zone[2])+float(zone[3])/2*np.sin(zone[2]),zone[1]+float(zone[4])/2*np.sin(zone[2])-float(zone[3])/2*np.cos(zone[2])],
                                    [zone[0]+float(zone[4])/2*np.cos(zone[2])-float(zone[3])/2*np.sin(zone[2]),zone[1]+float(zone[4])/2*np.sin(zone[2])+float(zone[3])/2*np.cos(zone[2])] ])
        # print(corners_sitting)
        while not found:
            x = random.uniform(x_min,x_max)
            y = random.uniform(y_min,y_max)
            point = Point([x,y])
            is_in_sitting_zone = False

            if point.within(corners_sitting):
                is_in_sitting_zone = True

            is_out_of_object = True
            for object in objects:
                if obj in object.name:
                    corners_object = Polygon([ [object.conf.x-object.width/2*np.cos(object.conf.z)-object.length/2*np.sin(object.conf.z),object.conf.y-object.width/2*np.sin(object.conf.z)+object.length/2*np.cos(object.conf.z)],
                                                [object.conf.x-object.width/2*np.cos(object.conf.z)+object.length/2*np.sin(object.conf.z),object.conf.y-object.width/2*np.sin(object.conf.z)-object.length/2*np.cos(object.conf.z)],
                                                [object.conf.x+object.width/2*np.cos(object.conf.z)+object.length/2*np.sin(object.conf.z),object.conf.y+object.width/2*np.sin(object.conf.z)-object.length/2*np.cos(object.conf.z)],
                                                [object.conf.x+object.width/2*np.cos(object.conf.z)-object.length/2*np.sin(object.conf.z),object.conf.y+object.width/2*np.sin(object.conf.z)+object.length/2*np.cos(object.conf.z)] ])
                    # print(corners_object)
                    if point.within(corners_object):
                        is_out_of_object = False

            is_out_of_obstacle = True
            for obs in obstacles:
                obstacle = []
                for i in range(len(obs[3])):
                    obstacle.append(obs[3][i])
                if point.within(Polygon(obstacle)):
                    is_out_of_obstacle = False

            if is_in_sitting_zone == True and is_out_of_object == True and is_out_of_obstacle == True:
                found = True
                point = [x,y, 0, 0, 0]

    elif obj in ['Sink-Bath']:
        x_min, x_max, y_min, y_max = [0, 10, 0, 10]
        found = False
        l = 0.5

        for object in env.objects:
            if "Sink-Bath" in object.name:
                zone = [object.conf.x, object.conf.y, object.conf.z, object.width, object.length, [1]] #Side Numbers: [P22-Inboard: 1, P22-Outboard:3, P22-Nested:3, A-K-Outboard:2, A-K-Inboard:4, S-B-outboard:2, J-M-outboard:4, J-C-Inboard:1, J-G-Inboard:1, B-L-Inboard:4, B-JH-Inboard:4, K-B-Inboard:2, K-B-Canted:2, Room-1:2, Room-2:1, Room-3:1, Room-4:3]
                if 1 in zone[5]:
                    zone[4] += l
                    zone[0] += l/2*np.cos(zone[2])
                    zone[1] += l/2*np.sin(zone[2])
                if 2 in zone[5]:
                    zone[3] += l
                    zone[0] -= l/2*np.sin(zone[2])
                    zone[1] += l/2*np.cos(zone[2])
                if 3 in zone[5]:
                    zone[4] += l
                    zone[0] -= l/2*np.cos(zone[2])
                    zone[1] -= l/2*np.sin(zone[2])
                if 4 in zone[5]:
                    zone[3] += l
                    zone[0] += l/2*np.sin(zone[2])
                    zone[1] -= l/2*np.cos(zone[2])

                while not found:
                    x = random.uniform(x_min,x_max)
                    y = random.uniform(y_min,y_max)
                    point = Point([x,y])
                    is_in_reaching_zone = False
                    corners_reaching = Polygon([[zone[0]-zone[4]/2*np.cos(zone[2])-zone[3]/2*np.sin(zone[2]),zone[1]-zone[4]/2*np.sin(zone[2])+zone[3]/2*np.cos(zone[2])],
                                                [zone[0]-zone[4]/2*np.cos(zone[2])+zone[3]/2*np.sin(zone[2]),zone[1]-zone[4]/2*np.sin(zone[2])-zone[3]/2*np.cos(zone[2])],
                                                [zone[0]+zone[4]/2*np.cos(zone[2])+zone[3]/2*np.sin(zone[2]),zone[1]+zone[4]/2*np.sin(zone[2])-zone[3]/2*np.cos(zone[2])],
                                                [zone[0]+zone[4]/2*np.cos(zone[2])-zone[3]/2*np.sin(zone[2]),zone[1]+zone[4]/2*np.sin(zone[2])+zone[3]/2*np.cos(zone[2])] ])
                    if point.within(corners_reaching):
                        is_in_reaching_zone = True

                    is_out_of_object = True
                    for object in objects:
                        if obj in object.name:
                            corners_object = Polygon([ [object.conf.x-object.width/2*np.cos(object.conf.z)-object.length/2*np.sin(object.conf.z),object.conf.y-object.width/2*np.sin(object.conf.z)+object.length/2*np.cos(object.conf.z)],
                                                        [object.conf.x-object.width/2*np.cos(object.conf.z)+object.length/2*np.sin(object.conf.z),object.conf.y-object.width/2*np.sin(object.conf.z)-object.length/2*np.cos(object.conf.z)],
                                                        [object.conf.x+object.width/2*np.cos(object.conf.z)+object.length/2*np.sin(object.conf.z),object.conf.y+object.width/2*np.sin(object.conf.z)-object.length/2*np.cos(object.conf.z)],
                                                        [object.conf.x+object.width/2*np.cos(object.conf.z)-object.length/2*np.sin(object.conf.z),object.conf.y+object.width/2*np.sin(object.conf.z)+object.length/2*np.cos(object.conf.z)] ])
                            if point.within(corners_object):
                                is_out_of_object = False

                    is_out_of_obstacle = True
                    for obs in obstacles:
                        obstacle = []
                        for i in range(len(obs[3])):
                            obstacle.append(obs[3][i])
                        if point.within(Polygon(obstacle)):
                            is_out_of_obstacle = False
                    # print(is_in_reaching_zone, is_out_of_object, is_out_of_obstacle)
                    if is_in_reaching_zone == True and is_out_of_object == True and is_out_of_obstacle == True:
                        found = True
                    point = [x,y, 0, 0, 0]

    else:
        x_min, x_max, y_min, y_max = [0, 10, 0, 10]
        found = False
        while not found:
            x = random.uniform(x_min,x_max)
            y = random.uniform(y_min,y_max)
            point = Point([x,y])
            is_in_object = False
            for door in env.doors:
                if "Door 11" or "Door 35" in door.name:
                    corners_door = Polygon([ [door.conf.x-door.width/2*np.cos(door.conf.z)-door.length/2*np.sin(door.conf.z),door.conf.y-door.width/2*np.sin(door.conf.z)+door.length/2*np.cos(door.conf.z)],
                                              [door.conf.x-door.width/2*np.cos(door.conf.z)+door.length/2*np.sin(door.conf.z),door.conf.y-door.width/2*np.sin(door.conf.z)-door.length/2*np.cos(door.conf.z)],
                                              [door.conf.x+door.width/2*np.cos(door.conf.z)+door.length/2*np.sin(door.conf.z),door.conf.y+door.width/2*np.sin(door.conf.z)-door.length/2*np.cos(door.conf.z)],
                                              [door.conf.x+door.width/2*np.cos(door.conf.z)-door.length/2*np.sin(door.conf.z),door.conf.y+door.width/2*np.sin(door.conf.z)+door.length/2*np.cos(door.conf.z)] ])
                    if point.within(corners_door):
                        is_in_object = True

            if is_in_object == True:
                found = True
                point = [x,y, 0, 0, 0]

    return point

def find_corners(x, y, phi, width, length):
    corners = []
    corners.append([x - (width/2) * math.sin(phi) - (length/2) * math.cos(phi), y + (width/2) * math.cos(phi) - (length/2) * math.sin(phi)])
    corners.append([x + (width/2) * math.sin(phi) - (length/2) * math.cos(phi), y - (width/2) * math.cos(phi) - (length/2) * math.sin(phi)])
    corners.append([x + (width/2) * math.sin(phi) + (length/2) * math.cos(phi), y - (width/2) * math.cos(phi) + (length/2) * math.sin(phi)])
    corners.append([x - (width/2) * math.sin(phi) + (length/2) * math.cos(phi), y + (width/2) * math.cos(phi) + (length/2) * math.sin(phi)])

    return corners

def define_obs(obss):
    obstacles = []
    r_patient = 0.1
    for obs in obss:
            m_box = [math.tan(obs[2]), math.tan(obs[2] + math.pi / 2), math.tan(obs[2]), math.tan(obs[2] + math.pi / 2)]
            dobs = math.sqrt(obs[3]**2+obs[4]**2)
            corners = find_corners(obs[0], obs[1], obs[2],2*r_patient + obs[4], 2*r_patient + obs[3])
            b = [corners[0][1] - m_box[0] * corners[0][0], corners[1][1] - m_box[1] * corners[1][0],
                        corners[2][1] - m_box[2] * corners[2][0], corners[3][1] - m_box[3] * corners[3][0]]
            center_pose = [obs[0], obs[1]]
            # print([m_box, b, center_pose, corners])
            obstacles.append([m_box, b, center_pose, corners])
    return obstacles

def is_near_sitting_object(state, sitting_zones, obj):
    is_near = False
    if obj in ['Bed', 'Chair', 'Toilet', 'Sofa', 'Couch']:
        obj_zone = sitting_zones[obj]
        zone = copy.deepcopy(obj_zone)
        zone[3] += 2
        zone[4] += 2
        corners = Polygon([[zone[0]-zone[4]/2*np.cos(zone[2])-zone[3]/2*np.sin(zone[2]),zone[1]-zone[4]/2*np.sin(zone[2])+zone[3]/2*np.cos(zone[2])],
                            [zone[0]-zone[4]/2*np.cos(zone[2])+zone[3]/2*np.sin(zone[2]),zone[1]-zone[4]/2*np.sin(zone[2])-zone[3]/2*np.cos(zone[2])],
                            [zone[0]+zone[4]/2*np.cos(zone[2])+zone[3]/2*np.sin(zone[2]),zone[1]+zone[4]/2*np.sin(zone[2])-zone[3]/2*np.cos(zone[2])],
                            [zone[0]+zone[4]/2*np.cos(zone[2])-zone[3]/2*np.sin(zone[2]),zone[1]+zone[4]/2*np.sin(zone[2])+zone[3]/2*np.cos(zone[2])]])
        if state.within(corners):
            is_near = True
    return is_near

def plotDistribution(scores,env):
	fig, ax = plt.subplots()
	c = ["navy", [69/float(255), 117/float(255), 180/float(255)], [116/float(255),173/float(255),209/float(255)], [171/float(255),217/float(255),233/float(255)], [224/float(255),243/float(255),248/float(255)], [255/float(255),255/float(255),191/float(255)], [254/float(255),224/float(255),144/float(255)], [253/float(255),174/float(255),97/float(255)], [244/float(255),109/float(255),67/float(255)], [215/float(255),48/float(255),39/float(255)], "firebrick"]
	v = [0, 0.1, 0.17, 0.25, .32, .42, 0.52, 0.62, 0.72, 0.85, 1.]
	l = list(zip(v,c))
	palette=LinearSegmentedColormap.from_list('rg',l, N=256)
	palette.set_under([0.5, 0.5, 0.5], 0.3)

	data = plt.imshow(scores, cmap=palette, interpolation='hamming', vmin=0.4, vmax=1.5)
    # data = plt.imshow(scores, cmap='jet', vmin=0, vmax=1.5)
    # data = plt.imshow(scores, cmap='jet',interpolation='gaussian', vmin=0, vmax=1.5)
	plt.xlim((0-0.5, env.numOfCols-0.5))
	plt.ylim((0-0.5, env.numOfRows-0.5))
	fig.colorbar(data, ax=ax)
	# Major ticks every 20, minor ticks every 5
	major_ticks_x = np.arange(0, env.numOfCols, 5)
	minor_ticks_x = np.arange(0, env.numOfCols, 1)
	major_ticks_y = np.arange(0, env.numOfRows, 5)
	minor_ticks_y = np.arange(0, env.numOfRows, 1)
	print("plotting...")
	ax.set_xticks(major_ticks_x)
	ax.set_xticks(minor_ticks_x, minor=True)
	ax.set_yticks(major_ticks_y)
	ax.set_yticks(minor_ticks_y, minor=True)
	ax.grid(which='minor', alpha=0.4)
	# ax.grid(which='major', alpha=0.7)
	plt.savefig("/home/roya/Fall Risk Evaluation Results/HFES-Rooms/Room-3-Inboard-Headwall/Day/Room-3-Inboard-Headwall_day_final.pdf", dpi =300)
	plt.savefig("/home/roya/Fall Risk Evaluation Results/HFES-Rooms/Room-3-Inboard-Headwall/Day/Room-3-Inboard-Headwall_day_final.png", dpi =300)

	plt.show()

print("Environment Setup...")
env = Environment_Image()

if __name__ == '__main__':
    weights = [0.7, 0.3]
    v_mean = 1.2
    v_sigma = 0.3
    w_mean = 1
    w_sigma = 0.3
    num_trials = 1
    fallRisk = FallRiskAssesment(env)
    print("Fall Risk Baseline Calculation...")
    fallRisk.update(assistive_device = False)
    fallRisk.plotDistribution(fallRisk.scores_light, 'light', 'nearest')
    fallRisk.plotDistribution(fallRisk.scores_floor, 'floor', 'nearest')
    fallRisk.plotDistribution(fallRisk.scores_door, 'door', 'nearest')
    fallRisk.plotDistribution(fallRisk.scores_support, 'support', 'nearest')
    fallRisk.plotDistribution(fallRisk.scores, 'baseline', 'hamming')

    obss = []
    for obj in env.objects:
        obss.append([obj.conf.x, obj.conf.y, obj.conf.z, obj.length, obj.width, obj.name])
    for wall in env.walls:
        wall_c = [(wall[0][0]+wall[0][2])/2, (wall[0][1]+wall[0][3])/2]
        wall_d = [np.sqrt((wall[0][0]-wall[0][2])**2+(wall[0][1]-wall[0][3])**2),0.6]
        wall_angle = np.arctan2((wall[0][1]-wall[0][3]), (wall[0][0]-wall[0][2]))
        obss.append([wall_c[0], wall_c[1], wall_angle+0.001, wall_d[0], wall_d[1], "wall"])
    obstacles = define_obs(obss)


    n = 50
    # intention_set = {'start': ['Bed', 'Main Door', 'Bed', 'Toilet', 'Sink-Bath', 'Chair-Patient', 'Toilet', 'Bed', 'Chair-Patient', 'Bed', 'Chair-Visitor'], 'end': ['Main Door', 'Bed', 'Toilet', 'Sink-Bath', 'Bed', 'Toilet', 'Chair-Patient', 'Chair-Patient', 'Bed', 'Chair-Visitor', 'Bed'], 'frequency': [3, 3, 6, 6, 6, 3, 3, 2, 2, 1, 1]} # for P22 & Room-4 & Room-2 designs
    intention_set = {'start': ['Bed', 'Main Door', 'Bed', 'Toilet', 'Sink-Bath', 'Chair-Patient', 'Toilet', 'Bed', 'Chair-Patient', 'Bed', 'Sofa'], 'end': ['Main Door', 'Bed', 'Toilet', 'Sink-Bath', 'Bed', 'Toilet', 'Chair-Patient', 'Chair-Patient', 'Bed', 'Sofa', 'Bed'], 'frequency': [3, 3, 6, 6, 6, 3, 3, 2, 2, 1, 1]} # for A-K & S-B & J-M & J-C & J-G & B-L & B-JH & Room-1 & Room-3 designs
    # intention_set = {'start': ['Bed', 'Sofa'], 'end': ['Sofa', 'Bed'], 'frequency': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]} # for test

    scenarios = []
    TrajectoryPoints = []
    counter = 0
    for intention in range(len(intention_set['start'])):
        for trial in range(intention_set['frequency'][intention]):
            found = 0
            while found == 0:
                patient_s = sample_point(intention_set['start'][intention], env.sitting_zones, env.objects, obstacles)
                print("start")
                print(patient_s)
                patient_g = sample_point(intention_set['end'][intention], env.sitting_zones, env.objects, obstacles)
                print("end")
                print(patient_g)
                scenario = {'start': patient_s, 'end': patient_g, 'v_max': random.gauss(v_mean, v_sigma), 'w_max': random.gauss(w_mean, w_sigma)}
                scenarios.append(scenario)
                print("scenario")
                print(scenario)
                print("Trajectory prediction for scenario {0}, trial {1}".format(intention+1, trial+1))
                cost, predicted_patient_traj, status = OptPath_patient2(scenario['start'], scenario['end'],  [scenario['v_max'], scenario['w_max']] , obstacles, n, assistive_device=False)
                if status == 2 :
                    found = 1
                    traj = []
                    print("Finding activity types...")
                    for i in range(len(predicted_patient_traj)):
                        if is_near_sitting_object(Point(predicted_patient_traj[i]), env.sitting_zones, intention_set['start'][intention]) :
                            traj.append([predicted_patient_traj[i], 'sit_to_stand'])
                        elif is_near_sitting_object(Point(predicted_patient_traj[i]), env.sitting_zones, intention_set['end'][intention]):
                            traj.append([predicted_patient_traj[i], 'stand_to_sit'])
                        else:
                            traj.append([predicted_patient_traj[i], 'walking'])
            # print(traj)
            counter += 1
            TrajectoryPoints.append(fallRisk.getDistibutionForTrajectory(traj, counter, plot=True, assistive_device = False))

            print("***********************************************")

    print("Final Scores Calculation...")
    num = np.ones([env.numOfCols,env.numOfRows]) # Initializing scores as a zero matrix with size of number_of_rows * number_of_columns.
    for traj in TrajectoryPoints:
        for point in traj:
            [i,j] = fallRisk.meter2grid(point[0][0])
            fallRisk.scores[i,j] += point[1]
            num[i,j] += 1
    for i in range(env.numOfCols):
        for j in range(env.numOfRows):
            fallRisk.scores[i,j] = fallRisk.scores[i,j]/num[i,j]
    print("Plot...")
    plotDistribution(fallRisk.scores, env)


    # print(trajFallRisk)
	#fall_risk = []
	#for i in range(len(trajFallRisk)):
	#	risk = 0
	#	for j in range(i):
	#		risk += trajFallRisk[j]
	#	m = n - i
        #	cost, new_predicted_patient_traj = OptPath_patient(predicted_patient_traj[i], patient_g, v_max, obstacles, m, assistive_device=True)
	#	trajFallRisk_new = fallRisk.getDistibutionForTrajectory(new_predicted_patient_traj, plot=False, assistive_device = True)
	#	for j in range(len(trajFallRisk_new)):
	#		risk += trajFallRisk_new[j]
	#	fall_risk.append(risk)
