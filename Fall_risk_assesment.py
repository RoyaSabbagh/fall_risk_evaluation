

#!/usr/bin/env python

import copy
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
import matplotlib.cbook as cbook
import matplotlib.image as image
from matplotlib._png import read_png
import cv2
from read_blueprint import read_blueprint
from shapely.geometry import Point, Polygon

class Environment_Image():

	""" This a class to define the environment in a room to be used in the FallRiskAssesment class."""
	def __init__(self):

		self.unit_size_m = 0.10 # Grid size in cm (X,Y).
		self.numOfRows = 100 # Number of grids in a Y direction.
		self.numOfCols = 100 # Number of grids in a X direction.
		self.occupied = np.zeros([self.numOfCols, self.numOfRows]) # Initializing occupied as a zero matrix with size of number_of_rows * number_of_columns. When a grid is occupied, its corresponding value in this matrix is one, otherwise it is zero.

		library_file_name = "/home/roya/catkin_ws/src/Risk_Aware_Planning/Object_Library.csv" # Put the object library file address here.
		image_file_name = "/home/roya/catkin_ws/src/Risk_Aware_Planning/Room_Designs/Room-3-Inboard-Headwall_day.png" # Put the image file address here.
		self.rooms, self.objects, self.sitting_zones, self.walls, self.doors, self.lights= read_blueprint(image_file_name, library_file_name) # Reading floor plan and finding room sections, objects, walls, doors and lights.

		self.floor = np.zeros([self.numOfRows, self.numOfCols]) # Initializing floor_type as a zero matrix with size of number_of_rows * number_of_columns.
		for room in self.rooms:
			for row in range(self.numOfRows):
				for col in range(self.numOfCols):
					gridCoordinate = self.grid2meter(col,row)
					gridPoint = Point(gridCoordinate)
					if gridPoint.within(room.polygon): # Assigning the floor type based on the room section to each grid in the space
						self.floor[col, row] = room.surfaceRisk

	def grid2meter(self, col, row):
		x = col*self.unit_size_m
		y = row*self.unit_size_m
		return (x,y)

# class Environment_Generated():
#
#

class FallRiskAssesment():

	def __init__(self, env):
		self.env = env # env is an Environment class.
		self.scores = np.zeros([self.env.numOfCols,self.env.numOfRows]) # Initializing scores as a zero matrix with size of number_of_rows * number_of_columns.
		self.scores_light = np.zeros([self.env.numOfCols,self.env.numOfRows])
		self.scores_door = np.zeros([self.env.numOfCols,self.env.numOfRows])
		self.scores_support = np.zeros([self.env.numOfCols,self.env.numOfRows])
		self.scores_floor = np.zeros([self.env.numOfCols,self.env.numOfRows])
		self.theta_d = [0.8,0.8,1.5] # [f_min, d_min, d_max] used in the supporting object factor
		self.theta_l = [100,500,1000] # [night_min, night_max/day_min, day_max] used in the light factor
		self.theta_r = 1 #
		self.theta_v = 1 #
		self.ESP = [] # Initializing External Supporting Point list as an empty array.
		self.occupied = np.zeros([self.env.numOfRows, self.env.numOfCols])
		for obj in self.env.objects:
			# print(obj.name)
			# print(obj.conf)
			self.update_ESP(obj) # For each object in list of objects, the surrounding grids are updated basd on support level of the object.

	def update_ESP(self, obj):
		"""In this function, we find the effect of the input object on all the grids in the environment. When a grid is occupied we find from which direction it is free and add that as a External Supporting Point, if it not free from any direction, it is not added as a ESP and only it is marked as an occupied cell."""
		for row in range(self.env.numOfRows):
			for col in range(self.env.numOfCols):
				coordinate = self.grid2meter([row, col]) # Changing grid to meter coordinate system
				gridPoint = Point(coordinate)
				if gridPoint.within(obj.polygon):
					coordinate = self.grid2meter([row+1, col]) # Changing grid to meter coordinate system
					rightGridPoint = Point(coordinate)
					if not rightGridPoint.within(obj.polygon):
						self.ESP.append([rightGridPoint, 3, obj.support])
					coordinate = self.grid2meter([row, col+1]) # Changing grid to meter coordinate system
					upGridPoint = Point(coordinate)
					if  not upGridPoint.within(obj.polygon):
						self.ESP.append([upGridPoint, 2, obj.support])
					coordinate = self.grid2meter([row-1, col]) # Changing grid to meter coordinate system
					leftGridPoint = Point(coordinate)
					if not leftGridPoint.within(obj.polygon):
						self.ESP.append([leftGridPoint, 1, obj.support])
					coordinate = self.grid2meter([row, col-1]) # Changing grid to meter coordinate system
					downGridPoint = Point(coordinate)
					if not downGridPoint.within(obj.polygon):
						self.ESP.append([downGridPoint, 0, obj.support])
					self.occupied[row, col] = 1

	def findFactors(self, grid):
		# print(grid)
		gridCoordinate = self.grid2meter(grid)
		gridPoint = Point(gridCoordinate)

		self.scores_support[grid[0],grid[1]] = copy.deepcopy(self.closestSupportDistance_effect(gridPoint)) # Adding the effect of ESP on this grid.
		self.scores_floor[grid[0],grid[1]] = copy.deepcopy(self.floor_effect(grid)) # Adding the effect of ESP on this grid.
		self.scores_light[grid[0],grid[1]] = copy.deepcopy(self.lighting_effect(gridPoint)) # Adding the effect of lighting on this grid.
		self.scores_door[grid[0],grid[1]] = copy.deepcopy(self.door_passing_effect(gridPoint)) # Adding the effect of door passing on this grid.
		# print(self.scores_light[grid[0],grid[1]])

	def floor_effect(self,grid):
		floorTypeRisk = self.env.floor[grid[0], grid[1]] # Initializing the effect of floor type on this grid base on its own floor type.
		for i in [-1,1]: # Cheking if there is any transition to another type of floor or not in the surrounding grids and modifying the effect of floor type.
			if grid[0]+i < self.env.numOfCols and grid[0]+i > 0:
				if self.env.floor[grid[0]+i, grid[1]]!= self.env.floor[grid[0], grid[1]] and self.env.floor[grid[0]+i, grid[1]] != 0:
					floorTypeRisk = floorTypeRisk * 1.05
			if grid[1] + i < self.env.numOfRows and grid[1] + i > 0:
				if self.env.floor[grid[0], grid[1]+i]!= self.env.floor[grid[0], grid[1]] and self.env.floor[grid[0], grid[1]+i] != 0:
					floorTypeRisk = floorTypeRisk * 1.05
		return floorTypeRisk

	def door_passing_effect(self, gridPoint):
		""" This function calculates the effect of door passing for a given grid. It needs to be updated. """
		risk = 0
		for room in self.env.rooms:
			if gridPoint.within(room.polygon) == 1:
				risk = 1
		for door in self.env.doors:
			if gridPoint.within(door.polygon) == True:
				risk = float(1)/door.support
		return risk

	def lighting_effect(self, gridPoint):
		""" This function calculates the effect of lighting for a given grid. It needs to be updated. """
		risk = 1
		light_intensity = 0
		for room in self.env.rooms:
			if gridPoint.within(room.polygon) == 1:
				light_intensity = 1
		for light in self.env.lights:
			if self.in_the_same_room(light.point, gridPoint):
				dist = self.distance(light.point, gridPoint)
				if dist == 0:
					light_intensity += self.theta_l[2]
				else:
					light_intensity += float(1)/((dist)**2)*light.intensity
		if light_intensity == 0:
			risk = 0
		elif light_intensity <= self.theta_l[0]:
			risk = risk * 1.07
		elif light_intensity > self.theta_l[0] and light_intensity < self.theta_l[1]:
			risk = risk * 1.03
		return risk

	def closestSupportDistance_effect(self, gridPoint):
		""" This function calculates the effect of ESPs for a given grid. It needs to be updated. """
		risk = 0
		for room in self.env.rooms:
			if gridPoint.within(room.polygon) == 1:
				risk = 1
		min_dist = self.theta_d[2]
		support_type = 1
		for support in self.ESP:
			if self.in_the_same_room(support[0], gridPoint):
				dist = self.distance(support[0], gridPoint)
				if dist < min_dist:
					min_dist = dist
					support_type = support[2]
		for wall in self.env.walls:
			wall_dist = self.distance_wall(gridPoint, wall)
			if wall_dist < min_dist:
				min_dist = wall_dist
				support_type = 1.1
		if min_dist <= self.theta_d[1]:
			risk *= self.theta_d[0]
		elif min_dist > self.theta_d[1] and min_dist <= self.theta_d[2]:
			risk *= self.theta_d[0] + (min_dist - self.theta_d[1])*(1-self.theta_d[0])/(self.theta_d[2]-self.theta_d[1])
		risk = float(risk) / support_type
		return risk

	def in_the_same_room(self, grid1, grid2):
		""" This function determines whether two grids are in the same room or not. It is mostly used for lights to make sure it doesn't pass through walls. """
		room_grid_1 = "out"
		room_grid_2 = "out"
		for room in self.env.rooms:
			if grid1.within(room.polygon) == 1:
				room_grid_1 = room.name
			if grid2.within(room.polygon) == 1:
				room_grid_2 = room.name
		if room_grid_1 == room_grid_2:
			return True
		else:
			return False

	def distance(self, obj, gridPoint):
		""" This function finds the Euclidean distance between two grids."""
		dist_x =  obj.x - gridPoint.x
		dist_y = obj.y - gridPoint.y
		dist = np.sqrt(dist_x**2+dist_y**2)
		return dist

	def distance_wall(self, point, wall):
		""" This function calculates the minimum distance from a grid and a wall segment"""
		LineMag = math.sqrt(math.pow((wall[0][0] - wall[0][2]), 2)+ math.pow((wall[0][1] - wall[0][3]), 2))

		if LineMag < 0.00000001:
			DistancePointLine = float("inf")
			return DistancePointLine

		u1 = (((point.x - wall[0][0]) * (wall[0][2] - wall[0][0])) + ((point.y - wall[0][1]) * (wall[0][3] - wall[0][1])))
		u = u1 / (LineMag * LineMag)

		if (u < 0.00001) or (u > 1):
			#// closest point does not fall within the line segment
			DistancePointLine = float("inf")
			return DistancePointLine
		else:
			# Intersecting point is on the line, use the formula
			ix = wall[0][0] + u * (wall[0][2] - wall[0][0])
			iy = wall[0][1] + u * (wall[0][3] - wall[0][1])
			DistancePointLine = math.sqrt(math.pow((point.x - ix), 2)+ math.pow((point.y - iy), 2))
		return DistancePointLine

	def grid2meter(self, grid):
	    ''' grid to meter'''
	    x = grid[0] * self.env.unit_size_m
	    y = grid[1] * self.env.unit_size_m
	    return (x,y)

	def meter2grid(self, point):
	    ''' meter to grid'''
	    x = int(point[0] / self.env.unit_size_m)
	    y = int(point[1] / self.env.unit_size_m)
	    return [x,y]

	def update(self, assistive_device):
		""" This function updates the fall risk score based on all the factors except the trajectory. It needs to be updated. """
		for i in range(self.env.numOfCols):
			for j in range(self.env.numOfRows):
				if self.occupied[i,j] == 1:
					self.scores[i, j] = 0.1
				else:
					if assistive_device == False:
						assistive_device_risk = 1
					else:
						assistive_device_risk = 1.05
					self.findFactors([i,j])
					self.scores[i, j] = assistive_device_risk * self.scores_light[i, j] * self.scores_floor[i, j] * self.scores_support[i, j] * self.scores_door[i, j]
        # print self.scores

	def trajectoryRiskEstimate(self, point):
		""" This function calculates the effect of trajectory. It needs to be updated. """
		risk = 1
		if point[1] == 'walking':
			risk *= 1.2
		elif point[1] == 'sit-to-stand':
			risk *= 1.05
		elif point[1] == 'stand-to-sit':
			risk *= 1.1
		if point[0][4]**2 > 0.04 and point[0][4]**2 < 0.16:
			risk *= 1.2
		if point[0][4]**2 > 0.16:
			risk *= 1.4
		return risk

	def getDistibutionForTrajectory(self, trajectory, counter, plot, assistive_device):
		""" Finding the risk distribution over a given trajectory defined by waypoints. """
		# self.update(assistive_device)
		TrajectoryScores = []
		TrajectoryPoints = []
		for point in trajectory:
			# print point
			# point: [[x, y, phi, v, w],"activity"]
			grid = self.meter2grid(point[0])
			PointScore = self.scores[grid[0],grid[1]]
			PointScore *= self.trajectoryRiskEstimate(point)
			TrajectoryScores.append(PointScore)
			TrajectoryPoints.append([point,PointScore])
		if plot:
			self.plotTrajDist(TrajectoryScores, trajectory, counter)

		return TrajectoryPoints

	def plotDistribution(self, distribution, factor, interpolation):
		fig, ax =plt.subplots()

		c = ["navy", [69/float(255), 117/float(255), 180/float(255)], [116/float(255),173/float(255),209/float(255)], [171/float(255),217/float(255),233/float(255)], [224/float(255),243/float(255),248/float(255)], [255/float(255),255/float(255),191/float(255)], [254/float(255),224/float(255),144/float(255)], [253/float(255),174/float(255),97/float(255)], [244/float(255),109/float(255),67/float(255)], [215/float(255),48/float(255),39/float(255)], "firebrick"]
		v = [0, 0.1, 0.17, 0.25, .32, .42, 0.52, 0.62, 0.72, 0.85, 1.]
		l = list(zip(v,c))
		palette=LinearSegmentedColormap.from_list('rg',l, N=256)
		palette.set_under([0.5, 0.5, 0.5], 0.3)

		data = plt.imshow(distribution, cmap=palette, interpolation=interpolation, vmin=0.4, vmax=1.5)
		# data = plt.imshow(distribution, cmap='jet', interpolation=interpolation, vmin=0, vmax=1.5)
		plt.xlim((0-0.5, self.env.numOfCols-0.5))
		plt.ylim((0-0.5, self.env.numOfRows-0.5))
		fig.colorbar(data, ax=ax)
		# Major ticks every 20, minor ticks every 5
		major_ticks_x = np.arange(0, self.env.numOfCols, 5)
		minor_ticks_x = np.arange(0, self.env.numOfCols, 1)
		major_ticks_y = np.arange(0, self.env.numOfRows, 5)
		minor_ticks_y = np.arange(0, self.env.numOfRows, 1)

		ax.set_xticks(major_ticks_x)
		ax.set_xticks(minor_ticks_x, minor=True)
		ax.set_yticks(major_ticks_y)
		ax.set_yticks(minor_ticks_y, minor=True)
		ax.grid(which='minor', alpha=0.4)
		# ax.grid(which='major', alpha=0.7)
		plt.savefig("/home/roya/Fall Risk Evaluation Results/HFES-Rooms/Room-3-Inboard-Headwall/Day/Room-3-Inboard-Headwall_day_{}.pdf".format(factor), dpi =300)
		plt.savefig("/home/roya/Fall Risk Evaluation Results/HFES-Rooms/Room-3-Inboard-Headwall/Day/Room-3-Inboard-Headwall_day_{}.png".format(factor), dpi =300)
		plt.show()

	def plotTrajDist(self, trajFallRisk, trajectory, counter):
		x = []
		y = []
		dydx = []
		for i in range(len(trajectory)):
			x.append(trajectory[i][0][1])
			y.append(trajectory[i][0][0])
			dydx.append(trajFallRisk[i])

		# Create a set of line segments so that we can color them individually
		# This creates the points as a N x 1 x 2 array so that we can stack points
		# together easily to get the segments. The segments array for line collection
		# needs to be (numlines) x (points per line) x 2 (for x and y)
		points = np.array([x, y]).T.reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)

		fig, ax = plt.subplots()
		# Create a continuous norm to map from data points to colors
		datafile = cbook.get_sample_data("/home/roya/catkin_ws/src/Risk_Aware_Planning/Room_Designs/Room-3-Inboard-Headwall_objects_rotated.png", asfileobj=False)
		im = image.imread(datafile)
		ax.imshow(im, aspect='auto', extent=(0, 10, 0, 10), alpha=0.5, zorder=-1)

		c = ["navy", [69/float(255), 117/float(255), 180/float(255)], [116/float(255),173/float(255),209/float(255)], [171/float(255),217/float(255),233/float(255)], [253/float(255),174/float(255),97/float(255)], [244/float(255),109/float(255),67/float(255)], [215/float(255),48/float(255),39/float(255)], "firebrick"]
		v = [0, 0.15, 0.3, 0.45, 0.6, 0.72, 0.85, 1.]
		l = list(zip(v,c))
		palette=LinearSegmentedColormap.from_list('rg',l, N=256)
		lc = LineCollection(segments, cmap=palette, norm=plt.Normalize(0, 1.5))
		# Set the values used for colormapping
		lc.set_array(np.array(dydx))
		lc.set_linewidth(4)
		line = ax.add_collection(lc)
		fig.colorbar(line, ax=ax)
		plt.xlim(0, 10)
		plt.ylim(0, 10)
		plt.savefig("/home/roya/Fall Risk Evaluation Results/HFES-Rooms/Room-3-Inboard-Headwall/Day/Trajectories/Room-3-Inboard-Headwall_day_traj_{}.pdf".format(counter), dpi =300)
		plt.savefig("/home/roya/Fall Risk Evaluation Results/HFES-Rooms/Room-3-Inboard-Headwall/Day/Trajectories/Room-3-Inboard-Headwall_day_traj_{}.png".format(counter), dpi =300)
		# plt.show()

#
# env = Environment_Image()
# fallRisk = FallRiskAssesment(env)
# fallRisk.update(False)
# fallRisk.plotDistribution()
