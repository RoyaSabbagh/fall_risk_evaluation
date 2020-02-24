#!/usr/bin/env python


from Fall_risk_assesment import Environment_Image, FallRiskAssesment
from Trajectory_generation import generagte_trajectory, define_obstacles
import numpy as np
import random


if __name__ == '__main__':

    # ***************************************** Inputs ******************************************************

    # Set file addresses
    input_type = "image"
    path = "/home/roya/catkin_ws/src/fall_risk_evaluation"
    design_name = "Room-3-Inboard-Headwall"
    day_night = "day"
    plots = True


    # Setup parameters for evaluation
    unit_size_m = 0.1
    num_rows = 100
    num_cols = 100

    # Setup parameters used for motion prediction and evaluation
    v = [1.2, 0.3] # maximum linear velocity = [mu, sigma]
    w = [1, 0.3] # maximum angular velocity = [mu, sigma]
    num_points = 10

    # Setup scenarios within the room
    num_trials = 36
    # intention_set = {'start': ['Bed', 'Main Door', 'Bed', 'Toilet', 'Sink-Bath', 'Chair-Patient', 'Toilet', 'Bed', 'Chair-Patient', 'Bed', 'Chair-Visitor'], 'end': ['Main Door', 'Bed', 'Toilet', 'Sink-Bath', 'Bed', 'Toilet', 'Chair-Patient', 'Chair-Patient', 'Bed', 'Chair-Visitor', 'Bed'], 'frequency': [3, 3, 6, 6, 6, 3, 3, 2, 2, 1, 1]} # for P22 & Room-4 & Room-2 designs
    intention_set = {'start': ['Bed', 'Main Door', 'Bed', 'Toilet', 'Sink-Bath', 'Chair-Patient', 'Toilet', 'Bed', 'Chair-Patient', 'Bed', 'Sofa'], 'end': ['Main Door', 'Bed', 'Toilet', 'Sink-Bath', 'Bed', 'Toilet', 'Chair-Patient', 'Chair-Patient', 'Bed', 'Sofa', 'Bed'], 'frequency': [3, 3, 6, 6, 6, 3, 3, 2, 2, 1, 1]} # for A-K & S-B & J-M & J-C & J-G & B-L & B-JH & Room-1 & Room-3 designs
    # intention_set = {'start': ['Bed', 'Sofa'], 'end': ['Sofa', 'Bed'], 'frequency': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]} # for test

    # Setup filenames to save plots
    pdf_filenames = []
    png_filenames = []
    traj_pdf_filenames = []
    traj_png_filenames = []
    library_file = "{0}/Object_Library.csv".format(path) # Put the object library file address here.
    image_file = "{0}/Room_Designs/{1}_{2}.png".format(path, design_name, day_night) # Put the image file address here.
    for factor in ["light", "floor", "door", "support", "baseline"]:
        pdf_filenames.append("{0}/results/{1}/{2}/{1}_{2}_{3}.pdf".format(path, design_name, day_night, factor))
        png_filenames.append("{0}/results/{1}/{2}/{1}_{2}_{3}.png".format(path, design_name, day_night, factor))

    background_filename = "{0}/Room_Designs/{1}_objects_rotated.png".format(path, design_name)
    for counter in range(num_trials):
        traj_pdf_filenames.append("{0}/results/{1}/{2}/Trajectories/{1}_{2}_traj_{3}.pdf".format(path, design_name, day_night, counter+1))
        traj_png_filenames.append("{0}/results/{1}/{2}/Trajectories/{1}_{2}_traj_{3}.png".format(path, design_name, day_night, counter+1))

    # ************************************** Setup environment *************************************************

    print("Environment Setup...")
    if input_type == "image":
        env = Environment_Image(image_file, library_file, unit_size_m, num_rows, num_cols) # Basically, reads an input image and setups the room environment properties for fall risk assessment
    elif input_image == "generated":
        env = Environment_Generated(unit_size_m, num_rows, num_cols)

    obstacles = define_obstacles(env) # Defines obstacles including furniture and walls

    # ************************************ Baseline evaluation **************************************************

    print("Baseline Evaluation...")
    fallRisk = FallRiskAssesment(env) # Initial FallRiskAssesment class
    fallRisk.update(png_filenames, pdf_filenames, assistive_device = False, plot = plots) # Find scores for each baseline factor and baseline evaluation

    # ************************************ Motion evaluation **************************************************

    print("Motion Evaluation...")

    # Initialization
    TrajectoryPoints = []
    counter = 0

    for intention in range(len(intention_set['start'])):
        for trial in range(intention_set['frequency'][intention]):
            # Generating a trajectory for each scenario each trial
            print("Trajectory prediction for scenario {0}, trial {1}: ".format(intention+1, trial+1))
            traj = generagte_trajectory(intention_set['start'][intention], intention_set['end'][intention], env, obstacles, random.gauss(v[0], v[1]), random.gauss(w[0], w[1]), num_points)
            counter += 1
            # Evaluating the generated trajectory
            TrajectoryPoints.append(fallRisk.getDistibutionForTrajectory(traj, counter, background_filename, traj_png_filenames, traj_pdf_filenames, plot=plots, assistive_device = False))

    # ************************************ Overall evaluation **************************************************

    print("Final Scores Calculation...")
    num = np.ones([env.numOfRows,env.numOfCols]) # Initializing number of points in each grid cell as one
    for traj in TrajectoryPoints:
        for point in traj:
            [i,j] = fallRisk.meter2grid(point[0][0]) # Finding the grid cell for each point in trajectories
            fallRisk.scores[i,j] += point[1] # Add the score of that point to the associated grid cell
            num[i,j] += 1 # Add 1 to number of points inside that grid cell
    for i in range(env.numOfRows):
        for j in range(env.numOfCols):
            fallRisk.scores[i,j] = fallRisk.scores[i,j]/num[i,j] # Take the avarage score for each grid cell
    print("Final evaluation plot...")
    fallRisk.plotDistribution(fallRisk.scores, env, 'final', 'hamming')
