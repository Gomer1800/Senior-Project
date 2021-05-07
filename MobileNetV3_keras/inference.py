from keras.models import load_model

from keras import backend as K

from src.MobileNet_V3 import Hswish
import numpy as np

import sys
import os
import cv2
import configparser
from math import sqrt, atan, sin, cos, pi
from random import randint

from trajectory import *

import matplotlib.pyplot as plt

'''Parse the command line to find what flags were given'''


def parseCommandLine(numArgs, args):
    flags = []

    for i in range(numArgs):
        if args[i] == '-c':
            if '-c' in flags:
                print("Usage: python3 inference.py -c <config_file> [-n <int>] [-w <weights>] | [-r]")
                return []
            flags.append(args[i])

        if args[i] == '-w':
            if '-w' in flags:
                print("Usage: python3 inference.py -c <config_file> [-n <int>] [-w <weights>] | [-r]")
                return []
            flags.append(args[i])

        if args[i] == '-r':
            if '-r' in flags:
                print("Usage: python3 inference.py -c <config_file> [-n <int>] [-w <weights>] | [-r]")
                return []
            flags.append(args[i])

        if args[i] == '-n':
            if '-n' in flags:
                print("Usage: python3 inference.py -c <config_file> [-n <int>] [-w <weights>] | [-r]")
                return []
            flags.append(args[i])

    return flags


'''Run through the images in the given directory in the config file and determine the
   predictions of the model, save the original image with the prediction overlayed'''


def main():
    config_file = None
    weight_path = None
    numArgs = len(sys.argv)
    args = sys.argv
    rank = False
    numInfer = None

    flags = parseCommandLine(numArgs, args)

    for f in flags:
        index = args.index(f)

        # Save the configuration file
        if f == '-c':
            config_file = args[index + 1]

        # USe the given weights file instead of the one in the config file
        elif f == '-w':
            weight_path = args[index + 1]

        # Rank the images
        elif f == '-r':
            rank = True

        # Predict only the given portion of the images
        elif f == '-n':
            try:
                numInfer = int(args[index + 1])
            except:
                print("Usage: python3 inference.py -c <config_file> [-n <int>] [-w <weights>] | [-r]")
                return

    # Config file flag was not specified
    if config_file is None:
        print("Usage: python3 inference.py -c <config_file> [-n <int>] [-w <weights>] | [-r]")
        return

    # Rank the unlabeled images
    if rank:
        print("Rank Images")
        rankImages(config_file)

    # Predict the free space in the images
    else:
        print("Predict Images")
        predictImages(config_file, weight_path, numInfer)

    return


'''Make predictions about the free space for the given images using the model in the 
   config file or given weight_path'''


def predictImages(config_file, weight_path, numInfer):
    # Make inferences on the images in the image_dir
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** get configuration
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    # ** Weights
    if weight_path is None:
        weight_path = config_client.get('inference', 'weight_path')

    # ** Images
    image_dir = config_client.get('inference', 'image_path')
    if not os.path.exists(image_dir):
        raise Exception('input dir does not exist')

    # ** Where to store the predictions
    inference_dir = config_client.get('inference', 'inference_dir')
    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)

    # ** Input Sizes
    input_width = config_client.getint('model', 'input_width')
    input_height = config_client.getint('model', 'input_height')

    # ** Number of outputs
    num_outputs = config_client.getint('model', 'num_outputs')

    print("Loading Model")
    # Load the old model
    '''model = model = load_model(os.path.join(ROOT_DIR, weight_path),
                            custom_objects={'Hswish':Hswish})'''

    # Load the the new model
    model = load_model(os.path.join(ROOT_DIR, weight_path),
                       custom_objects={'_hard_swish': _hard_swish,
                                       '_relu6': _relu6})

    trajectory = Trajectory(input_width, input_height, num_outputs)
    robotCenter = trajectory.robotCenter
    robotWidth = trajectory.robotWidth
    robotCenter = trajectory.robotCenter
    robotLeft = trajectory.robotLeft
    robotRight = trajectory.robotRight
    robotFront = trajectory.robotFront  # Front of the robot
    maxTranslation = trajectory.maxTranslation

    # Run through the images and predict the free space
    n = 1
    stepSize = input_width // num_outputs
    for _id in os.listdir(image_dir):
        # Get the image
        _id_path = os.path.join(image_dir, _id)

        # Normalize the input image
        X = np.empty((1, input_height, input_width, 3), dtype='float32')
        X[0, :, :, :] = image_augmentation(cv2.imread(_id_path))

        # Predict the free space
        print("Predicting Image: " + str(n))
        prediction = model.predict(X)[0]

        # Load the image to draw the extracted mask data on for validation
        validationMaskImage = cv2.imread(_id_path)

        highestPoint = robotCenter
        x = 0
        for i in range(len(prediction)):
            y = int(round(prediction[i] * input_height))

            if y < highestPoint[1]:
                highestPoint = (x, y)

            # Draw circles on the original image to show where the predicted free space occurs
            validationMaskImage = cv2.circle(validationMaskImage, (x, y), 1, (0, 255, 0), -1)
            x += stepSize

        # Draw a line across the image where the furthest point from the center is
        cv2.line(validationMaskImage, (0, highestPoint[1]), (input_width - 1, highestPoint[1]), (0, 0, 255), 2)

        # Draw lines representing the sides of the robot
        cv2.line(validationMaskImage, (robotCenter[0] - robotWidth, robotCenter[1]),
                 (robotCenter[0] - robotWidth, robotFront), (0, 0, 255), 2)
        cv2.line(validationMaskImage, (robotCenter[0] + robotWidth, robotCenter[1]),
                 (robotCenter[0] + robotWidth, robotFront), (0, 0, 255), 2)

        # Draw a line representing the boundary of the front of the robot
        cv2.line(validationMaskImage, (robotLeft, robotFront), (robotRight, robotFront), (0, 0, 255), 2)

        # Calculate the trajectory of the robot
        (translation, rotation) = trajectory.calculateTrajectory(prediction)
        # Convert the trajectory percentages to the target point
        (translation_x, translation_y) = trajectory.trajectoryToPoint(translation, rotation)

        # Display the magnitude and direction of the vector the robot should drive along
        cv2.putText(validationMaskImage, "Rotation: " + str(rotation), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(validationMaskImage, "Translation: " + str(translation), (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw an arrowed line indicating the predicted trajectory of the robot
        if rotation == -1 and translation == 0:
            cv2.arrowedLine(validationMaskImage, (robotCenter[0], robotFront),
                            (input_width - 1, robotFront), (0, 0, 255), 2)
        elif rotation == 1 and translation == 0:
            cv2.arrowedLine(validationMaskImage, (robotCenter[0], robotFront),
                            (0, robotFront), (0, 0, 255), 2)
        else:
            cv2.arrowedLine(validationMaskImage, robotCenter, (translation_x, translation_y), (0, 0, 255), 2)

        # Save the overlayed image
        cv2.imwrite(inference_dir + "/" + _id.replace(".jpg", "") + "_inference.jpg", validationMaskImage)

        n += 1
        if numInfer is not None and n - 1 == numInfer:
            break

    return


'''Compute the Euclidean distance between 2 points'''


def distance(p1, p2):
    return sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


'''Rank the images in based on performace in 4 trained models. Save the worst 20% by default
   and return the average error of the worst 10%'''


def rankImages(config_file):
    # Make inferences on the images in the image_dir
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Open the file to store the names of the images with the largest error
    try:
        worst_file = open(os.path.join(ROOT_DIR, 'Rankings'), 'w')
    except OSError as err:
        print("Error: {0}".format(err))
        return

    # ** get configuration
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    # ** Images
    image_dir = config_client.get('rank', 'image_path')
    if not os.path.exists(image_dir):
        raise Exception('input dir does not exist')

    # ** Input Sizes
    input_width = config_client.getint('model', 'input_width')
    input_height = config_client.getint('model', 'input_height')

    # ** Number of outputs
    num_outputs = config_client.getint('model', 'num_outputs')

    # ** Prefix for weight files
    weight_path_prefix = config_client.get('rank', 'weight_path_prefix')

    # ** Ranking percentage
    rank_percent = config_client.getfloat('rank', 'rank_percent')

    # Load the 4 models
    print("Loading model 1")
    model_1 = load_model(os.path.join(ROOT_DIR, weight_path_prefix + '_h1'),
                         custom_objects={'_hard_swish': _hard_swish,
                                         '_relu6': _relu6})
    print("Loading model 2")
    model_2 = load_model(os.path.join(ROOT_DIR, weight_path_prefix + '_h2'),
                         custom_objects={'_hard_swish': _hard_swish,
                                         '_relu6': _relu6})
    print("Loading model 3")
    model_3 = load_model(os.path.join(ROOT_DIR, weight_path_prefix + '_h3'),
                         custom_objects={'_hard_swish': _hard_swish,
                                         '_relu6': _relu6})
    print("Loading model 4")
    model_4 = load_model(os.path.join(ROOT_DIR, weight_path_prefix + '_h4'),
                         custom_objects={'_hard_swish': _hard_swish,
                                         '_relu6': _relu6})

    # Run through the images and rank them
    n = 0
    errors = []
    # Determine the sum of squared differences for each image
    for _id in os.listdir(image_dir):
        # Get the image
        _id_path = os.path.join(image_dir, _id)

        # Normalize the input image
        X = np.empty((1, input_height, input_width, 3), dtype='float32')
        X[0, :, :, :] = image_augmentation(cv2.imread(_id_path))

        # Predict the free space for the 4 models
        print("Ranking Image: " + str(n))
        prediction_1 = model_1.predict(X)[0]
        prediction_2 = model_2.predict(X)[0]
        prediction_3 = model_3.predict(X)[0]
        prediction_4 = model_4.predict(X)[0]

        # Get the average prediction of the 4 models
        avg_prediction = []
        for i in range(num_outputs):
            avg = prediction_1[i] + prediction_2[i] + prediction_3[i] + prediction_4[i]
            avg = avg / 4
            avg_prediction.append(avg)

        # Get the squared error for each model
        sqrd_error_1 = []
        sqrd_error_2 = []
        sqrd_error_3 = []
        sqrd_error_4 = []
        for i in range(num_outputs):
            sqrd_error_1.append((prediction_1[i] - avg_prediction[i]) ** 2)
            sqrd_error_2.append((prediction_2[i] - avg_prediction[i]) ** 2)
            sqrd_error_3.append((prediction_3[i] - avg_prediction[i]) ** 2)
            sqrd_error_4.append((prediction_4[i] - avg_prediction[i]) ** 2)

        # Sum the squared errors for each model
        error_sum_1 = sum(sqrd_error_1)
        error_sum_2 = sum(sqrd_error_2)
        error_sum_3 = sum(sqrd_error_3)
        error_sum_4 = sum(sqrd_error_4)

        # Sum all the errors
        error_total = error_sum_1 + error_sum_2 + error_sum_3 + error_sum_4

        errors.append((_id, error_total))
        n += 1

    # Rank the errors in descending order
    rankings = sorted(errors, key=lambda x: x[1], reverse=True)

    '''ranks = []
             for i in range(len(rankings)-1, -1, -1):
                ranks.append(rankings[i][1])
          
             #Plot the errors
             plt.plot(ranks)
             plt.ylabel("Errors")
             plt.xlabel("Image")
             plt.show()'''

    # Find the average error of the worst 10%
    num_worst = int(n * 0.1)
    avg_error = 0
    for i in range(num_worst):
        avg_error += rankings[i][1]
    avg_error = avg_error / num_worst
    print(avg_error)

    # Save the worst rank_percent of the images
    num_worst = int(n * rank_percent)
    for i in range(num_worst):
        worst_file.write(rankings[i][0] + '\n')
    return


# Turn the image to RGB and normalize it
def image_augmentation(img):
    if img is None:
        raise Exception('** Failed to read image.')
    # to rgb
    img_copy = img.copy()
    img_copy = img_copy[:, :, ::-1]

    img_norm = normalize(img_copy)

    return img_norm


# Normalize the image
def normalize(img):
    return img / 255.0


def _relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _hard_swish(x):
    """Hard swish
    """
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0


if __name__ == '__main__':
    main()
