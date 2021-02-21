# TODO(LUIS): identify scaleAI logic in dataExtractor
import cv2
import os
import configparser
import numpy as np
import json
import Data_Extractor.generic as generic
from PIL import Image

'''Use python json library to open json file and return list of tasks'''


def parse_json(json_file):
    tasks = None

    try:
        # imageFile = open(json_file, 'r')  # Open the json file
        with open(json_file) as f:  # Open the json file
            try:
                # Tasks are a list of dictionaries
                tasks = json.load(f)
            except:
                print("Error loading json file" + json_file)
    except:
        print("Error opening file: " + json_file)
        return

    # temp for debugging
    for task in tasks:
        pass

    return tasks


'''Download the image and mask data from the .csv file.'''


def downloadImageData(flag, json_file, config_file):
    # UNIQUE
    tasks = parse_json(json_file)

    for task in tasks:
        pass

    # GENERIC
    # ** get configuration
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** Image Sizes
    img_width = config_client.getint('model', 'input_width')
    img_height = config_client.getint('model', 'input_height')

    # ** Number of outputs
    num_outputs = config_client.getint('model', 'num_outputs')

    # GENERIC
    white_list, black_list = None, None
    try:
        if flag == '-a':
            white_list = open("Whitelisted_Images.txt", 'w')
            black_list = open("Blacklisted_Images.txt", 'w')
        elif flag == '-n':
            white_list = open("Whitelisted_Images.txt", 'a')
            black_list = open("Blacklisted_Images.txt", 'a')
        else:
            return
    except OSError as err:
        print("Error: {0}".format(err))
        return

    # GENERIC
    dir_path = os.getcwd()  # Get the current directory path

    # Make the directories to store the image information
    try:
        if not os.path.isdir(dir_path + '/Input_Images'):
            os.mkdir(dir_path + '/Input_Images')
        if not os.path.isdir(dir_path + '/Image_Masks'):
            os.mkdir(dir_path + '/Image_Masks')
        if not os.path.isdir(dir_path + '/Mask_Data'):
            os.mkdir(dir_path + '/Mask_Data')
        if not os.path.isdir(dir_path + '/Mask_Validation'):
            os.mkdir(dir_path + '/Mask_Validation')
        if not os.path.isdir(dir_path + '/Blacklist_Masks'):
            os.mkdir(dir_path + '/Blacklist_Masks')
        if not os.path.isdir(dir_path + '/Whitelist_Masks'):
            os.mkdir(dir_path + '/Whitelist_Masks')
        if not os.path.isdir(dir_path + '/Unlabeled'):
            os.mkdir(dir_path + '/Unlabeled')
    except OSError as err:
        print("Error: {0}".format(err))
        return

    # UNIQUE

    # Download the images and masks from the json file
    img_num = 0
    for task in tasks:
        # UNIQUE
        img_num += 1
        print("Image: " + str(img_num), end='')

        # The name of the original image
        img_name = task['task_id'] + ".jpg"

        # Get the review score of the image
        review = task['customer_review_status']
        """
        review = ast.literal_eval(review)
        runningScore = 0
        numScores = len(review)
        for i in range(numScores):
            # Load the current entry as a dictionary
            entry = ast.literal_eval(str(review[i]))
            # Add the score of the entry to the running total
            runningScore += entry['score']
        """

        # If the image has a non-positive score, do not download it
        if review != 'accepted':
            print('\nImage ' + task['task_id'] + " was not accepted. Skipping image")
            continue

        '''Check if current image is already downloaded and only new images
           need to be download. If it exists, continue to the next image'''
        if flag == '-n' and os.path.isfile(dir_path + "/Input_Images/" + img_name):
            print(" Skipping Image")
            continue

        # Get the original image
        print(" Getting Original, ", end='')
        params = task['params']
        img_url = params['attachment']
        org_img = generic.getImageFromURL(img_url)  # Retrieve the original image
        new_img = Image.open(org_img[0])
        new_img = new_img.convert("RGB")  # Convert the image to RGB format
        orig_width, orig_height = new_img.size

        # Failed to download the image
        if org_img is None:
            print("Downloading the original image " + str(img_num) + " failed")
            continue

        # Save the original image
        # print("Saving original image")
        new_img = new_img.resize((img_width, img_height))  # Resize the image to be 640x360
        new_img.save(dir_path + "/Input_Images/" + img_name)
        new_img.close()

        print("Generating Mask\tscaleAI")
        # Create a blank image to draw the mask on
        org_mask = np.zeros([orig_height, orig_width, 3], dtype=np.uint8)

        # Get the mask labels
        response = task['response']
        annotations = response['annotations']

        """ 
        freeSpace = row['Label']
        freeSpace = ast.literal_eval(freeSpace)
        freeSpace = freeSpace['Free space']
        """

        # Get each polygon in the mask
        polygons = []

        for annotation in annotations:
            if annotation['label'] == 'free space':
                vertices = annotation['vertices']

                # Form an array of points for the current polygon
                points = []
                for vertex in vertices:
                    x = vertex['x']
                    y = vertex['y']
                    points.append((x, y))

                # Change the points array to a numpy array
                points = np.array(points, dtype=np.int)
                polygons.append(points)

        # GENERIC
        # Draw the mask and save it
        org_mask = cv2.fillPoly(org_mask, polygons, (255, 255, 255), lineType=cv2.LINE_AA)
        new_mask = cv2.resize(org_mask, (img_width, img_height))
        cv2.imwrite(dir_path + "/Image_Masks/" + task['task_id'] + "_mask.png", new_mask)

        # Open the mask using PIL
        new_mask = Image.open(dir_path + "/Image_Masks/" + task['task_id'] + "_mask.png").convert('L')
        mask_data_file = open(dir_path + "/Mask_Data/" + task['task_id'] + "_mask_data.txt", 'w')

        # Get the pixel array and width/height of the original image
        pixels = new_mask.load()
        width, height = new_mask.size

        # Extract the mask data
        # print("Extracting points")
        points = generic.extractMaskPoints(pixels, width, height, num_outputs)

        # Load the image to draw the extracted mask data on for validation
        validation_mask_image = cv2.imread(dir_path + "/Input_Images/" + img_name)

        '''Write the mask data to a file in x,y column format, where y is normalized between 0 and 1 and
           draw the extracted mask points over the original image'''
        x = 0
        step_size = img_width // num_outputs
        # print("Drawing points")
        for y in points:
            # Draw a circle on the original image to validate the correct mask data is extracted
            validation_mask_image = cv2.circle(validation_mask_image, (x, round(y * (height - 1))), 1, (0, 255, 0), -1)

            # Write the mask point to the file
            mask_data_file.write(str(x) + ',' + str(y) + '\n')
            x += step_size

        # Save the overlaid image
        cv2.imwrite(dir_path + "/Mask_Validation/" + task['task_id'] + "_validation_mask.jpg",
                    validation_mask_image)

        # Check if the mask for the current image can be whitelisted
        # print("Validating mask")
        in_valid = generic.checkForBlackEdges(pixels, width, height)
        if not in_valid:
            new_mask.save(dir_path + "/Whitelist_Masks/" + task['task_id'] + "_mask.png")
            white_list.write(task['task_id'] + '.png\n')
        else:
            new_mask.save(dir_path + "/Blacklist_Masks/" + task['task_id'] + "_mask.png")
            print("Potential labeling error for image: " + task['task_id'])
            black_list.write(task['task_id'] + '.png\n')

        mask_data_file.close()
        new_mask.close()

    # UNIQUE
    # image_file.close()

    # GENERIC
    white_list.close()
    black_list.close()

    return


def _main():
    tasks = parse_json('tasks.json')

    for task in tasks:
        print(task['task_id'])
        params = task['params']
        print(params['attachment'])


if __name__ == '__main__':
    _main()
