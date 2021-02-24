import cv2
import json
import os

import Data_Extractor.generic as generic
import numpy as np

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


'''Download the image and mask data from the .json file.'''


def download_images(flag,
                    num_outputs,
                    white_list,
                    black_list,
                    json_file,
                    img_width,
                    img_height,
                    dir_path):
    try:
        tasks = parse_json(json_file)
    except:
        print("Error opening file: " + json_file)
        return

    # Download the images and masks from the json file
    img_num = 0
    for task in tasks:
        img_num += 1
        print("Image: " + str(img_num), end='')

        # The name of the original image
        img_name = task['task_id'] + ".jpg"

        # Get the review score of the image
        review = task['customer_review_status']

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
        org_img = None
        for i in range(5):
            try:
                org_img = generic.getImageFromURL(img_url)  # Retrieve the original image
                break
            except:
                # TODO(LUIS): How should we handle this error, if at all?
                pass
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

    return
