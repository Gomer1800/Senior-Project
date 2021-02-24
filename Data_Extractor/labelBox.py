import ast
import csv
import cv2
import os

import Data_Extractor.generic as generic
import numpy as np

from PIL import Image

'''Download the image and mask data from the .csv file.'''


def download_images(flag,
                    num_outputs,
                    white_list,
                    black_list,
                    csv_file,
                    img_width,
                    img_height,
                    dir_path):
    try:
        image_file = open(csv_file, 'r')  # Open the csv file
    except:
        print("Error opening file: " + csv_file)
        return

    reader = csv.DictReader(image_file)
    # Download the images and masks from the csv file
    img_num = 0
    for row in reader:
        img_num += 1
        print("Image: " + str(img_num), end='')

        # The name of the original image
        img_name = row['ID'] + ".jpg"

        # Get the review score of the image
        review = row['Reviews']
        review = ast.literal_eval(review)
        running_score = 0
        num_scores = len(review)
        for i in range(num_scores):
            # Load the current entry as a dictionary
            entry = ast.literal_eval(str(review[i]))
            # Add the score of the entry to the running total
            running_score += entry['score']

        # If the image has a non-positive score, do not download it
        if running_score <= 0:
            print('\nImage ' + row['ID'] + " has a non-positive score. Skipping image")
            continue

        '''Check if current image is already downloaded and only new images
           need to be download. If it exists, continue to the next image'''
        if flag == '-n' and os.path.isfile(dir_path + "/Input_Images/" + img_name):
            print(" Skipping Image")
            continue

        # Get the original image
        print(" Getting Original, ", end='')
        img_url = row['Labeled Data']
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

        print("Generating Mask\tlabelBox")
        # Create a blank image to draw the mask on
        org_mask = np.zeros([orig_height, orig_width, 3], dtype=np.uint8)

        # Get the mask labels
        free_space = row['Label']
        free_space = ast.literal_eval(free_space)
        free_space = free_space['Free space']

        # Get each polygon in the mask
        polygons = []
        num_polygons = len(free_space)
        for i in range(num_polygons):
            # Get the dictionary storing the points for the current polygon
            geometry = ast.literal_eval(str(free_space[i]))
            geometry = geometry['geometry']
            num_points = len(geometry)

            # Form an array of points for the current polygon
            points = []
            for p in range(num_points):
                point = ast.literal_eval(str(geometry[p]))
                x = point['x']
                y = point['y']
                points.append((x, y))

            # Change the points array to a numpy array
            points = np.array(points)
            polygons.append(points)

        # Draw the mask and save it
        org_mask = cv2.fillPoly(org_mask, polygons, (255, 255, 255), lineType=cv2.LINE_AA)
        new_mask = cv2.resize(org_mask, (img_width, img_height))
        cv2.imwrite(dir_path + "/Image_Masks/" + row['ID'] + "_mask.png", new_mask)

        # Open the mask using PIL
        new_mask = Image.open(dir_path + "/Image_Masks/" + row['ID'] + "_mask.png").convert('L')

        mask_data_file = open(dir_path + "/Mask_Data/" + row['ID'] + "_mask_data.txt", 'w')
        # Get the pixel array and witdh/height of the original image
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
        cv2.imwrite(dir_path + "/Mask_Validation/" + row['ID'] + "_validation_mask.jpg",
                    validation_mask_image)

        # Check if the mask for the current image can be whitelisted
        # print("Validating mask")
        in_valid = generic.checkForBlackEdges(pixels, width, height)
        if not in_valid:
            new_mask.save(dir_path + "/Whitelist_Masks/" + row['ID'] + "_mask.png")
            white_list.write(row['ID'] + '.png\n')
        else:
            new_mask.save(dir_path + "/Blacklist_Masks/" + row['ID'] + "_mask.png")
            print("Potential labeling error for image: " + row['ID'])
            black_list.write(row['ID'] + '.png\n')

        mask_data_file.close()
        new_mask.close()

    image_file.close()

    return
