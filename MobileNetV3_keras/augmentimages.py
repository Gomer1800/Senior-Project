import os
import cv2
import sys
import logging
import argparse
import configparser
import numpy as np
import albumentations
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAAPerspective, IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class AugmentImages:
    def __init__(self, num_outputs, p):
        self.num_outputs = num_outputs
        self.alpha = 0
        self.counter = 0
        self.p = p

    def augmentation_pipeline(self, p=0.5):
        return Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                #MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.1),
            OneOf([
               ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=5, p=0.9),
               IAAPerspective(scale=(.02,.05))
            ], p=0.3)
        ], p=p)

    '''This function takes the path to the image and the path to the image
    maks.  It returns the augmented image/mask and the output data'''
    def augment_image(self, path, mask_path):
        img = cv2.imread(path) #read in the image
        img_mask = cv2.imread(mask_path) #read in the mask

        if img is None:
            raise '** Failed to read image.'
        if img_mask is None:
            raise '** Failed to read mask image.'

        # convert the image to RGB
        img_copy = img.copy()
        img_copy = img_copy[:, :, ::-1]

        #run the augmentation
        augmentation = self.augmentation_pipeline(self.p)
        data = {"image": img_copy, "mask": img_mask}
        augmented = augmentation(**data)

        mask = augmented["mask"]

        data_list = []
        height = mask.shape[0]
        width = mask.shape[1]
        #print (self.num_outputs)

        for i in range(0,width,int(width/self.num_outputs)): #run through width, 0 to 639
            found = 0
            for row in range(height-1,-1,-1): #run from bottom (359) to top (0)
                point = mask[row,i] #point is a 3-element list (RGB)
                if point[0] < 128: #if the color is black, then save the point
                    found = 1
                    if row < (height-1):
                        data_list.append(float(row+1) / float(height))
                    else:
                        data_list.append(1.0)
                    break
            if found == 0:
                data_list.append(float(0.0))

        #print (data_list)
        try:
            assert len(data_list) == self.num_outputs
        except:
            print ("length of list: " + str(len(data_list)))
            print (data_list)
            augmented["image"] = img_copy
            self.write_image(augmented)
            sys.exit()

        #debug
        # i = augmented["image"]
        # i = i[:, :, ::-1]

        # f = open("./test" + str(self.counter) + ".txt", "w")
        # for x in range(len(data_list)):
        #     y = int(data_list[x] * height)
        #     x1 = x * 5

        #     f.write(str(x1) + "," + str(y) + '\n')

        #     #Draw a circle on the original image to validate the correct mask data is extracted
        #     # cv2.circle(cv2.UMat(tempimg), (x1, y), 1, (0, 255, 0), -1)
        #     #print ((x,y))
        # f.close()
        # cv2.imwrite("./test" + str(self.counter) + ".jpg", i)
        #end debug

        #normalize the image data
        i = augmented["image"]
        #augmented["image"] = preprocess_input(i.astype(float))
        augmented["image"] = i / 255.0

        return augmented, data_list

    def write_image(self, augmented):
        #Save the image and mask
        cv2.imwrite("./test.jpg", augmented["image"])
        cv2.imwrite("./test_mask.jpg", augmented["mask"])

    def save_image(self, img, f, data):
        #Save the image and mask
        img= img[:, :, ::-1]
        img = img * 255.0
        cv2.imwrite("./sample_img/"+f, img)

    def _main(self, args):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        # ** get configuration
        config_file = args.conf
        config_client = configparser.ConfigParser()
        config_client.read(config_file)

        # ** MobileNet V3 configuration
        input_width = config_client.getint('model', 'input_width')
        input_height = config_client.getint('model', 'input_height')


        image_list = os.listdir("../Input_Images/")
        image_list.sort()

        #run one prediction
        for f in image_list:
            if f.endswith(".jpg"):
                test_file = '../Input_Images/' + f
                test_mask = '../Image_Masks/' + f.replace('.','_mask.')
                print(test_mask)
                img,data = self.augment_image(test_file, test_mask)
                self.save_image(img["image"],f,data)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    a = AugmentImages(128,.5)
    a._main(args)

