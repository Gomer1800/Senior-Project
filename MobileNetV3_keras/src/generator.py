import os
import cv2
import sys
import tensorflow.keras as keras
import numpy as np
import threading
from tensorflow.keras.utils import Sequence
from MobileNetV3_keras.augmentimages import AugmentImages

class DataGenerator(Sequence):
    def __init__(self, dir_path, batch_size=16, aug_freq=0.5, image_width=640, image_height=360, shuffle=True, n=128):
        self.dir_path = dir_path
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.num_outputs = n
        self.lock = threading.Lock()
        self.aug_freq = aug_freq #probability of augmenting an image
        self.shuffle = shuffle #if true, randomized the files order each epoch
        self.augment = AugmentImages(self.num_outputs, self.aug_freq)

        self.__get_all_paths() #get all filenames and output data
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.paths_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __get_all_paths(self):
        if not os.path.exists(self.dir_path):
            raise 'input dir does not exist'
        print('** start getting images/labels path.......')
        self.paths_list = []
        self.label_list = []
        for i, _id in enumerate(os.listdir(self.dir_path)):
            if not 'output' in _id: #skip the directory Augmentor was creating
                #add the image filename to the list
                _id_path = os.path.join(self.dir_path, _id)
                self.paths_list.append(_id_path)

                #read the data from the mask file
                f = open("../Mask_Data/"+_id.replace(".jpg","")+"_mask_data.txt","r")
                temp_list = []
                for x in f:
                    x = x.split(",")[1]
                    x = x.strip()
                    temp_list.append(float(x))

                self.label_list.append(temp_list) #store the output data

        return None

    def __len__(self):
        return int(np.floor(len(self.paths_list) / self.batch_size))

    def __getitem__(self, index):
        #with self.lock:
        # ** get batch indices
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.paths_list))
        indices = self.indices[start_index:end_index]

        # ** get batch inputs
        paths_batch_list = [self.paths_list[i] for i in indices]
        label_batch_list = [self.label_list[i] for i in indices]

        assert len(paths_batch_list) == len(label_batch_list)

        # ** generate batch data
        X, y = self.__data_generation(paths_batch_list, label_batch_list)

        return X, y

    def __data_generation(self, paths_batch_list, label_batch_list):
        X = np.zeros((len(paths_batch_list), self.image_height, self.image_width, 3), dtype='float32')
        y = np.zeros((len(label_batch_list), self.num_outputs), dtype='float32')

        for i, (path, label) in enumerate(zip(paths_batch_list, label_batch_list)):
        #for i, path in enumerate(paths_batch_list):
            # for each batch, X is the array of image data and y is the array
            # of output data
            mask_path = path
            mask_path = mask_path.replace('.jpg','_mask.png')
            mask_path= '../Image_Masks/' + mask_path.split('/')[2]
            #print (mask_path)

            a,l = self.augment.augment_image(path,mask_path)
            X[i, :, :, :] = a["image"]
            y[i, :] = np.array(l)

        return X, y

    def __image_augmentation(self, img, label):
        if img is None:
            raise '** Failed to read image.'
        # to rgb
        img_copy = img.copy()
        img_copy = img_copy[:, :, ::-1]

        # do augmentation
        if self.aug_freq > np.random.uniform(0, 1, 1):
            img_aug, label = self.__augmentation_operations(img_copy, label)
        else:
            img_aug = img_copy

        img_norm = self.__normalize(img_aug)

        return img_norm, label

    def __augmentation_operations(self, img, label):
        # kps = KeypointsOnImage([Keypoint(x=int(i*5), y=int(label[i]*img.shape[0])) \
        #     for i in range(len(label))], shape=img.shape)
        # print (kps)

        seq = iaa.Sequential([
            #iaa.PerspectiveTransform(scale=(.01, .15)) # change brightness, doesn't affect keypoints
            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
        ])

        # Augment keypoints and images.
        #image_aug, kps_aug = seq(image=img, keypoints=kps)

        return img,label

    def __normalize(self, img):
        return img / 255.0

    def __sometimes(self, aug, prob=0.5):
        return iaa.Sometimes(prob, aug)
