import os
import sys
import logging
import argparse
import configparser
import numpy as np
from tensorflow.keras.models import load_model
from src.generator import DataGenerator
from src.MobileNet_V3 import build_mobilenet_v3
from tensorflow.keras.optimizers import (Adam, RMSprop)
from tensorflow.keras.callbacks import (ModelCheckpoint,
                             LearningRateScheduler,
                             ReduceLROnPlateau,
                             EarlyStopping)

from model.mobilenet_v3_small import MobileNetV3_Small
# from model.mobilenet_v2 import MobileNetv2

from model.mobilenetv2_pretrained import MobileNetV2_Pretrained

logging.basicConfig(level=logging.INFO)

def _main(args):

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** get configuration
    config_file = args.conf
    config_client = configparser.ConfigParser()
    config_client.read(config_file)

    # ** set gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = config_client.get('gpu', 'gpu')

    # ** MobileNet V3 configuration
    input_width = config_client.getint('model', 'input_width')
    input_height = config_client.getint('model', 'input_height')
    model_size = config_client.get('model', 'model_size')
    pooling_type = config_client.get('model', 'pooling_type')
    num_outputs = config_client.getint('model', 'num_outputs')

    # ** training configuration
    epochs = config_client.getint('train', 'epochs')
    batch_size = config_client.getint('train', 'batch_size')
    save_path = config_client.get('train', 'save_path')

    # ** Dataset
    train_directory = config_client.get('data', 'train')
    valid_directory = config_client.get('data', 'valid')

    # ** initialize data generators
    train_generator = DataGenerator(dir_path=train_directory, batch_size=batch_size, aug_freq=0.5, image_width=input_width, image_height=input_height, n=num_outputs)
    valid_generator = DataGenerator(dir_path=valid_directory, batch_size=batch_size, aug_freq=0.0, image_width=input_width, image_height=input_height, n=num_outputs)

    #model_test = MobileNetV3_Small((input_height,input_width,3), num_outputs).build()

    #model_test = MobileNetv2((input_height,input_width,3), num_outputs)

    #pretrained model test
    model_test = MobileNetV2_Pretrained(shape = (input_width,input_height,3), num_outputs=num_outputs)
    model_test = model_test.build()

    # ** initalize model

    rmsprop = RMSprop(lr=0.001, rho=0.9)
    #model.compile(optimizer=Adam(lr=3e-3), loss='mean_absolute_error', metrics=['accuracy'])
    #model.compile(optimizer=Adam(), loss='mean_absolute_error', metrics=['accuracy'])

    #model_test.compile(optimizer=Adam(), loss='mean_absolute_error', metrics=['accuracy'])
    model_test.compile(optimizer=rmsprop, loss='mean_absolute_error', metrics=['accuracy'])

    # ** setup keras callback
    #filename = 'ep{epoch:03d}-loss{loss:.3f}.h5'
    myhost = os.uname()[1]
    filename = str(myhost.split('.')[0]) + '_ep-loss.h5'
    weights_directory = os.path.join(ROOT_DIR, 'weights')

    if not os.path.exists(weights_directory):
      os.mkdir(weights_directory)

    save_path = os.path.join(weights_directory, filename)
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, period=1, verbose=1)
    #scheduler = LearningRateScheduler(learning_rate_scheduler)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1)

    #print model information
    print(model_test.summary())

    # ** start training
    model_test.fit_generator(
                        generator       = train_generator,
                        validation_data = valid_generator,
                        steps_per_epoch = 500,
                        validation_steps =200,
                        epochs          = epochs,
                        callbacks       = [checkpoint],
                        use_multiprocessing=True,
                        workers=4
                        )

    model_test.save(os.path.join(ROOT_DIR, save_path))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    _main(args)

