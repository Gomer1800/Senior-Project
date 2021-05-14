"""MobileNet v3 models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape, Flatten

from tensorflow.keras import backend as K
from tensorflow.keras import Input

class MobileNetV2_Pretrained:
    def __init__(self, shape, num_outputs, alpha=1.0):
        """Init

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
        """
        self.shape = shape
        self.num_outputs = num_outputs
        self.alpha = alpha

    def build(self):
        base_model = MobileNetV2(input_tensor=Input(shape=(360,640,3)), \
         alpha=.75, include_top=False, weights='imagenet', pooling='avg')
        #freeze_weights(base_model)
        #for layer in base_model.layers[:154]:
        #    layer.trainable=False

        #for i,layer in enumerate(base_model.layers):
        #    print(i,layer.name)

        x = base_model.output

        #testing
        x = Dense(200, activation='relu')(x)
        #x = Conv2D(8, (1, 1), padding='same')(x)
        #x = Flatten()(x)

        #x = Dropout(0.2, name='Dropout1')(x)
        x = Dense(self.num_outputs, activation='linear')(x)
        x = Reshape((self.num_outputs,))(x)

        model = Model(inputs=base_model.inputs, outputs=x)
        return model

if __name__ == '__main__':
    m = MobileNetV2_Pretrained(shape = (360,640,3), num_outputs=128)
    model = m.build()

    print(model.summary())
