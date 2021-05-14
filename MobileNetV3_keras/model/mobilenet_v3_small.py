"""MobileNet v3 small models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Reshape, Dropout
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import plot_model

from .mobilenet_base import MobileNetBase

class MobileNetV3_Small(MobileNetBase):
    def __init__(self, shape, num_outputs, alpha=1.0, include_top=True):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if include classification layer.

        # Returns
            MobileNetv3 model.
        """
        super(MobileNetV3_Small, self).__init__(shape, num_outputs, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)

        #x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')
        x = self._conv_block(inputs, 32, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')

        #x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')

        #x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')

        x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')

        #x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=240, s=2, squeeze=True, nl='HS')

        #x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=400, s=1, squeeze=True, nl='HS')

        #x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=400, s=1, squeeze=True, nl='HS')

        #x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        x = self._conv_block(x, 500, (1, 1), strides=(1, 1), nl='HS')

        #x = Dropout(0.3, name='Dropout')(x)


        x = GlobalAveragePooling2D()(x)
        #x = Reshape((1, 1, 576))(x)
        x = Reshape((1, 1, 500))(x)

        x = Dropout(0.3, name='Dropout')(x)

        #x = Conv2D(1280, (1, 1), padding='same')(x)
        x = Conv2D(1100, (1, 1), padding='same')(x)

        x = Dropout(0.3, name='Dropout1')(x)

        x = self._return_activation(x, 'HS')

        if self.include_top:
            #testing
            #x = Dense(300, activation='relu')(x)
            #x = Dense(600, activation='relu')(x)
            x = Dense(self.num_outputs, activation='linear')(x)
            # x = Flatten()(x)

            # x = Conv2D(self.num_outputs, (1, 1), padding='same', activation='linear')(x)
            x = Reshape((self.num_outputs,))(x)

            # x = Flatten()(x)
            # x = Dense(self.num_outputs, activation='linear')(x)

        model = Model(inputs, x)

        return model
