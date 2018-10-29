import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from Model import model_base

class Auto_encoder(model_base.NN_Base):
    def __init__(self, config):
        self.is_training = None
        super(Auto_encoder, self).__init__(self.is_training,\
              config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
        self._batch_size = config.BATCH_SIZE
        self.config = config
        self.appearance_autoencoder = appearance_autoencoder(config, self.is_training)
        self.landmark_autoencoder = landmark_autoencoder(config, self.is_training)

    def forward(self, input_im, input_lm):
        output_im, latent_im = self.appearance_autoencoder.forward(input_im)
        output_lm, latent_lm = self.landmark_autoencoder.forward(input_lm)
        return output_im, latent_im, output_lm, latent_lm

    def forward_latent(self, latent_im, latent_lm):
        output_im = self.appearance_autoencoder.forward_latent(latent_im)
        output_lm = self.landmark_autoencoder.forward_latent(latent_lm)
        return output_im, output_lm

class appearance_autoencoder(model_base.NN_Base):
    def __init__(self, config, is_training):
        self.is_training = is_training
        super(appearance_autoencoder, self).__init__(self.is_training,\
              config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
        self._batch_size = config.BATCH_SIZE
        self.config = config

    def _encoder(self, input_x):
        with tf.device('/gpu:0'):
            with tf.name_scope('Conv_Block_0'):
                input_x = self._conv(input_x, filters = 16, \
                                     kernel_size = 5, strides = (2, 2), name = "app_conv0",\
                                     padding='SAME')
                input_x = self._leakyrelu(input_x, leak=0.2, name="app_lrelu0")

            with tf.name_scope('Conv_Block_1'):
                input_x = self._conv(input_x, filters=32, \
                                     kernel_size=3, strides=(2, 2), name="app_conv1",\
                                     padding='SAME')
                input_x = self._leakyrelu(input_x, leak=0.2, name="app_lrelu1")

            with tf.name_scope('Conv_Block_2'):
                input_x = self._conv(input_x, filters=64, \
                                     kernel_size=3, strides=(2, 2), name="app_conv2",\
                                     padding='SAME')
                input_x = self._leakyrelu(input_x, leak=0.2, name="app_lrelu2")

            with tf.name_scope('Conv_Block_3'):
                input_x = self._conv(input_x, filters=128, \
                                     kernel_size=3, strides=(2, 2), name="app_conv3",\
                                     padding='SAME')
                input_x = self._leakyrelu(input_x, leak=0.2, name="app_lrelu3")
            with tf.name_scope('FC1'):
                input_x = tf.layers.flatten(input_x, name='app_input_flatten')
                input_x = self._fully_connected(input_x, 50, name = 'app_input_fc')
                latent_x = self._leakyrelu(input_x, leak=0.2, name="app_lrelu4")
        return latent_x


    def _decoder(self, latent_x):
        with tf.device('/gpu:0'):
            with tf.name_scope('FC2'):
                latent_x = self._fully_connected(latent_x, 8*8*128, name = 'app_output_fc')
                latent_x = self._leakyrelu(latent_x, leak=0.2, name="app_lrelu5")
                latent_x = tf.reshape(latent_x, shape=[tf.shape(latent_x)[0], 8, 8, 128], name="app_output_reshape")

            with tf.name_scope('DeConv_Block_0'):
                latent_x = self._deconv(latent_x, filters=128, \
                                     kernel_size=8, strides=(1, 1), name="app_deconv0", \
                                     padding='same')
                latent_x = self._leakyrelu(latent_x, leak=0.2, name="app_delrelu0")

            with tf.name_scope('DeConv_Block_1'):
                latent_x = self._deconv(latent_x, filters=64, \
                                     kernel_size=3, strides=(2, 2), name="app_deconv1", \
                                     padding='same')
                latent_x = self._leakyrelu(latent_x, leak=0.2, name="app_delrelu1")

            with tf.name_scope('DeConv_Block_2'):
                latent_x = self._deconv(latent_x, filters=32, \
                                     kernel_size=3, strides=(2, 2), name="app_deconv2", \
                                     padding='same')
                latent_x = self._leakyrelu(latent_x, leak=0.2, name="app_delrelu2")

            with tf.name_scope('DeConv_Block_3'):
                latent_x = self._deconv(latent_x, filters=16, \
                                     kernel_size=3, strides=(2, 2), name="app_deconv3", \
                                     padding='same')
                latent_x = self._leakyrelu(latent_x, leak=0.2, name="app_delrelu3")

            with tf.name_scope('DeConv_Block_4'):
                latent_x = self._deconv(latent_x, filters=3, \
                                     kernel_size=5, strides=(2, 2), name="app_deconv4", \
                                     padding='same')
                output_x = tf.nn.sigmoid(latent_x, name = "app_output_sigmoid")

        return output_x

    def forward(self, input_x):
        latent_x = self._encoder(input_x)
        output_x = self._decoder(latent_x)
        return output_x, latent_x

    def forward_latent(self, latent_x):
        return self._decoder(latent_x)

class landmark_autoencoder(model_base.NN_Base):
    def __init__(self, config, is_training):
        self.is_training = is_training
        super(landmark_autoencoder, self).__init__(self.is_training, \
                                                   config.BATCH_NORM_DECAY, config.BATCH_NORM_EPSILON)
        self._batch_size = config.BATCH_SIZE
        self.config = config

    def _encoder(self, input_x):
        with tf.device('/gpu:0'):
            with tf.name_scope('FC_0'):
                input_x = self._fully_connected(input_x, 100, name = 'lm_fc0')
                input_x = self._leakyrelu(input_x, leak=0.2, name="lm_lrelu0")
            with tf.name_scope('FC_1'):
                input_x = self._fully_connected(input_x, 10, name = 'lm_fc1')
                latent_x = self._leakyrelu(input_x, leak=0.2, name="lm_lrelu1")
        return latent_x

    def _decoder(self, latent_x):
        with tf.device('/gpu:0'):
            with tf.name_scope('FC_3'):
                latent_x = self._fully_connected(latent_x, 100, name = 'lm_fc3')
                latent_x = self._leakyrelu(latent_x, leak=0.2, name="lm_lrelu3")
            with tf.name_scope('Sigmoid'):
                latent_x = self._fully_connected(latent_x, 68*2, name = 'lm_fc4')
                output_x = tf.nn.sigmoid(latent_x, name="lm_output_sigmoid")
        return output_x


    def forward(self, input_x):
        latent_x = self._encoder(input_x)
        output_x = self._decoder(latent_x)
        return output_x, latent_x

    def forward_latent(self, latent_x):
        return self._decoder(latent_x)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.getcwd())
    sys.path.append(root_dir)
    from config import Config

    config = Config()
    tf.reset_default_graph()
    model = Auto_encoder(config)
    input_im = tf.ones([100, 128, 128, 3])
    input_lm = tf.ones([100, 136])
    output_im, latent_im, output_lm, latent_lm = model.forward(input_im, input_lm)
    print(output_im, latent_im, output_lm, latent_lm)