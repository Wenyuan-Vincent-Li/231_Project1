import os, sys
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import numpy as np

import tensorflow as tf
from Training.Saver import Saver
from Testing.utils import initialize_uninitialized_vars

class Interpol(object):
    def __init__(self, config, save_dir):
        self.config = config
        self.save_dir = save_dir

    def interpol(self, Model, im_batch_i, lm_batch_i, dir_names=None, epoch=None):
        # Reset the tensorflow graph
        tf.reset_default_graph()
        # Input node
        im_latent_batch = tf.placeholder(tf.float32, shape = (None, 50), name = "apperance_latent")
        lm_latent_batch = tf.placeholder(tf.float32, shape = (None, 10), name = "landmark_latent")

        # Build up the train graph
        with tf.name_scope("Test"):
            model = Model(self.config)
            output_list = model.forward_latent(im_latent_batch, lm_latent_batch)

        # Add saver
        saver = Saver(self.save_dir)
        # List to store the results

        # Create a session
        with tf.Session() as sess:
            # restore the weights
            _ = saver.restore(sess, dir_names=dir_names, epoch=epoch)
            # initialize the unitialized variables
            initialize_uninitialized_vars(sess)
            out = sess.run(output_list, feed_dict={im_latent_batch:im_batch_i, lm_latent_batch:lm_batch_i})
        return out


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs
    tf.logging.set_verbosity(tf.logging.INFO)

    from config import Config
    from Model.Auto_encoder import Auto_encoder as Model


    class TempConfig(Config):
        NAME = "Face Auto Encoder"
        SUMMARY = False
        SAVE = False


    # Create a global configuration object
    config = TempConfig()
    ## Specify the trained weights localtion
    save_dir = os.path.join(root_dir, "Training/Weight")  # Folder that saves the trained weights
    Run = None
    epoch = None

    im_batch_i = np.zeros((100, 50))
    lm_batch_i = np.zeros((100, 10))
    interpol = Interpol(config, save_dir)
    output = interpol.interpol(Model, im_batch_i, lm_batch_i, dir_names=Run, epoch=epoch)
    print(len(output))
    print(output[0].shape, output[1].shape)