'''
This is a python file that used for evaluate the model.
'''
import os, sys
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)

import tensorflow as tf
from Training.Saver import Saver
from InputPipeline.FaceDataSet import FaceDataSet as DataSet
from Testing.eval_base import Evaler_base
from Testing.utils import save_dict_as_txt, initialize_uninitialized_vars, \
    convert_list_2_nparray

class Evaler(Evaler_base):
    def __init__(self, config, save_dir):
        super(Evaler, self).__init__()
        self.config = config
        self.save_dir = save_dir

    def evaler(self, Model, dir_names=None, epoch=None):
        # Reset the tensorflow graph
        tf.reset_default_graph()
        # Input node
        im_batch, lm_batch, init_op = self._input_fn()

        # Build up the train graph
        with tf.name_scope("Test"):
            model = Model(self.config)
            output_list = model.forward(im_batch, lm_batch)

        # Add saver
        saver = Saver(self.save_dir)
        # List to store the results
        Out = []

        # Create a session
        with tf.Session() as sess:
            # restore the weights
            _ = saver.restore(sess, dir_names=dir_names, epoch=epoch)
            # initialize the unitialized variables
            initialize_uninitialized_vars(sess)
            # initialize the dataset iterator
            sess.run(init_op)
            # start evaluation
            count = 1
            while True:
                try:
                    out = \
                        sess.run(output_list)
                    # store results
                    Out.append(out)
                    tf.logging.debug("The current validation sample batch num is {}." \
                                     .format(count))
                    count += 1
                except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError):
                    break
        return Out

    def _input_fn(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                dataset = DataSet(self.config.DATA_DIR, self.config, 'Train')
                im_batch, lm_batch, init_op = dataset.inputpipline()
        return im_batch, lm_batch, init_op

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable all debugging logs
    tf.logging.set_verbosity(tf.logging.INFO)

    from config import Config
    from Model.Auto_encoder import Auto_encoder as Model


    class TempConfig(Config):
        NAME = "Face Auto Encoder"
        ## Input pipeline
        DATA_DIR = os.path.join(root_dir, "DataSet")
        BATCH_SIZE = 100
        SUMMARY = False
        SAVE = False


    # Create a global configuration object
    config = TempConfig()
    ## Specify the trained weights localtion
    save_dir = os.path.join(root_dir, "Training/Weight")  # Folder that saves the trained weights
    Run = None
    epoch = None

    Eval = Evaler(config, save_dir)
    output = Eval.evaler(Model, dir_names=Run, epoch=epoch)
    print(len(output))
    print(output[0][0].shape)