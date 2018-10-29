import sys, os
root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import tensorflow as tf
from time import strftime
from datetime import datetime
from pytz import timezone


from train_base import Train_base
from Training.Saver import Saver
from InputPipeline.FaceDataSet import FaceDataSet as DataSet
from Training.Summary import Summary


class Train(Train_base):
    def __init__(self, config, log_dir, save_dir, **kwargs):
        super(Train, self).__init__(config.LEARNING_RATE, config.MOMENTUM)
        self.config = config
        self.save_dir = save_dir
        self.comments = kwargs.get('comments', '')
        if self.config.SUMMARY:
            if self.config.SUMMARY_TRAIN_VAL:
                self.summary_train = Summary(log_dir, config, log_type = 'train', \
                                    log_comments = kwargs.get('comments', ''))
                self.summary_val = Summary(log_dir, config, log_type = 'val', \
                                    log_comments = kwargs.get('comments', ''))
            else:
                self.summary = Summary(log_dir, config, \
                                     log_comments = kwargs.get('comments', ''))

    def train(self, Model):
        tf.reset_default_graph()
        # Input node
        im_batch, lm_batch, init_op = self._input_fn()

        # Build up the train graph
        with tf.name_scope("Train"):
            model = Model(self.config)
            out_im_batch, _, out_lm_batch, _ = model.forward(im_batch, lm_batch)
            im_loss, lm_loss = self._loss(im_batch, out_im_batch, lm_batch, out_lm_batch)
            optimizer = self._Adam_optimizer()
            im_solver, im_grads = self._train_op_w_grads(optimizer, im_loss)
            lm_solver, lm_grads = self._train_op_w_grads(optimizer, lm_loss)

        # Add summary
        if self.config.SUMMARY:
            summary_dict_train = {}
            if self.config.SUMMARY_SCALAR:
                scalar_train = {'appearance_loss': im_loss, \
                                'landmarks_loss': lm_loss}
                summary_dict_train['scalar'] = scalar_train

            # Merge summary
            merged_summary_train = \
                self.summary_train.add_summary(summary_dict_train)

        # Add saver
        saver = Saver(self.save_dir)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        # Use soft_placement to place those variables, which can be placed, on GPU
        # Create Session
        with tf.Session(config=sess_config) as sess:
            # Add graph to tensorboard
            if self.config.SUMMARY and self.config.SUMMARY_GRAPH:
                self.summary_train._graph_summary(sess.graph)

            # Restore the weights from the previous training
            if self.config.RESTORE:
                start_epoch = saver.restore(sess)
            else:
                # Create a new folder for saving model
                saver.set_save_path(comments=self.comments)
                start_epoch = 0

            # initialize the variables
            init_var = tf.group(tf.global_variables_initializer(), \
                                tf.local_variables_initializer())
            sess.run(init_var)

            # Start Training
            tf.logging.info("Start training!")
            for epoch in range(1, self.config.EPOCHS + 1):
                tf.logging.info("Training for epoch {}.".format(epoch))
                train_pr_bar = tf.contrib.keras.utils.Progbar(target= \
                                                                  int(800 / self.config.BATCH_SIZE))
                sess.run(init_op)
                batch = 0
                while True:
                    try:
                        im_loss_o, lm_loss_o, summary_o, _, _ = sess.run([im_loss, lm_loss, merged_summary_train, \
                                                                          im_solver, lm_solver])
                        batch += 1
                        train_pr_bar.update(batch)

                        if self.config.SUMMARY:
                            # Add summary
                            self.summary_train.summary_writer.add_summary(summary_o, epoch + start_epoch)

                    except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError):
                        break

                tf.logging.info(
                    "\nThe current epoch {}, appearance loss is {:.2f}, landmark loss is {:.2f}.\n" \
                    .format(epoch, im_loss_o, lm_loss_o))
                # Save the model per SAVE_PER_EPOCH
                if epoch % self.config.SAVE_PER_EPOCH == 0:
                    save_name = str(epoch + start_epoch)
                    saver.save(sess, 'model_' + save_name.zfill(4) \
                               + '.ckpt')

            if self.config.SUMMARY:
                self.summary_train.summary_writer.flush()
                self.summary_train.summary_writer.close()

            # Save the model after all epochs
            save_name = str(epoch + start_epoch)
            saver.save(sess, 'model_' + save_name.zfill(4) + '.ckpt')
        return

    def _input_fn(self):
        with tf.device('/cpu:0'):
            with tf.name_scope('Input_Data'):
                dataset = DataSet(self.config.DATA_DIR, self.config, 'Train')
                im_batch, lm_batch, init_op = dataset.inputpipline()
        return im_batch, lm_batch, init_op

    def _loss(self, im_batch, out_im_batch, lm_batch, out_lm_batch):
        im_loss = self._mean_squared_error(im_batch, out_im_batch)
        lm_loss = self._mean_squared_error(lm_batch, out_lm_batch)
        return im_loss, lm_loss

if __name__ == "__main__":
    from config import Config
    from Model.Auto_encoder import Auto_encoder as Model

    tf.logging.set_verbosity(tf.logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable all debugging logs


    class TempConfig(Config):
        NAME = "Face Auto Encoder"

        ## Input pipeline
        DATA_DIR = os.path.join(root_dir, "DataSet")
        BATCH_SIZE = 100

        ## Training settings
        # Restore
        RESTORE = False  # Whether to use the previous trained weights

        # Training schedule
        EPOCHS = 300  # Num of epochs to train in the current run
        SAVE_PER_EPOCH = 50  # How often to save the trained weights

    config = TempConfig()
    save_dir = os.path.join(root_dir, "Training/Weight")
    log_dir = os.path.join(root_dir, "Training/Log")
    comments = "This training is for 231 Project."
    comments += config.config_str() + datetime.now(timezone('America/Los_Angeles')).strftime("%Y-%m-%d_%H_%M_%S")
    # Create a training object
    training = Train(config, log_dir, save_dir, comments=comments)
    # Train the model
    training.train(Model)