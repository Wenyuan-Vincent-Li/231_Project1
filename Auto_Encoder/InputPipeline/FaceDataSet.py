import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HEIGHT = 128
WIDTH = 128
DEPTH = 3

class FaceDataSet(object):

    def __init__(self, data_dir, config, subset='Train'):
        self.data_dir = data_dir
        self.subset = subset
        self.config = config

    def get_filenames(self):
        if self.subset in ['Train', 'Test']:
            return [os.path.join(self.data_dir, self.subset + '_aligned_image' + '.tfrecords'), \
                    os.path.join(self.data_dir, self.subset + '_landmarks' + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def input_from_tfrecord_filename(self):
        image_file, LM_file = self.get_filenames()
        im_dataset = tf.data.TFRecordDataset(image_file)
        LM_dataset = tf.data.TFRecordDataset(LM_file)
        return [im_dataset, LM_dataset]

    def shuffle_and_repeat(self, dataset):
        dataset = dataset.shuffle(buffer_size= \
                                      self.config.MIN_QUEUE_EXAMPLES + \
                                      3 * self.config.BATCH_SIZE, \
                                  )
        dataset = dataset.repeat(1)
        return dataset

    def batch(self, dataset):
        dataset = dataset.batch(batch_size=self.config.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=self.config.BATCH_SIZE)
        return dataset

    def image_parser(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
            })

        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([HEIGHT * WIDTH * DEPTH])

        image = tf.cast(tf.reshape(image, [HEIGHT, WIDTH, DEPTH]), tf.float32)
        image = tf.divide(image, 255) ## normalize the image to 0-1
        return image

    def LM_parser(self, serialized_example):
        features = tf.parse_single_example(
            serialized_example,
            features={
                'attribute': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
            })

        feature = tf.cast(features['attribute'], tf.float32)
        feature.set_shape([68 * 2])
        ## pre-processing data, normalize the landmarks to 0-1
        feature = self.LM_pre_processing_128(feature)
        return feature

    def LM_pre_processing(self, tensor, epsilon = 1e-12):
        tensor = tf.to_float(tensor)
        scale = [tf.reduce_min(tensor), tf.reduce_max(tensor)]
        tensor = tf.div(
            tf.subtract(
                tensor,
                tf.reduce_min(tensor)
            ) + epsilon,
            tf.math.maximum(tf.subtract(
                tf.reduce_max(tensor),
                tf.reduce_min(tensor)
            ), 2 * epsilon)
        )
        return tensor, scale

    def LM_pre_processing_128(self, tensor):
        return tf.divide(tensor, 128)


    def inputpipline(self):
        # 1 Read in tfrecords
        image_dataset, LM_dataset = self.input_from_tfrecord_filename()

        # 2 Parser tfrecords and preprocessing the data
        image_dataset = image_dataset.map(self.image_parser, \
                              num_parallel_calls=self.config.BATCH_SIZE)
        LM_dataset = LM_dataset.map(self.LM_parser, \
                            num_parallel_calls=self.config.BATCH_SIZE)

        # 3 Shuffle and repeat
        if self.subset == "Train":
            image_dataset = self.shuffle_and_repeat(image_dataset)
            LM_dataset = self.shuffle_and_repeat(LM_dataset)
        else:
            image_dataset = image_dataset.repeat(1)
            LM_dataset = LM_dataset.repeat(1)

        # 4 Batch it up
        image_dataset = self.batch(image_dataset)
        LM_dataset = self.batch(LM_dataset)

        # 5 Make iterator
        im_iterator = image_dataset.make_initializable_iterator()
        image_batch = im_iterator.get_next()

        LM_iterator = LM_dataset.make_initializable_iterator()
        landmark_batch = LM_iterator.get_next()

        init_op = [im_iterator.initializer, LM_iterator.initializer]

        return image_batch, landmark_batch, init_op


def plot(samples,Nh,Nc,channel,IMG_HEIGHT, IMG_WIDTH):
    fig = plt.figure(figsize=(Nc, Nh))
    plt.clf()
    gs = gridspec.GridSpec(Nh, Nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i in range(samples.shape[0]):
        sample = samples[i, :]
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if channel==1:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH)
            immin=(image[:,:]).min()
            immax=(image[:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image,cmap ='gray')
        else:
            image=sample.reshape(IMG_HEIGHT, IMG_WIDTH,channel)
            immin=(image[:,:,:]).min()
            immax=(image[:,:,:]).max()
            image=(image-immin)/(immax-immin+1e-8)
            plt.imshow(image)
    return fig

if __name__ == "__main__":
    root_dir = os.path.dirname(os.getcwd())
    sys.path.append(root_dir)
    from config import Config
    import numpy as np
    data_dir = os.path.join(root_dir, 'DataSet')
    config = Config()
    subset = "Train"
    with tf.device('/cpu:0'):
        DataSet = FaceDataSet(data_dir, config, subset)
        im_batch, lm_batch, init_op = DataSet.inputpipline()

    num_batch = 0

    with tf.Session() as sess:
        sess.run(init_op)
        while True:
            try:
                im_batch_o, lm_batch_o = sess.run([im_batch, lm_batch])
                num_batch += 1
            except tf.errors.OutOfRangeError:
                break;
    print(np.max(lm_batch_o), np.min(lm_batch_o))
    print(num_batch)

