import os, sys
import numpy as np
import tensorflow as tf
import scipy.io
import skimage

# Helper functions for defining tf types
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _image_channel_mean(image, accumulated_mean):
    '''
    Compute the accumulated channel mean
    Input: image -- current image
           accumulated_mean -- current accumulated mean
    '''
    accumulated_mean += np.mean(image,axis = (0,1))
    return accumulated_mean

def _image_read_and_process(image_file):
    im = skimage.io.imread(image_file)
    # im = im.flatten()
    return im

def image_2_tfrecord(image_folder, image_filenames, tfrecords_filename, CH_mean = False):
    '''
    Convert image data and label to tfrecord
    '''
    with tf.python_io.TFRecordWriter(tfrecords_filename) as record_writer:
        accumulated_mean = np.zeros(3)
        for image_file in image_filenames:
            image = _image_read_and_process(os.path.join(image_folder, image_file))
            ## compute the image channel_mean
            accumulated_mean = _image_channel_mean(image, accumulated_mean)
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(image.tobytes())
                    }))
            record_writer.write(example.SerializeToString())
        if CH_mean:
            channel_mean = accumulated_mean / len(image_filenames)
            scipy.io.savemat("channel_mean.mat", mdict = {"channel_mean": channel_mean})

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    sys.path.append(root_dir)
    import utils
    image_folder = os.path.join(root_dir, "aligned_images")
    image_files = utils.files_under_folder_with_suffix(image_folder, suffix = '.jpg')
    train_image_file = image_files[:800]
    test_image_file = image_files[800:]
    image_2_tfrecord(image_folder, train_image_file, "Train_aligned_image.tfrecords", True)
    image_2_tfrecord(image_folder, test_image_file, "Test_aligned_image.tfrecords")