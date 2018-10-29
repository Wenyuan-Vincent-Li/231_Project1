import os, sys
import numpy as np
import tensorflow as tf
import scipy.io

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _read_and_process(mat_filename):
    landMark = scipy.io.loadmat(mat_filename)['lms']
    landMark = landMark.flatten()
    return landMark

def mat_2_TF(mat_folder, mat_filenames, tf_filename, LM_mean = False):
  '''
  A function that conver csv_data to tfrecord.
  Note that the csv_data is reqired to have the format:
  label, feature 0, feature 1, ... , feature n.
  '''
  with tf.python_io.TFRecordWriter(tf_filename) as writer:
    mean_LM = np.zeros((68 * 2))
    for mat_filename in mat_filenames:
      landmark = _read_and_process(os.path.join(mat_folder, mat_filename))
      example = tf.train.Example(features=tf.train.Features(
                feature={
                    'attribute': _float_feature(landmark)
                    }))
      mean_LM += landmark
      writer.write(example.SerializeToString())
    if LM_mean:
        mean_LM = mean_LM / len(mat_filenames)
        scipy.io.savemat("LM_mean.mat", mdict = {"LM_mean": mean_LM})


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    sys.path.append(root_dir)
    import utils
    mat_folder = os.path.join(root_dir, "landmarks")
    mat_files = utils.files_under_folder_with_suffix(mat_folder, suffix = '.mat')
    train_mat_file = mat_files[:800]
    test_mat_file = mat_files[800:]
    mat_2_TF(mat_folder, train_mat_file, "Train_landmarks.tfrecords")
    mat_2_TF(mat_folder, test_mat_file, "Test_landmarks.tfrecords")