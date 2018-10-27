from os import listdir
from os.path import isfile, join
import numpy as np
import skimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io
import mywarper

def files_under_folder_with_suffix(dir_name, suffix = ''):
    """
    Return a filename list that under certain folder with suffix
    :param dir_name: folder specified
    :param suffix: suffix of the file name, eg '.jpg'
    :return: List of filenames in order
    """
    files = [f for f in listdir(dir_name) if (isfile(join(dir_name, f)) and f.endswith(suffix))]
    files.sort()
    return files

def X_mean_V_chanel(file_folder, file_names):
    """
    Takes in the data file and return the mean image and the data X
    :param file_folder: "/images"
    :param file_names: image name
    :return: mean_image and data X in V chanel
    """
    mean_image_V = np.zeros((1, 128*128))
    for idx, file in enumerate(file_names):
        im = skimage.io.imread(join(file_folder, file))
        im_hsv = skimage.color.rgb2hsv(im)
        v_vector = flatten(im_hsv[:, :, 2])
        mean_image_V += v_vector
        if idx == 0:
            X = v_vector
        else:
            X = np.concatenate([X, v_vector], axis = 0)

    return mean_image_V / len(file_names), X

def flatten(x):
    x = x.flatten()
    x = np.expand_dims(x, axis = 0)
    return x

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

def reconstruct_image(file_folder, file_names, mean_image, eigen_face):
    for idx, file in enumerate(file_names):
        im = skimage.io.imread(join(file_folder, file))
        im_vector = flatten(im)
        im_hsv = skimage.color.rgb2hsv(im)
        v_chanel = im_hsv[:, :, 2]
        v_chanel = np.expand_dims(v_chanel.flatten(), axis=0) - mean_image
        coef = np.matmul(eigen_face, np.transpose(v_chanel))
        reconstruct = np.matmul(np.transpose(coef), eigen_face) + mean_image
        im_hsv[:, :, 2] = reconstruct.reshape((128, 128))
        re_im = skimage.color.hsv2rgb(im_hsv)
        re_im_vector = flatten(re_im)
        if idx == 0:
            X = im_vector
            X_re = re_im_vector
        else:
            X = np.concatenate([X, im_vector], axis = 0)
            X_re = np.concatenate([X_re, re_im_vector], axis = 0)

    return X, X_re

def reconstructed_loss(file_folder, file_names, mean_image, eigen_face):
    loss = 0
    for idx, file in enumerate(file_names):
        im = skimage.io.imread(join(file_folder, file))
        im_hsv = skimage.color.rgb2hsv(im)
        v_chanel = im_hsv[:, :, 2]
        v_chanel = np.expand_dims(v_chanel.flatten(), axis=0) - mean_image
        coef = np.matmul(eigen_face, np.transpose(v_chanel))
        reconstruct = np.matmul(np.transpose(coef), eigen_face)
        cur_loss = np.square(v_chanel - reconstruct)
        cur_loss = np.sum(cur_loss) / (v_chanel.shape[1])
        loss += cur_loss
    return loss / (idx + 1)

def LM_mean_LM_data(file_folder, file_names):
    """
    Takes in the data file and return the mean image and the data X
    :param file_folder: "/images"
    :param file_names: image name
    :return: mean_image and data X in V chanel
    """
    mean_LM = np.zeros((1, 68*2))
    for idx, file in enumerate(file_names):
        landMark = scipy.io.loadmat(join(file_folder, file))['lms']
        landMark = flatten(landMark)
        mean_LM += landMark
        if idx == 0:
            X = landMark
        else:
            X = np.concatenate([X, landMark], axis = 0)

    return mean_LM / len(file_names), X

def reconstructed_loss_landmark(X_center, eigen_warpings):
    coef = np.matmul(X_center, eigen_warpings.T)
    recons = np.matmul(coef, eigen_warpings)
    loss = np.square(X_center - recons)
    loss = np.sum(loss) / (loss.shape[0] * loss.shape[1])
    return loss

def generate_aligned_images(image_folder, landmark_folder, im_file, LM_file, aligned_folder, target_LM):
    assert len(im_file) == len(LM_file), "Image number and landmark number don't match!"
    for i in range(len(im_file)):
        im = skimage.io.imread(join(image_folder, im_file[i]))
        org_LM = scipy.io.loadmat(join(landmark_folder, LM_file[i]))['lms']
        warp_im = mywarper.warp(im, org_LM, target_LM)
        skimage.io.imsave(join(aligned_folder, im_file[i]), warp_im)