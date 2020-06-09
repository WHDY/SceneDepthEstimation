import os
import time
import cv2
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
from tqdm import tqdm
from process_functional import *


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="stereo matching based on trained model and post-processing")

parser.add_argument("-g", "--gpu", type=str, default="1,2", help="gpu id to use, \
                    multiple ids should be separated by commons(e.g. 0,1,2,3)")
parser.add_argument("-i", "--id", type=int, default=0, help="image_id")
parser.add_argument("-f", "--file", type=str, default="11_11", help="file to save result")

def main():

    args = parser.parse_args()

    # GPU preparation
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    id = args.id
    file_path = args.file

    # image_path = r'./test_data/test_r/'
    image_path = r'./eval/'
    left_image_path = os.path.join(image_path, 'left_{}.png'.format(id))
    right_image_path = os.path.join(image_path, 'right_{}.png'.format(id))

    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    _left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    _right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
    right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))
    left_image = np.expand_dims(left_image, axis=2)
    right_image = np.expand_dims(right_image, axis=2)

    # left_image, right_image = preprocess_image_with_epipolar_transform(_left_image, _right_image, left_image, right_image)
    left_feature, right_feature = compute_feature(left_image, right_image, 11, 11, 64, r'./check_points_11_11/model_epoch14.ckpt')
    # left_cost_volume, right_cost_volume = disparity_compute_by_gpu(_left_image, _right_image, left_feature, right_feature)
    detail_time = np.zeros(shape=[7], dtype=np.float32)
    left_disparity, right_disparity, detail_time = disparity_compute_by_gpu(_left_image, _right_image, left_feature,right_feature, detail_time)
    
    # left_cost_volume = compute_cost_volume(left_feature, right_feature, 128)

    # left_disparity = WTA1(left_cost_volume)

    cv2.imwrite('./result/{}/ld{}.png'.format(file_path, id), left_disparity.astype('uint8'))
    # cv2.imwrite('./disparity/rd{}.png', right_disparity.astype('uint8'))
    #     # cv2.imshow('disparity', left_disparity.astype('uint8'))
    #     # cv2.waitKey(0)

if __name__=="__main__":
    main()

