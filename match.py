import os
import time
import cv2
import numpy as np
import tensorflow as tf
import argparse
from datetime import datetime
from tqdm import tqdm
from mc_cnn_brunch import Net
from process_functional import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="stereo matching based on trained model and post-processing")

parser.add_argument("-g", "--gpu", type=str, default="0,1,2,3,4,5,6,7", help="gpu id to use, \
                    multiple ids should be separated by commons(e.g. 0,1,2,3)")


def main():

    args = parser.parse_args()

    # GPU preparation
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    detail_time = np.zeros(shape=[7], dtype=np.float32)

    patch_height = 11
    patch_width = 11
    height = 480
    width = 752
    checkpoint = r'./check_points_11_11/model_epoch14.ckpt'
    image_path = r'./test/'

    x = tf.placeholder(tf.float32, [1, height + patch_height - 1, width + patch_width - 1, 1])
    model = Net(x, input_patch_size=patch_height, num_of_conv_layers=patch_height // 2, num_of_conv_feature_maps=64, batch_size=1)
    features = model.features
    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        print('{} : restoring from {}...'.format(datetime.now(), checkpoint))
        saver.restore(sess, checkpoint)

        for i in range(1, 19):
            # i = i % 7
            left_image_path = os.path.join(image_path, 'left_{}.jpg'.format(i))
            right_image_path = os.path.join(image_path, 'right_{}.jpg'.format(i))

            left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

            height, width = left_image.shape[0:2]

            left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
            right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))
            left_image = np.expand_dims(left_image, axis=2)
            right_image = np.expand_dims(right_image, axis=2)

            auged_left_image = np.zeros([1, height + patch_height - 1, width + patch_width - 1, 1], dtype=np.float32)
            auged_right_image = np.zeros([1, height + patch_height - 1, width + patch_width - 1, 1], dtype=np.float32)

            row_start = (patch_height - 1) // 2
            col_start = (patch_width - 1) // 2
            auged_left_image[0, row_start:height + row_start, col_start:width + col_start] = left_image
            auged_right_image[0, row_start:height + row_start, col_start:width + col_start] = right_image

            model.input = tf.placeholder(tf.float32, [1, height + patch_height - 1, width + patch_width - 1, 1])
            # print('{} : features computing...'.format(datetime.now()))
            start = time.time()
            featuresl = sess.run(features, feed_dict={x: auged_left_image})
            featuresr = sess.run(features, feed_dict={x: auged_right_image})
            # print('compute features detail: {}'.format(time.time() - start))
            detail_time[0] += (time.time() - start)

            left_feature = np.squeeze(featuresl, axis=0)
            right_feature = np.squeeze(featuresr, axis=0)  # (height, width, 64)


            _left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
            _right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

            # left_cost_volume, right_cost_volume = disparity_compute_by_gpu(_left_image, _right_image, left_feature, right_feature)
            left_disparity, right_disparity, detail_time = disparity_compute_by_gpu(_left_image, _right_image, left_feature, right_feature, detail_time)

            # left_cost_volume = compute_cost_volume(left_feature, right_feature, 128)
            # left_disparity = WTA1(left_cost_volume)

            cv2.imwrite('./disparity/ld{}.png'.format(i), left_disparity.astype('uint8')*2)
            # cv2.imwrite('./disparity/rd{}.png'.format(i), right_disparity.astype('uint8'))
            # cv2.imshow('disparity', left_disparity.astype('uint8'))
            # cv2.waitKey(0)
            # print('image{} done'.format(i))
        '''
        print('time of computing features: {}s'.format(detail_time[0]/700))
        print('time of computing cost volume: {}s'.format(detail_time[1]/700))
        print('time of "*" cost aggregation: {}s'.format(detail_time[2]/700))
        print('time of SGM: {}s'.format(detail_time[3]/700))
        print('time of WTA & Subpixel refinement: {}s'.format(detail_time[4]/700))
        print('time of LR Check: {}s'.format(detail_time[5]/700))
        print('time of Filtering: {}s'.format(detail_time[6]/700))
       '''
if __name__=="__main__":
    main()


