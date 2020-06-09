import os
import sys
import cv2
import numpy as np
import random


def save_pfm(fname, image, scale=1):
    file = open(fname, 'w')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    np.flipud(image).tofile(file)


image_dir = r'./2003.txt'
left_image_path = []
right_image_path = []
left_disp_path = []
right_disp_path = []

with open(image_dir, 'r') as image_list_file:
    lines = image_list_file.readlines()  # return the list of the image file paths

for line in lines:
    path = line.strip()
    left_image_path.append(os.path.join(path, 'im2.ppm'))
    right_image_path.append(os.path.join(path, 'im6.ppm'))
    left_disp_path.append(os.path.join(path, 'disp2.pgm'))
    right_disp_path.append(os.path.join(path, 'disp6.pgm'))

number_of_patches = 59361
patch_size = 11
pos_high = 0.5
neg_high=6
neg_low=1.5
half_window = patch_size // 2
for i in range(len(left_image_path)):
    left_image = cv2.imread(left_image_path[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    right_image = cv2.imread(right_image_path[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    left_disparity = cv2.imread(left_disp_path[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    right_disparity = cv2.imread(right_disp_path[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)

    height, width = left_image.shape[0:2]

    
    # ----------------------------------------------------------------------------------------------------------------
    left_image = cv2.resize(left_image, (width//2, height//2))
    right_image = cv2.resize(right_image, (width//2, height//2))
    left_disparity = cv2.resize(left_disparity, (width // 2, height // 2))
    right_disparity = cv2.resize(right_disparity, (width // 2, height // 2))
    left_disparity = left_disparity / 2
    right_disparity = right_disparity / 2

    height, width = left_image.shape[0:2]
    # ----------------------------------------------------------------------------------------------------------------
    

    left_image = (left_image - np.mean(left_image, axis=(0, 1))) / np.std(left_image, axis=(0, 1))
    right_image = (right_image - np.mean(right_image, axis=(0, 1))) / np.std(right_image, axis=(0, 1))

    for row in range(half_window, height - half_window, 3):
        for col in range(half_window, width - half_window, 3):
            ld = left_disparity[row, col]
            while ld == float('inf') or ld == 0 or ld + half_window> col or \
                    right_disparity[row, int(col - ld)] == float('inf') or \
                    right_disparity[row, int(col - ld)] == 0 or \
                    np.fabs(right_disparity[row, int(col - ld)] - ld) > 1:
                row = random.randint(half_window, height - half_window - 1)
                col = random.randint(half_window, width - half_window - 1)
                ld = left_disparity[row, col]

            right_col = col - ld

            # left patches
            left_patch = left_image[row-half_window:row+half_window+1, col-half_window:col+half_window+1]

            # postive patches
            pos_col = -1
            while pos_col < half_window or pos_col + half_window >= width:
                pos_col = int(right_col + np.random.uniform(-1 * pos_high - 0.001, pos_high + 0.001))
            right_pos_patch = right_image[row-half_window:row+half_window+1, pos_col-half_window:pos_col+half_window+1]

            # negative patches
            neg_col = -1
            while neg_col < half_window or neg_col + half_window >= width:
                neg_dev = np.random.uniform(neg_low, neg_high)
                if np.random.randint(-1, 1) == -1:
                    neg_dev = -1 * neg_dev
                neg_col = int(right_col + neg_dev)
            right_neg_patch = right_image[row-half_window:row+half_window+1, neg_col-half_window:neg_col+half_window+1]

            save_pfm(r'/home/rjt1/mc_cnn_fst/training_data_set/left/{}.pfm'.format(number_of_patches), left_patch)
            save_pfm(r'/home/rjt1/mc_cnn_fst/training_data_set/right_pos/{}.pfm'.format(number_of_patches), right_pos_patch)
            save_pfm(r'/home/rjt1/mc_cnn_fst/training_data_set/right_neg/{}.pfm'.format(number_of_patches), right_neg_patch)
            number_of_patches += 1
            
print(number_of_patches)






