import cv2
import numpy as np
from numba import cuda, float32
import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="local stereo matching method")

parser.add_argument("-k", "--kernel", type=int, default=5, help="window size")


@cuda.jit
def AD(d_image_l, d_image_r, d_cost_volume, max_dis, kernel):
    rows, cols = d_image_l.shape[0:2]
    x, y = cuda.grid(2)
    if y < rows and x < cols:
        for d in range(max_dis):
            if x - d >= 0:
                sum = 0
                start_row = (y - kernel) if (y - kernel) >= 0 else 0
                end_row = (y + kernel) if (y + kernel) < rows else (rows - 1)
                if x < max_dis + kernel:
                    start_col_l = x
                    end_col_l = x + kernel
                    for row in range(start_row, end_row + 1):
                        for col in range(start_col_l, end_col_l + 1):
                            sum += abs(d_image_l[row, col] - d_image_r[row, col - d])
                            sum += (d_image_l[row, col] - d_image_r[row, col - d]) * (d_image_l[row, col] - d_image_r[row, col - d])
                elif x >= max_dis + kernel and x + kernel < cols - kernel:
                    start_col_l = x - kernel
                    end_col_l = x + kernel
                    for row in range(start_row, end_row + 1):
                        for col in range(start_col_l, end_col_l + 1):
                            sum += abs(d_image_l[row, col] - d_image_r[row, col - d])
                            sum += (d_image_l[row, col] - d_image_r[row, col - d]) * (d_image_l[row, col] - d_image_r[row, col - d])

                else:
                    start_col_l = x - kernel
                    end_col_l = x
                    for row in range(start_row, end_row + 1):
                        for col in range(start_col_l, end_col_l + 1):
                            sum += abs(d_image_l[row, col] - d_image_r[row, col - d])
                            sum += (d_image_l[row, col] - d_image_r[row, col - d]) * (d_image_l[row, col] - d_image_r[row, col - d])

                d_cost_volume[y, x, d] = sum

            else:
                d_cost_volume[y, x, d] = 999999999999

@cuda.jit
def Rank_transform(d_image_l, d_image_r, d_tranl, d_tranr):
    rows, cols = d_image_l.shape[0:2]

    x, y = cuda.grid(2)
    if y < rows and x < cols:
        start_row = y - 3
        end_row = y + 3
        start_col = x - 3
        end_col = x + 3

        center1 = d_image_l[y, x]
        center2 = d_image_r[y, x]
        sum1 = 0
        sum2 = 0
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                is_in_image = 1 if row >= 0 and row < rows and col >= 0 and col < cols else 0
                temp1 = d_image_l[row, col] if is_in_image==1 else 0
                temp2 = d_image_r[row, col] if is_in_image==1 else 0
                sum1 += 1 if center1 >= temp1 else 0
                sum2 += 1 if center2 >= temp2 else 0

        d_tranl[y, x] = sum1
        d_tranr[y, x] = sum2

@cuda.jit
def Census_transform(d_image_l, d_image_r, d_censusl, d_censusr):
    rows, cols = d_image_l.shape[0:2]
    kernel_height = 3
    kernel_width = 4

    x, y = cuda.grid(2)
    if y < rows and x < cols:
        censusl = 0
        censusr = 0
        start_row = y - kernel_height
        start_col = x - kernel_width
        for row in range(start_row, y):
            for col in range(start_col, x):
                censusl <<= 1
                censusr <<= 1
                row_ = y*2 - row
                col_ = x*2 - col
                temp1 = d_image_l[row, col] if row >= 0 and row < rows and col >= 0 and col < cols else 0
                temp2 = d_image_l[row_, col_] if row_ >= 0 and row_ < rows and col_ >= 0 and col_ < cols else 0
                censusl |= 1 if temp1 > temp2 else 0

                temp1 = d_image_r[row, col] if row >= 0 and row < rows and col >= 0 and col < cols else 0
                temp2 = d_image_r[row_, col_] if row_ >= 0 and row_ < rows and col_ >= 0 and col_ < cols else 0
                censusr |= 1 if temp1 > temp2 else 0



        for row in range(start_row, y):
            censusl <<= 1
            censusr <<= 1
            row_ = y*2 - row
            temp1 = d_image_l[row, x] if row >= 0 and row < rows else 0
            temp2 = d_image_l[row_, x] if row_ >= 0 and row_ < rows else 0
            censusl |= 1 if temp1 > temp2 else 0

            temp1 = d_image_r[row, x] if row >= 0 and row < rows else 0
            temp2 = d_image_r[row_, x] if row_ >= 0 and row_ < rows else 0
            censusr |= 1 if temp1 > temp2 else 0

        for col in range(start_col, x):
            censusl <<= 1
            censusr <<= 1
            col_ = x*2 - col
            temp1 = d_image_l[y, col] if col >= 0 and col < cols else 0
            temp2 = d_image_l[y, col_] if col_>= 0 and col_ < cols else 0
            censusl |= 1 if temp1 > temp2 else 0

            temp1 = d_image_r[y, col] if col >= 0 and col < cols else 0
            temp2 = d_image_r[y, col_] if col_ >= 0 and col_ < cols else 0
            censusr |= 1 if temp1 > temp2 else 0

        d_censusl[y, x] = censusl
        d_censusr[y, x] = censusr

@cuda.jit
def Census_Cost(d_censusl, d_censusr, d_cost_volume, max_dis):
    rows, cols = d_censusl.shape[0:2]
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        temp1 = d_censusl[y, x]
        for d in range(max_dis):
            if x - d >= 0:
                sum = 0
                temp2 = d_censusr[y, x - d]
                temp = temp1 ^ temp2
                while temp :
                    sum += (temp%2)
                    temp = temp // 2

                d_cost_volume[y, x, d] = sum

            else:
                d_cost_volume[y, x, d] = 999999999999

@cuda.jit
def WTA__kernel(d_s_volume, d_disparity, max_ids):
    rows, cols = d_disparity.shape[0: 2]
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        min_s = d_s_volume[y, x, 0]
        index = 0
        for i in range(1, max_ids):
            tmp = d_s_volume[y, x, i]
            if min_s > tmp:
                min_s = tmp
                index = i

        d_disparity[y, x] = index



args = parser.parse_args()
kernel = args.kernel

image_dir = r'/home/rjt1/mc_cnn_fst/eval/'
left_image_path = []
right_image_path = []

for i in range(5):
    left_image_path.append(os.path.join(image_dir, 'left_{}.png'.format(i)))
    right_image_path.append(os.path.join(image_dir, 'right_{}.png'.format(i)))

for i in range(len(left_image_path)):
    left_image = cv2.imread(left_image_path[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    right_image = cv2.imread(right_image_path[i], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    height, width = left_image.shape[0:2]
    rows, cols = left_image.shape[0:2]
    max_dis = 200

    disparity = np.zeros(shape=[height, width], dtype=np.uint8)
    d_image_l = cuda.to_device(left_image)
    d_image_r = cuda.to_device(right_image)
    d_disparity = cuda.to_device(disparity)
    d_cost_volume = cuda.device_array(shape=(height, width, 200), dtype=np.float32)
    d_tranl = cuda.device_array(shape=(height, width), dtype=np.uint32)
    d_tranr = cuda.device_array(shape=(height, width), dtype=np.uint32)

    blocksize_x = 16
    blocksize_y = 16
    blocksize = (blocksize_x, blocksize_y)
    gridsize_x = (cols + blocksize_x - 1) // blocksize_x
    gridsize_y = (rows + blocksize_y - 1) // blocksize_y
    gridsize = (gridsize_x, gridsize_y)
    AD[gridsize, blocksize](d_image_l, d_image_r, d_cost_volume, max_dis, kernel)

    # Rank_transform[gridsize, blocksize](d_image_l, d_image_r, d_tranl, d_tranr)
    # AD[gridsize, blocksize](d_tranl, d_tranr, d_cost_volume, max_dis, kernel)
    # Census_transform[gridsize, blocksize](d_image_l, d_image_r, d_tranl, d_tranr)
    # Census_Cost[gridsize, blocksize](d_tranl, d_tranr, d_cost_volume, max_dis)

    WTA__kernel[gridsize, blocksize](d_cost_volume, d_disparity, max_dis)

    disparity = d_disparity.copy_to_host()
    cv2.imwrite(r'/home/rjt1/mc_cnn_fst/result/SAD/ld{}.png'.format(i), disparity)
