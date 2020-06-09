import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from numba import cuda, float32, uint32, uint8, int8
from mc_cnn_brunch import Net



def compute_feature(left_image, right_image, patch_height, patch_width, num_of_feature_maps, checkpoint):
    height, width = left_image.shape[0:2]
    auged_left_image = np.zeros([1, height + patch_height - 1, width + patch_width - 1, 1], dtype=np.float32)
    auged_right_image = np.zeros([1, height + patch_height - 1, width + patch_width - 1, 1], dtype=np.float32)

    row_start = (patch_height - 1) // 2
    col_start = (patch_width - 1) // 2
    auged_left_image[0, row_start:height + row_start, col_start:width + col_start] = left_image
    auged_right_image[0, row_start:height + row_start, col_start:width + col_start] = right_image

    x = tf.placeholder(tf.float32, [1, height + patch_height - 1, width + patch_width - 1, 1])

    model = Net(x, input_patch_size=patch_height, num_of_conv_layers=patch_height // 2, num_of_conv_feature_maps=num_of_feature_maps, batch_size=1)
    saver = tf.train.Saver(max_to_keep=10)

    features = model.features

    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        print('{} : restoring from {}...'.format(datetime.now(), checkpoint))
        saver.restore(sess, checkpoint)

        print('{} : features computing...'.format(datetime.now()))

        start = time.time()
        featuresl = sess.run(features, feed_dict={x: auged_left_image})
        featuresr = sess.run(features, feed_dict={x: auged_right_image})
        print('compute features detail: {}'.format(time.time() - start))

        featuresl = np.squeeze(featuresl, axis=0)
        featuresr = np.squeeze(featuresr, axis=0)  # (height, width, 64)

    return featuresl, featuresr


def compute_cost_volume(featuresl, featuresr, ndisp):
    print('{} : computing cost_volume for the left image...'.format(datetime.now()))
    height, width = featuresl.shape[:2]
    left_cost_volume = np.zeros([ndisp, height, width], dtype=np.float32)

    # compute left image's cost volume
    tem_xl = featuresl
    tem_xr = featuresr
    for d in range(ndisp):
        print('{} : disparity {}...'.format(datetime.now(), d))
        left_cost_volume[d, :, d:] = np.sum(np.multiply(tem_xl, tem_xr), axis=-1)
        tem_xl = tem_xl[:, 1:]
        tem_xr = tem_xr[:, :tem_xr.shape[1]-1]

    # use maximum value to fill in those not calculated
    for d in range(ndisp-1, 0, -1):
        left_cost_volume[d:ndisp, :, :d-1] = 0

    ####   compute right image's cost volume   ###
    #
    #
    #
    #
    ###############################################
    left_cost_volume = -1 * left_cost_volume
    return left_cost_volume


def WTA(left_cost_volume):
    print('{} : left disparity map making...'.format(datetime.now()))
    height, width, ndisp = left_cost_volume.shape
    left_disparity_map = np.ndarray([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
            min_cost = float('inf')
            min_disparity = -1
            for d in range(ndisp):
                if left_cost_volume[h, w, d] < min_cost:
                    min_cost = left_cost_volume[h, w, d]
                    min_disparity = d
            assert min_disparity >= 0
            left_disparity_map[h, w] = min_disparity

    print('{}: disparity map done...'.format(datetime.now()))
    return left_disparity_map


def WTA1(left_cost_volume):
    print('{} : left disparity map making...'.format(datetime.now()))
    ndisp, height, width = left_cost_volume.shape
    left_disparity_map = np.ndarray([height, width], dtype=np.float32)

    for h in range(height):
        for w in range(width):
            min_cost = float('inf')
            min_disparity = -1
            for d in range(ndisp):
                if left_cost_volume[d, h, w] < min_cost:
                    min_cost = left_cost_volume[d, h, w]
                    min_disparity = d
            assert min_disparity >= 0
            left_disparity_map[h, w] = min_disparity

    print('{}: disparity map done...'.format(datetime.now()))
    return left_disparity_map


##################################################################################################################
#                            ---  CUDA Python: Numba implementation  ---                                         #
##################################################################################################################

@cuda.jit
def compute_cost_volume_kernel(d_featuresl, d_featuresr, d_cost_volumel, d_cost_volumer):
    rows, cols = d_featuresl.shape[0:2]
    x, y = cuda.grid(2)
    if y < rows and x < cols:
        for d in range(128):
            if x - d >= 0:
                temp = 0.0
                for i in range(64):
                    temp += d_featuresl[y, x, i] * d_featuresr[y, x - d, i]
                d_cost_volumel[y, x, d] = -temp
                d_cost_volumer[y, x - d, d] = -temp


@cuda.jit
def sgm_penelty_kernel(d_imagel, d_imager, d_sgm_pl, d_sgm_pr, P1, P2, threshold, lamda):
    rows, cols = d_imagel.shape[0:2]
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        templ = d_imagel[y, x]
        tempr = d_imager[y, x]
        p1 = P1 / lamda
        p2 = P2 / lamda

        if y - 1 >= 0:
            diffl = d_imagel[y-1, x] - templ
            diffr = d_imager[y-1, x] - tempr
            diffl = -diffl if diffl < 0 else diffl
            diffr = -diffr if diffr < 0 else diffr
            d_sgm_pl[y, x, 2] = p1 if (diffl > threshold) else P1
            d_sgm_pl[y, x, 3] = p2 if (diffl > threshold) else P2
            d_sgm_pr[y, x, 2] = p1 if (diffr > threshold) else P1
            d_sgm_pr[y, x, 3] = p2 if (diffr > threshold) else P2
        else:
            d_sgm_pl[y, x, 2] = P1
            d_sgm_pl[y, x, 3] = P2
            d_sgm_pr[y, x, 2] = P1
            d_sgm_pr[y, x, 3] = P2

        if y + 1 < rows:
            diffl = d_imagel[y+1, x] - templ
            diffr = d_imager[y+1, x] - tempr
            diffl = -diffl if diffl < 0 else diffl
            diffr = -diffr if diffr < 0 else diffr
            d_sgm_pl[y, x, 2] = p1 if (diffl > threshold) else P1
            d_sgm_pl[y, x, 3] = p2 if (diffl > threshold) else P2
            d_sgm_pr[y, x, 2] = p1 if (diffr > threshold) else P1
            d_sgm_pr[y, x, 3] = p2 if (diffr > threshold) else P2
        else:
            d_sgm_pl[y, x, 2] = P1
            d_sgm_pl[y, x, 3] = P2
            d_sgm_pr[y, x, 2] = P1
            d_sgm_pr[y, x, 3] = P2

        if x - 1 >= 0:
            diffl = d_imagel[y, x - 1] - templ
            diffr = d_imager[y, x - 1] - tempr
            diffl = -diffl if diffl < 0 else diffl
            diffr = -diffr if diffr < 0 else diffr
            d_sgm_pl[y, x, 4] = p1 if (diffl > threshold) else P1
            d_sgm_pl[y, x, 5] = p2 if (diffl > threshold) else P2
            d_sgm_pr[y, x, 4] = p1 if (diffr > threshold) else P1
            d_sgm_pr[y, x, 5] = p2 if (diffr > threshold) else P2
        else:
            d_sgm_pl[y, x, 4] = P1
            d_sgm_pl[y, x, 5] = P2
            d_sgm_pr[y, x, 4] = P1
            d_sgm_pr[y, x, 5] = P2

        if x + 1 < cols:
            diffl = d_imagel[y, x + 1] - templ
            diffr = d_imager[y, x + 1] - tempr
            diffl = -diffl if diffl < 0 else diffl
            diffr = -diffr if diffr < 0 else diffr
            d_sgm_pl[y, x, 6] = p1 if (diffl > threshold) else P1
            d_sgm_pl[y, x, 7] = p2 if (diffl > threshold) else P2
            d_sgm_pr[y, x, 6] = p1 if (diffr > threshold) else P1
            d_sgm_pr[y, x, 7] = p2 if (diffr > threshold) else P2
        else:
            d_sgm_pl[y, x, 6] = P1
            d_sgm_pl[y, x, 7] = P2
            d_sgm_pr[y, x, 6] = P1
            d_sgm_pr[y, x, 7] = P2

        if y + 1 < rows and x - 1 >= 0:
            diffl = d_imagel[y + 1, x - 1] - templ
            diffr = d_imager[y + 1, x - 1] - tempr
            diffl = -diffl if diffl < 0 else diffl
            diffr = -diffr if diffr < 0 else diffr
            d_sgm_pl[y, x, 8] = p1 if (diffl > threshold) else P1
            d_sgm_pl[y, x, 9] = p2 if (diffl > threshold) else P2
            d_sgm_pr[y, x, 8] = p1 if (diffr > threshold) else P1
            d_sgm_pr[y, x, 9] = p2 if (diffr > threshold) else P2
        else:
            d_sgm_pl[y, x, 8] = P1
            d_sgm_pl[y, x, 9] = P2
            d_sgm_pr[y, x, 8] = P1
            d_sgm_pr[y, x, 9] = P2

        if y + 1 < rows and x + 1 < cols:
            diffl = d_imagel[y + 1, x + 1] - templ
            diffr = d_imager[y + 1, x + 1] - tempr
            diffl = -diffl if diffl < 0 else diffl
            diffr = -diffr if diffr < 0 else diffr
            d_sgm_pl[y, x, 10] = p1 if (diffl > threshold) else P1
            d_sgm_pl[y, x, 11] = p2 if (diffl > threshold) else P2
            d_sgm_pr[y, x, 10] = p1 if (diffr > threshold) else P1
            d_sgm_pr[y, x, 11] = p2 if (diffr > threshold) else P2
        else:
            d_sgm_pl[y, x, 10] = P1
            d_sgm_pl[y, x, 11] = P2
            d_sgm_pr[y, x, 10] = P1
            d_sgm_pr[y, x, 11] = P2

        if y - 1 >= 0 and x + 1 < cols:
            diffl = d_imagel[y - 1, x + 1] - templ
            diffr = d_imager[y - 1, x + 1] - tempr
            diffl = -diffl if diffl < 0 else diffl
            diffr = -diffr if diffr < 0 else diffr
            d_sgm_pl[y, x, 12] = p1 if (diffl > threshold) else P1
            d_sgm_pl[y, x, 13] = p2 if (diffl > threshold) else P2
            d_sgm_pr[y, x, 12] = p1 if (diffr > threshold) else P1
            d_sgm_pr[y, x, 13] = p2 if (diffr > threshold) else P2
        else:
            d_sgm_pl[y, x, 12] = P1
            d_sgm_pl[y, x, 13] = P2
            d_sgm_pr[y, x, 12] = P1
            d_sgm_pr[y, x, 13] = P2

        if y - 1 >= 0 and x - 1 >= 0:
            diffl = d_imagel[y - 1, x - 1] - templ
            diffr = d_imager[y - 1, x - 1] - tempr
            diffl = -diffl if diffl < 0 else diffl
            diffr = -diffr if diffr < 0 else diffr
            d_sgm_pl[y, x, 14] = p1 if (diffl > threshold) else P1
            d_sgm_pl[y, x, 15] = p2 if (diffl > threshold) else P2
            d_sgm_pr[y, x, 14] = p1 if (diffr > threshold) else P1
            d_sgm_pr[y, x, 15] = p2 if (diffr > threshold) else P2
        else:
            d_sgm_pl[y, x, 14] = P1
            d_sgm_pl[y, x, 15] = P2
            d_sgm_pr[y, x, 14] = P1
            d_sgm_pr[y, x, 15] = P2


@cuda.jit(device=True)
def SGM_Interation(row_index, col_index, disp, P1, P2, \
                   old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2, is_copy, is_first, is_cal_min, lr, \
                   d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer):
    c1 = 0.0
    c2 = 0.0
    c3 = 0.0
    c4 = 0.0
    if (is_first is True) or (is_copy is True):
        if lr == 0:
            c1 = d_cost_volumel_after_aggr[row_index, col_index, disp]
            c2 = d_cost_volumel_after_aggr[row_index, col_index, disp + 1]
            c3 = d_cost_volumel_after_aggr[row_index, col_index, disp + 2]
            c4 = d_cost_volumel_after_aggr[row_index, col_index, disp + 3]
        else:
            c1 = d_cost_volumer_after_aggr[row_index, col_index, disp]
            c2 = d_cost_volumer_after_aggr[row_index, col_index, disp + 1]
            c3 = d_cost_volumer_after_aggr[row_index, col_index, disp + 2]
            c4 = d_cost_volumer_after_aggr[row_index, col_index, disp + 3]

    else:
        if lr == 0:
            c1 = d_cost_volumel_after_aggr[row_index, col_index, disp]
            c2 = d_cost_volumel_after_aggr[row_index, col_index, disp + 1]
            c3 = d_cost_volumel_after_aggr[row_index, col_index, disp + 2]
            c4 = d_cost_volumel_after_aggr[row_index, col_index, disp + 3]
        else:
            c1 = d_cost_volumer_after_aggr[row_index, col_index, disp]
            c2 = d_cost_volumer_after_aggr[row_index, col_index, disp + 1]
            c3 = d_cost_volumer_after_aggr[row_index, col_index, disp + 2]
            c4 = d_cost_volumer_after_aggr[row_index, col_index, disp + 3]

        pre = cuda.shfl_up_sync(0xffffffff, old_values4, 1)
        next = cuda.shfl_down_sync(0xffffffff, old_values1, 1)

        if disp // 4 == 0:
            pre = old_values1
        if disp // 4 == 31:
            next = old_values4

        m1 = min(pre + P1, old_values1)
        m2 = min(old_values2 + P1, min_cost_P2)
        c1 += (min(m1, m2) - min_cost)

        m1 = min(old_values1 + P1, old_values2)
        m2 = min(old_values3 + P1, min_cost_P2)
        c2 += (min(m1, m2) - min_cost)

        m1 = min(old_values2 + P1, old_values3)
        m2 = min(old_values4 + P1, min_cost_P2)
        c3 += (min(m1, m2) - min_cost)

        m1 = min(old_values3 + P1, old_values4)
        m2 = min(next + P1, min_cost_P2)
        c4 += (min(m1, m2) - min_cost)

    if lr == 0:
        d_s_volumel[row_index, col_index, disp] += c1
        d_s_volumel[row_index, col_index, disp + 1] += c2
        d_s_volumel[row_index, col_index, disp + 2] += c3
        d_s_volumel[row_index, col_index, disp + 3] += c4
    else:
        d_s_volumer[row_index, col_index, disp] += c1
        d_s_volumer[row_index, col_index, disp + 1] += c2
        d_s_volumer[row_index, col_index, disp + 2] += c3
        d_s_volumer[row_index, col_index, disp + 3] += c4

    if is_cal_min is True:
        m1 = min(c1, c2)
        m2 = min(c3, c4)
        min_cost = min(m1, m2)
        min_cost = min(min_cost, cuda.shfl_xor_sync(0xffffffff, min_cost, 1))
        min_cost = min(min_cost, cuda.shfl_xor_sync(0xffffffff, min_cost, 2))
        min_cost = min(min_cost, cuda.shfl_xor_sync(0xffffffff, min_cost, 4))
        min_cost = min(min_cost, cuda.shfl_xor_sync(0xffffffff, min_cost, 8))
        min_cost = min(min_cost, cuda.shfl_xor_sync(0xffffffff, min_cost, 16))
        min_cost_P2 = min_cost + P2

    return c1, c2, c3, c4, min_cost, min_cost_P2


@cuda.jit
def SGM_UpToDown_kernel(d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, \
                        d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr):
    rows, cols = d_cost_volumer_after_aggr.shape[0:2]
    col_index = cuda.blockDim.x * cuda.blockIdx.x // 32 + cuda.threadIdx.x // 32
    if col_index < cols:
        lane = cuda.threadIdx.x % 32
        disp = lane*4

        for k in range(2):
            row_index = 0
            max_iter = rows - 1

            old_values1 = 1.0
            old_values2 = 1.0
            old_values3 = 1.0
            old_values4 = 1.0
            min_cost = 1.0
            min_cost_P2 = 1.0

            if row_index - 1 < 0:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index - 1, col_index, 2] if k == 0 else d_sgm_pr[row_index - 1, col_index, 2]
            P2 = d_sgm_pl[row_index, col_index, 3] if k == 0 else d_sgm_pr[row_index, col_index, 3]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, True, True, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)
            for i in range(1, max_iter - 1):
                row_index += 1
                P1 = d_sgm_pl[row_index - 1, col_index, 2] if k == 0 else d_sgm_pr[row_index - 1, col_index, 2]
                P2 = d_sgm_pl[row_index, col_index, 3] if k == 0 else d_sgm_pr[row_index, col_index, 3]
                old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                    SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3,
                                   old_values4, min_cost, min_cost_P2, False, False, True, k, d_cost_volumel_after_aggr,
                                   d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            row_index += 1
            P1 = d_sgm_pl[row_index - 1, col_index, 2] if k == 0 else d_sgm_pr[row_index - 1, col_index, 2]
            P2 = d_sgm_pl[row_index, col_index, 3] if k == 0 else d_sgm_pr[row_index, col_index, 3]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, False, False, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)


@cuda.jit
def SGM_DownToUp_kernel(d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, \
                        d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr):
    rows, cols = d_cost_volumer_after_aggr.shape[0:2]
    col_index = cuda.blockDim.x * cuda.blockIdx.x // 32 + cuda.threadIdx.x // 32
    if col_index < cols:
        lane = cuda.threadIdx.x % 32
        disp = lane*4

        for k in range(2):
            row_index = rows - 1
            max_iter = rows - 1

            old_values1 = 1.0
            old_values2 = 1.0
            old_values3 = 1.0
            old_values4 = 1.0
            min_cost = 1.0
            min_cost_P2 = 1.0

            if row_index + 1 >= rows:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index + 1, col_index, 0] if k == 0 else d_sgm_pr[row_index + 1, col_index, 0]
            P2 = d_sgm_pl[row_index, col_index, 1] if k == 0 else d_sgm_pr[row_index, col_index, 1]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, True, True, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)
            for i in range(1, max_iter - 1):
                row_index -= 1
                P1 = d_sgm_pl[row_index + 1, col_index, 0] if k == 0 else d_sgm_pr[row_index + 1, col_index, 0]
                P2 = d_sgm_pl[row_index, col_index, 1] if k == 0 else d_sgm_pr[row_index, col_index, 1]
                old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                    SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3,
                                   old_values4, min_cost, min_cost_P2, False, False, True, k, d_cost_volumel_after_aggr,
                                   d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            row_index -= 1
            P1 = d_sgm_pl[row_index + 1, col_index, 0] if k == 0 else d_sgm_pr[row_index + 1, col_index, 0]
            P2 = d_sgm_pl[row_index, col_index, 1] if k == 0 else d_sgm_pr[row_index, col_index, 1]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, False, False, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)


@cuda.jit
def SGM_LeftToRight_kernel(d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, \
                           d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr):
    rows, cols = d_cost_volumer_after_aggr.shape[0:2]
    row_index = cuda.blockDim.x * cuda.blockIdx.x // 32 + cuda.threadIdx.x // 32
    if row_index < rows:
        lane = cuda.threadIdx.x % 32
        disp = lane*4

        for k in range(2):
            col_index = 0
            max_iter = cols - 1

            old_values1 = 1.0
            old_values2 = 1.0
            old_values3 = 1.0
            old_values4 = 1.0
            min_cost = 1.0
            min_cost_P2 = 1.0

            if col_index - 1 < 0:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index, col_index - 1, 6] if k == 0 else d_sgm_pr[row_index, col_index - 1, 6]
            P2 = d_sgm_pl[row_index, col_index, 7] if k == 0 else d_sgm_pr[row_index, col_index, 7]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, True, True, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            for i in range(1, max_iter - 1):
                col_index += 1
                P1 = d_sgm_pl[row_index, col_index - 1, 6] if k == 0 else d_sgm_pr[row_index, col_index - 1, 6]
                P2 = d_sgm_pl[row_index, col_index, 7] if k == 0 else d_sgm_pr[row_index, col_index, 7]
                old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                    SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3,
                                   old_values4, min_cost, min_cost_P2, False, False, True, k, d_cost_volumel_after_aggr,
                                   d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            col_index += 1
            P1 = d_sgm_pl[row_index, col_index - 1, 6] if k == 0 else d_sgm_pr[row_index, col_index - 1, 6]
            P2 = d_sgm_pl[row_index, col_index, 7] if k == 0 else d_sgm_pr[row_index, col_index, 7]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, False, False, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)


@cuda.jit
def SGM_RightToLeft_kernel(d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, \
                           d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr):
    rows, cols = d_cost_volumer_after_aggr.shape[0:2]
    row_index = cuda.blockDim.x * cuda.blockIdx.x // 32 + cuda.threadIdx.x // 32
    if row_index < rows:
        lane = cuda.threadIdx.x % 32
        disp = lane*4

        for k in range(2):
            col_index = cols - 1
            max_iter = cols - 1

            old_values1 = 1.0
            old_values2 = 1.0
            old_values3 = 1.0
            old_values4 = 1.0
            min_cost = 1.0
            min_cost_P2 = 1.0

            if col_index + 1 >= cols:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index, col_index + 1, 4] if k == 0 else d_sgm_pr[row_index, col_index + 1, 4]
            P2 = d_sgm_pl[row_index, col_index, 5] if k == 0 else d_sgm_pr[row_index, col_index, 5]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, True, True, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            for i in range(1, max_iter - 1):
                col_index -= 1
                P1 = d_sgm_pl[row_index, col_index + 1, 4] if k == 0 else d_sgm_pr[row_index, col_index + 1, 4]
                P2 = d_sgm_pl[row_index, col_index, 5] if k == 0 else d_sgm_pr[row_index, col_index, 5]
                old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                    SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3,
                                   old_values4, min_cost, min_cost_P2, False, False, True, k, d_cost_volumel_after_aggr,
                                   d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            col_index -= 1
            P1 = d_sgm_pl[row_index, col_index + 1, 4] if k == 0 else d_sgm_pr[row_index, col_index + 1, 4]
            P2 = d_sgm_pl[row_index, col_index, 5] if k == 0 else d_sgm_pr[row_index, col_index, 5]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, False, False, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)


@cuda.jit
def SGM_UpToDownAndLeftToRight_kernel(d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, \
                                      d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr):
    rows, cols = d_cost_volumer_after_aggr.shape[0:2]
    col_index = cuda.blockDim.x * cuda.blockIdx.x // 32 + cuda.threadIdx.x // 32
    if col_index < cols:
        lane = cuda.threadIdx.x % 32
        disp = lane*4

        for k in range(2):
            row_index = 0
            max_iter = rows - 1

            old_values1 = 1.0
            old_values2 = 1.0
            old_values3 = 1.0
            old_values4 = 1.0
            min_cost = 1.0
            min_cost_P2 = 1.0

            if col_index - 1 < 0 or row_index - 1 < 0:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index - 1, col_index - 1, 10] if k == 0 else d_sgm_pr[row_index - 1, col_index - 1, 10]
            P2 = d_sgm_pl[row_index, col_index, 11] if k == 0 else d_sgm_pr[row_index, col_index, 11]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, True, True, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            for i in range(1, max_iter - 1):
                row_index += 1
                col_index += 1
                is_copy = False
                if col_index >= cols:
                    col_index = 0
                    is_copy = True

                if col_index - 1 < 0 or row_index - 1 < 0:
                    P1 = 0
                else:
                    P1 = d_sgm_pl[row_index - 1, col_index - 1, 10] if k == 0 else d_sgm_pr[row_index - 1, col_index - 1, 10]
                P2 = d_sgm_pl[row_index, col_index, 11] if k == 0 else d_sgm_pr[row_index, col_index, 11]
                old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                    SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3,
                                   old_values4, min_cost, min_cost_P2, is_copy, False, True, k,
                                   d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            row_index += 1
            col_index += 1
            is_copy = False
            if col_index >= cols:
                col_index = 0
                is_copy = True

            if col_index - 1 < 0 or row_index - 1 < 0:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index - 1, col_index - 1, 10] if k == 0 else d_sgm_pr[row_index - 1, col_index - 1, 10]
            P2 = d_sgm_pl[row_index, col_index, 11] if k == 0 else d_sgm_pr[row_index, col_index, 11]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, is_copy, False, False, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)


@cuda.jit
def SGM_DownToUpAndLeftToRight_kernel(d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, \
                                      d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr):
    rows, cols = d_cost_volumer_after_aggr.shape[0:2]
    col_index = cuda.blockDim.x * cuda.blockIdx.x // 32 + cuda.threadIdx.x // 32
    if col_index < cols:
        lane = cuda.threadIdx.x % 32
        disp = lane*4

        for k in range(2):
            row_index = rows - 1
            max_iter = rows - 1

            old_values1 = 1.0
            old_values2 = 1.0
            old_values3 = 1.0
            old_values4 = 1.0
            min_cost = 1.0
            min_cost_P2 = 1.0

            if col_index - 1 < 0 or row_index + 1 >= rows:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index + 1, col_index - 1, 12] if k == 0 else d_sgm_pr[row_index + 1, col_index - 1, 12]
            P2 = d_sgm_pl[row_index, col_index, 13] if k == 0 else d_sgm_pr[row_index, col_index, 13]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, True, True, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            for i in range(1, max_iter - 1):
                row_index -= 1
                col_index += 1
                is_copy = False
                if col_index >= cols:
                    col_index = 0
                    is_copy = True

                if col_index - 1 < 0 or row_index + 1 >= rows:
                    P1 = 0
                else:
                    P1 = d_sgm_pl[row_index + 1, col_index - 1, 12] if k == 0 else d_sgm_pr[row_index + 1, col_index - 1, 12]
                P2 = d_sgm_pl[row_index, col_index, 13] if k == 0 else d_sgm_pr[row_index, col_index, 13]
                old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                    SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3,
                                   old_values4, min_cost, min_cost_P2, is_copy, False, True, k,
                                   d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            row_index -= 1
            col_index += 1
            is_copy = False
            if col_index >= cols:
                col_index = 0
                is_copy = True

            if col_index - 1 < 0 or row_index + 1 >= rows:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index + 1, col_index - 1, 12] if k == 0 else d_sgm_pr[row_index + 1, col_index - 1, 12]
            P2 = d_sgm_pl[row_index, col_index, 13] if k == 0 else d_sgm_pr[row_index, col_index, 13]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, is_copy, False, False, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)


@cuda.jit
def SGM_UpToDownAndRightToLeft_kernel(d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, \
                                      d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr):
    rows, cols = d_cost_volumer_after_aggr.shape[0:2]
    col_index = cuda.blockDim.x * cuda.blockIdx.x // 32 + cuda.threadIdx.x // 32
    if col_index < cols:
        lane = cuda.threadIdx.x % 32
        disp = lane*4

        for k in range(2):
            row_index = 0
            max_iter = rows - 1

            old_values1 = 1.0
            old_values2 = 1.0
            old_values3 = 1.0
            old_values4 = 1.0
            min_cost = 1.0
            min_cost_P2 = 1.0

            if col_index + 1 >= cols or row_index - 1 < 0:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index - 1, col_index + 1, 8] if k == 0 else d_sgm_pr[row_index - 1, col_index + 1, 8]
            P2 = d_sgm_pl[row_index, col_index, 9] if k == 0 else d_sgm_pr[row_index, col_index, 9]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, True, True, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            for i in range(1, max_iter - 1):
                row_index += 1
                col_index -= 1
                is_copy = False
                if col_index < 0:
                    col_index = cols - 1
                    is_copy = True

                if col_index + 1 >= cols or row_index - 1 < 0:
                    P1 = 0
                else:
                    P1 = d_sgm_pl[row_index - 1, col_index + 1, 8] if k == 0 else d_sgm_pr[row_index - 1, col_index + 1, 8]
                P2 = d_sgm_pl[row_index, col_index, 9] if k == 0 else d_sgm_pr[row_index, col_index, 9]
                old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                    SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3,
                                   old_values4, min_cost, min_cost_P2, is_copy, False, True, k,
                                   d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            row_index += 1
            col_index -= 1
            is_copy = False
            if col_index < 0:
                col_index = cols - 1
                is_copy = True

            if col_index + 1 >= cols or row_index - 1 < 0:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index - 1, col_index + 1, 8] if k == 0 else d_sgm_pr[row_index - 1, col_index + 1, 8]
            P2 = d_sgm_pl[row_index, col_index, 9] if k == 0 else d_sgm_pr[row_index, col_index, 9]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, is_copy, False, False, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)


@cuda.jit
def SGM_DownToUpAndRightToLeft_kernel(d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, \
                                      d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr):
    rows, cols = d_cost_volumer_after_aggr.shape[0:2]
    col_index = cuda.blockDim.x * cuda.blockIdx.x // 32 + cuda.threadIdx.x // 32
    if col_index < cols:
        lane = cuda.threadIdx.x % 32
        disp = lane*4

        for k in range(2):
            row_index = rows - 1
            max_iter = rows - 1

            old_values1 = 1.0
            old_values2 = 1.0
            old_values3 = 1.0
            old_values4 = 1.0
            min_cost = 1.0
            min_cost_P2 = 1.0

            if col_index + 1 >= cols or row_index + 1 >= rows:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index + 1, col_index + 1, 14] if k == 0 else d_sgm_pr[row_index + 1, col_index + 1, 14]
            P2 = d_sgm_pl[row_index, col_index, 15] if k == 0 else d_sgm_pr[row_index, col_index, 15]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, False, True, True, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            for i in range(1, max_iter - 1):
                row_index -= 1
                col_index -= 1
                is_copy = False
                if col_index < 0:
                    col_index = cols - 1
                    is_copy = True

                if col_index + 1 >= cols or row_index + 1 >= rows:
                    P1 = 0
                else:
                    P1 = d_sgm_pl[row_index + 1, col_index + 1, 14] if k == 0 else d_sgm_pr[row_index + 1, col_index + 1, 14]
                P2 = d_sgm_pl[row_index, col_index, 15] if k == 0 else d_sgm_pr[row_index, col_index, 15]
                old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                    SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3,
                                   old_values4, min_cost, min_cost_P2, is_copy, False, True, k,
                                   d_cost_volumel_after_aggr, d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)

            row_index -= 1
            col_index -= 1
            is_copy = False
            if col_index < 0:
                col_index = cols - 1
                is_copy = True

            if col_index + 1 >= cols or row_index + 1 >= rows:
                P1 = 0
            else:
                P1 = d_sgm_pl[row_index + 1, col_index + 1, 14] if k == 0 else d_sgm_pr[row_index + 1, col_index + 1, 14]
            P2 = d_sgm_pl[row_index, col_index, 15] if k == 0 else d_sgm_pr[row_index, col_index, 15]
            old_values1, old_values2, old_values3, old_values4, min_cost, min_cost_P2 = \
                SGM_Interation(row_index, col_index, disp, P1, P2, old_values1, old_values2, old_values3, old_values4,
                               min_cost, min_cost_P2, is_copy, False, False, k, d_cost_volumel_after_aggr,
                               d_cost_volumer_after_aggr, d_s_volumel, d_s_volumer)


@cuda.jit
def WTA_and_SupixelRefinement_kernel(d_s_volumel, d_s_volumer, d_disparityl, d_disparityr):
    rows, cols = d_disparityl.shape[0: 2]
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        min_s = d_s_volumel[y, x, 0]
        index = 0
        for i in range(1, 128):
            tmp = d_s_volumel[y, x, i]
            if min_s > tmp:
                min_s = tmp
                index = i
        min_index = index
        '''
        if index > 0 and index < 127:
            c = d_s_volumel[y, x, index]
            _c = d_s_volumel[y, x, index - 1]
            c_ = d_s_volumel[y, x, index + 1]
            min_index = index - (c_ - _c)/(2*(_c + c_ - 2 * c))
        '''
        d_disparityl[y, x] = min_index

        min_s = d_s_volumer[y, x, 0]
        index = 0
        for i in range(1, 128):
            tmp = d_s_volumer[y, x, i]
            if min_s > tmp:
                min_s = tmp
                index = i
        min_index = index
        '''
        if index > 0 and index < 127:
            c = d_s_volumer[y, x, index]
            _c = d_s_volumer[y, x, index - 1]
            c_ = d_s_volumer[y, x, index + 1]
            min_index = index - (c_ - _c) / (2 * (_c + c_ - 2 * c))
        '''
        d_disparityr[y, x] = min_index


@cuda.jit
def Median_Filter_kernel(d_disparityl, d_disparityr, d_disparityl_a, d_disparityr_a):
    rows, cols = d_disparityl.shape[0:2]
    x, y = cuda.grid(2)
    idx = x + 2
    idy = y + 2

    windows5x5 = cuda.local.array(shape=(25), dtype=float32)
    if (idy + 2) < rows and (idx + 2) < cols:
        for i in range(-2, 3):
            for j in range(-2, 3):
                windows5x5[(i+2)*5+j+2] = d_disparityl[idy+i, idx+j]

        current_min = 0.0
        for i in range(13):
            current_min = windows5x5[i]
            current_min_index = i
            for j in range(i+1, 25):
                tmp = windows5x5[j]
                if current_min > tmp:
                    current_min = tmp
                    current_min_index = j
            windows5x5[current_min_index] = windows5x5[i]
        d_disparityl_a[idy, idx] = current_min

        for i in range(-2, 3):
            for j in range(-2, 3):
                windows5x5[(i+2)*5+j+2] = d_disparityr[idy+i, idx+j]

        current_min = 0.0
        for i in range(13):
            current_min = windows5x5[i]
            current_min_index = i
            for j in range(i+1, 25):
                tmp = windows5x5[j]
                if current_min > tmp:
                    current_min = tmp
                    current_min_index = j
            windows5x5[current_min_index] = windows5x5[i]
        d_disparityr_a[idy, idx] = current_min


@cuda.jit
def Bilateral_Filter_kernel(d_imagel, d_imager, d_disparityl, d_disparityr, d_disparityl_a, d_disparityr_a):
    rows, cols = d_imagel.shape[0:2]

    # thread id which is correspond to pixel index
    idx = cuda.threadIdx.x
    idy = cuda.threadIdx.y
    x, y = cuda.grid(2)

    # allocate shared memory
    filter_window = cuda.shared.array(shape=(24, 24), dtype=float32)
    image_patch = cuda.shared.array(shape=(24, 24), dtype=uint8)
    weights = cuda.shared.array(shape=(10), dtype=float32)

    # thread id to carray the data from global memory to shared memory
    id = cuda.blockDim.x * cuda.threadIdx.y + cuda.threadIdx.x
    row = id // 24
    col = id % 24

    im_row = row + cuda.blockIdx.y * cuda.blockDim.y - 4
    im_col = col + cuda.blockIdx.x * cuda.blockDim.x - 4
    is_in_image = (im_row >= 0) & (im_row < rows) & (im_col >= 0) & (im_col < cols)

    # deal with the remains data
    _row = (id + 256) // 24
    _col = (id + 256) % 24
    _im_row = _row + cuda.blockIdx.y * cuda.blockDim.y - 4
    _im_col = _col + cuda.blockIdx.x * cuda.blockDim.x - 4
    _is_in_image = (_im_row >= 0) & (_im_row < rows) & (_im_col >= 0) & (_im_col < cols)

    row_ = (id + 512) // 24
    col_ = (id + 512) % 24
    im_row_ = row_ + cuda.blockIdx.y * cuda.blockDim.y - 4
    im_col_ = col_ + cuda.blockIdx.x * cuda.blockDim.x - 4
    is_in_image_ = (im_row_ >= 0) & (im_row_ < rows) & (im_col_ >= 0) & (im_col_ < cols)

    # initialize the weights
    if id == 0:
        weights[0] = 0.167747
        weights[1] = 0.165145
        weights[2] = 0.157581
        weights[3] = 0.145735
        weights[4] = 0.130632
        weights[5] = 0.113490
        weights[6] = 0.095563
        weights[7] = 0.077991
        weights[8] = 0.061692
        weights[9] = 0.047297

    cuda.syncthreads()

    for k in range(2):
        if k == 0:
            image_patch[row, col] = d_imagel[im_row, im_col] if is_in_image is True else 0
            filter_window[row, col] = d_disparityl[im_row, im_col] if is_in_image is True else 0
            image_patch[_row, _col] = d_imagel[_im_row, _im_col] if _is_in_image is True else 0
            filter_window[_row, _col] = d_disparityl[_im_row, _im_col] if _is_in_image is True else 0
            if id + 512 < 596:
                image_patch[row_, col_] = d_imagel[im_row_, im_col_] if is_in_image_ is True else 0
                filter_window[row_, col_] = d_disparityl[im_row_, im_col_] if is_in_image_ is True else 0
        else:
            image_patch[row, col] = d_imager[im_row, im_col] if is_in_image is True else 0
            filter_window[row, col] = d_disparityr[im_row, im_col] if is_in_image is True else 0
            image_patch[_row, _col] = d_imager[_im_row, _im_col] if _is_in_image is True else 0
            filter_window[_row, _col] = d_disparityr[_im_row, _im_col] if _is_in_image is True else 0
            if id + 512 < 596:
                image_patch[row_, col_] = d_imager[im_row_, im_col_] if is_in_image_ is True else 0
                filter_window[row_, col_] = d_disparityr[im_row_, im_col_] if is_in_image_ is True else 0

        cuda.syncthreads()

        weights_sum = 0.0
        disparity_sum = 0.0
        current_pixel_intensity = image_patch[idy+4, idx+4]
        current_pixel_disparity = filter_window[idy+4, idx+4]
        for i in range(0, 9):
            for j in range(0, 9):
                tmp_intensity = image_patch[idy+i, idx+j]
                minus = current_pixel_intensity - tmp_intensity
                absolute_minus = minus if minus >= 0 else -minus
                if absolute_minus < 5:
                    weight = weights[absolute_minus]
                    weights_sum += weight
                    tmp_disparity = filter_window[idy+i, idx+j]
                    disparity_sum += (weight*tmp_disparity)

        if (y < rows) and (x < cols):
            if k == 0:
                d_disparityl_a[y, x] = disparity_sum/weights_sum
            else:
                d_disparityr_a[y, x] = disparity_sum/weights_sum

        cuda.syncthreads()


@cuda.jit
def is_error_match_kernel(d_disparityl, d_disparityr, d_lrc_l, d_lrc_r):
    rows, cols = d_disparityl.shape[0:2]
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        ld = d_disparityl[y, x]
        rd = x - ld
        if rd >= 0:
            rd = d_disparityr[y, uint8(rd)]
            minus = ld - rd
            if minus > 1 or minus < -1:
                d_lrc_l[y, x] = 1
            else:
                d_lrc_l[y, x] = 0

        rd = d_disparityr[y, x]
        ld = x + rd
        if ld < cols:
            ld = d_disparityl[y, uint8(ld)]
            minus = rd - ld
            if minus > 1 or minus < -1:
                d_lrc_r[y, x] = 1
            else:
                d_lrc_r[y, x] = 0


@cuda.jit
def LRC_kernel(d_disparityl, d_disparityr, d_lrc_l, d_lrc_r, d_disparityl_a, d_disparityr_a):
    rows, cols = d_disparityl.shape[0:2]
    x, y = cuda.grid(2)
    if x < cols and y < rows:
        if d_lrc_l[y, x] == 1:
            number = 0
            sum_d = 0

            idy = y
            idx = x
            while idy >= 0 and d_lrc_l[idy, idx] == 1:
                idy -= 1
            if idy >= 0:
                number += 1
                sum_d += d_disparityl[idy, idx]

            idy = y
            idx = x
            while idy < rows and d_lrc_l[idy, idx] == 1:
                idy += 1
            if idy < rows:
                number += 1
                sum_d += d_disparityl[idy, idx]

            idy = y
            idx = x
            while idx < cols and d_lrc_l[idy, idx] == 1:
                idx += 1
            if idx < cols:
                number += 1
                sum_d += d_disparityl[idy, idx]

            idy = y
            idx = x
            while idx >= 0 and d_lrc_l[idy, idx] == 1:
                idx -= 1
            if idx >= 0:
                number += 1
                sum_d += d_disparityl[idy, idx]

            '''
            idy = y
            idx = x
            while idx >= 0 and idy >= 0 and d_lrc_l[idy, idx] == 1:
                idx -= 1
                idy -= 1
            if idx >= 0 and idy >= 0:
                number += 1
                sum_d += d_disparityl[idy, idx]

            idy = y
            idx = x
            while idx < cols and idy < rows and d_lrc_l[idy, idx] == 1:
                idx += 1
                idy += 1
            if idx < cols and idy < rows:
                number += 1
                sum_d += d_disparityl[idy, idx]

            idy = y
            idx = x
            while idx >= 0 and idy < rows and d_lrc_l[idy, idx] == 1:
                idx -= 1
                idy += 1
            if idx >= 0 and idy < rows:
                number += 1
                sum_d += d_disparityl[idy, idx]

            idy = y
            idx = x
            while idx < cols and idy >= 0 and d_lrc_l[idy, idx] == 1:
                idx += 1
                idy -= 1
            if idx < cols and idy >= 0:
                number += 1
                sum_d += d_disparityl[idy, idx]
            '''

            if number > 0:
                d_disparityl_a[y, x] = sum_d / number
            else:
                d_disparityl_a[y, x] = d_disparityl[y, x]

        else:
            d_disparityl_a[y, x] = d_disparityl[y, x]




def disparity_compute_by_gpu(imagel, imager, featuresl, featuresr, detail_time):
    
    cuda.select_device(1)
        
    assert imagel.shape == imager.shape
    height, width = imagel.shape[0:2]
    cols = width
    rows = height

    # transfer images to the device
    d_imagel = cuda.to_device(imagel)
    d_imager = cuda.to_device(imager)

    # transfer features to the memory of device
    d_featuresl = cuda.to_device(featuresl)
    d_featuresr = cuda.to_device(featuresr)

    # allocate memory in the device to hold cost_volume
    cost_volumel = np.ones(shape=[height, width, 128], dtype=np.float32)
    cost_volumer = np.ones(shape=[height, width, 128], dtype=np.float32)
    d_cost_volumel = cuda.to_device(cost_volumel)
    d_cost_volumer = cuda.to_device(cost_volumer)

    s_volumel = np.zeros(shape=[height, width, 128], dtype=np.float32)
    s_volumer = np.zeros(shape=[height, width, 128], dtype=np.float32)
    d_s_volumel = cuda.to_device(s_volumel)
    d_s_volumer = cuda.to_device(s_volumer)

    disparityl = np.zeros(shape=[height, width], dtype=np.float32)
    disparityr = np.zeros(shape=[height, width], dtype=np.float32)
    d_disparityl = cuda.to_device(disparityl)
    d_disparityr = cuda.to_device(disparityr)

    # ------------------------------------ compute cost volume -------------------------------------------- #
    start = time.time()
    blocksize_x = 32
    blocksize_y = 32
    blocksize = (blocksize_x, blocksize_y)
    gridsize_x = (cols + blocksize_x - 1) // blocksize_x
    gridsize_y = (rows + blocksize_y - 1) // blocksize_y
    gridsize = (gridsize_x, gridsize_y)
    compute_cost_volume_kernel[gridsize, blocksize](d_featuresl, d_featuresr, d_cost_volumel, d_cost_volumer)
    # print('detail time of compute cost volume: {}'.format(time.time()-start))
    detail_time[1] += (time.time()-start)

    # ---------------------------------------------- SGM -------------------------------------------------- #
    _start = time.time()
    start = time.time()
    P1 = 2.3
    P2 = 55.9
    threshold = 30
    lamda = 4
    s_sgm_pl = np.zeros(shape=[height, width, 16], dtype=np.float32)
    s_sgm_pr = np.zeros(shape=[height, width, 16], dtype=np.float32)
    d_sgm_pl = cuda.to_device(s_sgm_pl)
    d_sgm_pr = cuda.to_device(s_sgm_pr)

    blocksize_x = 32
    blocksize_y = 32
    blocksize = (blocksize_x, blocksize_y)
    gridsize_x = (cols + blocksize_x - 1) // blocksize_x
    gridsize_y = (rows + blocksize_y - 1) // blocksize_y
    gridsize = (gridsize_x, gridsize_y)
    sgm_penelty_kernel[gridsize, blocksize](d_imagel, d_imager, d_sgm_pl, d_sgm_pr, P1, P2, threshold, lamda)
    # print('detail time of sgm penelty calculate: {}'.format(time.time() - start))
    temp_time = time.time() - start

    blocksize = 256
    pixel_per_block = blocksize // 32
    gridsize_row = (rows + pixel_per_block - 1) // pixel_per_block
    gridsize_col = (cols + pixel_per_block - 1) // pixel_per_block

    start = time.time()
    SGM_UpToDown_kernel[gridsize_col, blocksize](d_cost_volumel, d_cost_volumer, \
                                                 d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr)
    # print('detail time of sgm with up to down path: {}'.format(time.time() - start))

    start = time.time()
    SGM_DownToUp_kernel[gridsize_col, blocksize](d_cost_volumel, d_cost_volumer, \
                                                 d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr)
    # print('detail time of sgm with down to up path: {}'.format(time.time() - start))

    start = time.time()
    SGM_LeftToRight_kernel[gridsize_row, blocksize](d_cost_volumel, d_cost_volumer, \
                                                    d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr)
    # print('detail time of sgm with left to right path: {}'.format(time.time() - start))

    start = time.time()
    SGM_RightToLeft_kernel[gridsize_row, blocksize](d_cost_volumel, d_cost_volumer, \
                                                    d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr)
    # print('detail time of sgm with right to left path: {}'.format(time.time() - start))

    start = time.time()
    SGM_UpToDownAndLeftToRight_kernel[gridsize_col, blocksize](d_cost_volumel, d_cost_volumer, \
                                                               d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr)
    # print('detail time of sgm with up to down and left to right path: {}'.format(time.time() - start))

    start = time.time()
    SGM_DownToUpAndLeftToRight_kernel[gridsize_col, blocksize](d_cost_volumel, d_cost_volumer, \
                                                               d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr)
    # print('detail time of sgm with down to up and left to right path: {}'.format(time.time() - start))

    start = time.time()
    SGM_UpToDownAndRightToLeft_kernel[gridsize_col, blocksize](d_cost_volumel, d_cost_volumer, \
                                                               d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr)
    # print('detail time of sgm with up to down and right to left path: {}'.format(time.time() - start))

    start = time.time()
    SGM_DownToUpAndRightToLeft_kernel[gridsize_col, blocksize](d_cost_volumel, d_cost_volumer, \
                                                               d_s_volumel, d_s_volumer, d_sgm_pl, d_sgm_pr)
    # print('detail time of sgm with down to up and right to left path: {}'.format(time.time() - start))
    # print('total time of sgm: {}'.format(time.time() - _start))
    detail_time[3] += (time.time() - _start - temp_time)

    # -------------------------------------- WTA & Subpixel refinement ------------------------------------ #
    _start = time.time()
    blocksize_x = 32
    blocksize_y = 32
    blocksize = (blocksize_x, blocksize_y)
    gridsize_x = (cols + blocksize_x - 1) // blocksize_x
    gridsize_y = (rows + blocksize_y - 1) // blocksize_y
    gridsize = (gridsize_x, gridsize_y)
    WTA_and_SupixelRefinement_kernel[gridsize, blocksize](d_s_volumel, d_s_volumer, d_disparityl, d_disparityr)
    # print('time of WTA and pixel refinement: {}'.format(time.time() - _start))
    detail_time[4] += (time.time() - _start)

    # ---------------------------------------------- LR-C ----------------------------------------------- #
    _start = time.time()
    s_lrc_l = np.zeros(shape=[height, width], dtype=np.uint8)
    s_lrc_r = np.zeros(shape=[height, width], dtype=np.uint8)
    d_lrc_l = cuda.to_device(s_lrc_l)
    d_lrc_r = cuda.to_device(s_lrc_r)

    d_disparityl_a = cuda.device_array(shape=(height, width), dtype=np.float32)
    d_disparityr_a = cuda.device_array(shape=(height, width), dtype=np.float32)

    blocksize_x = 32
    blocksize_y = 32
    blocksize = (blocksize_x, blocksize_y)
    gridsize_x = (cols + blocksize_x - 1) // blocksize_x
    gridsize_y = (rows + blocksize_y - 1) // blocksize_y
    gridsize = (gridsize_x, gridsize_y)

    is_error_match_kernel[gridsize, blocksize](d_disparityl, d_disparityr, d_lrc_l, d_lrc_r)
    LRC_kernel[gridsize, blocksize](d_disparityl, d_disparityr, d_lrc_l, d_lrc_r, d_disparityl_a, d_disparityr_a)
    # print('detail time of LR-Check: {}'.format(time.time() - start))
    detail_time[5] += (time.time() - _start)

    # ---------------------------------------------- Filter ----------------------------------------------- #
    _start = time.time()
    start = time.time()
    blocksize_x = 16
    blocksize_y = 16
    blocksize = (blocksize_x, blocksize_y)
    gridsize_x = (cols - 4 + blocksize_x - 1) // blocksize_x
    gridsize_y = (rows - 4 + blocksize_y - 1) // blocksize_y
    gridsize = (gridsize_x, gridsize_y)
    Median_Filter_kernel[gridsize, blocksize](d_disparityl_a, d_disparityr_a, d_disparityl, d_disparityr)
    # print('detail time of 5x5 median filter: {}'.format(time.time() - start))

    start = time.time()
    blocksize_x = 16
    blocksize_y = 16
    blocksize = (blocksize_x, blocksize_y)
    gridsize_x = (cols + blocksize_x - 1) // blocksize_x
    gridsize_y = (rows + blocksize_y - 1) // blocksize_y
    gridsize = (gridsize_x, gridsize_y)
    # Bilateral_Filter_kernel[gridsize, blocksize](d_imagel, d_imager, d_disparityl_a, d_disparityr_a, d_disparityl, d_disparityr)
    # print('detail time of 9x9 bilateral filter: {}'.format(time.time() - start))
    # print('total time of filtering: {}'.format(time.time() - _start))
    detail_time[6] += (time.time() - _start)


    # return s_volumel, s_volumer
    return d_disparityl.copy_to_host(), d_disparityr.copy_to_host(), detail_time
