import cv2
import numpy as np
import os
import re

def load_pfm(fname):
    color = None
    width = None
    height = None
    scale = None
    endian = None

    file = open(fname, 'rb')
    ###   first header: number of channels   ###
    header = file.readline().decode().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    channels = 3 if color == True else 1

    ###   second header: size   ###
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header')

    ### third header:scale and endinaness ###
    scale = float(file.readline().decode().rstrip())
    if scale < 0:
        endian = '<f'
        scale = -scale
    else:
        endian = '>f'

    ###   read the image data   ###
    data = np.fromfile(file, endian)
    file.close()
    shape = (height, width, channels)
    disparity = np.reshape(data, shape)
    disparity = np.flipud(disparity)  # up/down flip
    return disparity, scale

# result_disparity_dir = r'./result/'
# true_disparity_dir = r'./../'
result_disparity_dir = r'/home/rjt1/mc_cnn_fst/result/11_11'
true_disparity_dir = r'/home/rjt1/mc_cnn_fst/result/ground_truth'

result_disparity = []
true_disparity = []
for i in range(5):
    result_disparity.append(os.path.join(result_disparity_dir, 'ld{}.png'.format(i)))
    true_disparity.append(os.path.join(true_disparity_dir, 'disp{}.pfm'.format(i)))

total_error_rate = 0
for i in range(0, len(result_disparity)):
    disp = cv2.imread(result_disparity[i], cv2.IMREAD_GRAYSCALE)
    true_disp, scale = load_pfm(true_disparity[i])
    # true_disp = cv2.imread(true_disparity[i], cv2.IMREAD_GRAYSCALE)

    height, width = disp.shape[0:2]
    true_disp = cv2.resize(true_disp, (width, height))
    true_disp = true_disp / 2
    
    area = np.zeros(shape=[height, width, 3], dtype=np.uint8)

    total_pixel = height*width
    num_of_error_pixel = 0
    for h in range(height):
        for w in range(width):
            if true_disp[h, w] == float('inf') or true_disp[h, w] == 0.0:
                continue
            else:
                if np.fabs(disp[h,w] - true_disp[h, w]) > 1:
                    num_of_error_pixel += 1
                    area[h, w, 2] = 255
                else:
                    area[h, w, 1] = 255

    error_rate = num_of_error_pixel / total_pixel
    total_error_rate += error_rate
    print('error rate of disp{}: {}'.format(i, error_rate))
    cv2.imwrite(r'/home/rjt1/mc_cnn_fst/result/diff/area{}.png'.format(i), area)

print('mean error rate: {}'.format(total_error_rate/len(result_disparity)))
