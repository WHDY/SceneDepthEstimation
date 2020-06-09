import numpy as np
import cv2
import os
import re
import time

'''
    仍然存在的问题：
    1.scale的作用是什么？
    2.目前这样处理视差生成数据集是否合理，比如是否还存在其他的漏洞没有考虑，数据集大小是否合适，有样本可能重复是否合适
    3.dispy是否得考虑进去
    4.预操作处理还未进行
    5.load_pfm函数并不是很完美
'''
class ImagePatchesGenerator:
    def __init__(self, path_left,
                 path_right_pos,
                 path_right_neg,
                 number_of_data_set_head,
                 number_of_data_set_tail,
                 patch_size=(11, 11),
                 batch_size=128,
                 shuffle=True):
        '''   initial parameters   '''

        self.dataset_path_left = path_left
        self.dataset_path_right_pos = path_right_pos
        self.dataset_path_right_neg = path_right_neg
        self.number_of_data_set_head = number_of_data_set_head
        self.number_of_data_set_tail = number_of_data_set_tail
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.dataset_size = (number_of_data_set_tail - number_of_data_set_head) // batch_size


        self.pointer = 0   # pointer indicates which image are next to be used
        self.shuffle = shuffle
        self.patches_list = []
        if self.shuffle == True:
            self.patches_list = np.random.permutation(self.number_of_data_set_tail - self.number_of_data_set_head)
        else:
            self.patches_list = np.linspace(0, self.number_of_data_set_tail - self.number_of_data_set_head, \
                                            self.number_of_data_set_tail - self.number_of_data_set_head, \
                                            endpoint=False, dtype=np.int32)


    def next_batch(self):

        patches_left = np.ndarray([self.batch_size, self.patch_size[0], self.patch_size[1], 1], dtype=np.float32)
        patches_right_neg = np.ndarray([self.batch_size, self.patch_size[0], self.patch_size[1], 1], dtype=np.float32)
        patches_right_pos = np.ndarray([self.batch_size, self.patch_size[0], self.patch_size[1], 1], dtype=np.float32)
        for i in range(self.batch_size):
            # print(i)
            # temp, s = self.load_pfm(self.dataset_path_right_neg.format(self.patches_list[i + self.number_of_data_set_head + self.pointer]))
            # print(temp.shape)
            patches_left[i, :, :, :], scalel = self.load_pfm(self.dataset_path_left.format(self.number_of_data_set_head + self.patches_list[i + self.pointer]))
            patches_right_pos[i, :, :, :], scalel = self.load_pfm(self.dataset_path_right_pos.format(self.number_of_data_set_head + self.patches_list[i + self.pointer]))
            patches_right_neg[i, :, :, :], scalel = self.load_pfm(self.dataset_path_right_neg.format(self.number_of_data_set_head + self.patches_list[i + self.pointer]))
        self.update_pointer()
        return patches_left, patches_right_pos, patches_right_neg


    def load_pfm(self, fname):
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


    def update_pointer(self):
        self.pointer += self.batch_size


    def reset_pointer(self):
        self.pointer = 0
        if self.shuffle == True:
            self.patches_list = np.random.permutation(self.number_of_data_set_tail - self.number_of_data_set_head)
        else:
            self.patches_list = np.linspace(0, self.number_of_data_set_tail - self.number_of_data_set_head, \
                                            self.number_of_data_set_tail - self.number_of_data_set_head, \
                                            endpoint=False, dtype=np.int32)


if __name__=="__main__":
    test_data_set = ImagePatchesGenerator(r'./training_data_set/left/{}.pfm',
                                          r'./training_data_set/right_pos/{}.pfm',
                                          r'./training_data_set/right_neg/{}.pfm',
                                          0,
                                          236433,
                                          patch_size=(11, 11),
                                          shuffle=False)
    patches_left, patches_right_pos, patches_right_neg = test_data_set.next_batch()
    print(patches_left.shape)
    patches_left, patches_right_pos, patches_right_neg = test_data_set.next_batch()
