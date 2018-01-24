#!/usr/bin/python
import numpy as np
import cv2
import fnmatch
import os
import time
import PIL.Image as img_tk
import matplotlib.pyplot as plt


def calc_datascale(im_r, im_c, img_n, elem_h, elem_w, step_r, step_c):
    # check setting
    if (((im_r - elem_h)%step_r) != 0) or (((im_c - elem_w)%step_c) != 0):
        print('step in row or column is improper!')
        raise SystemExit

    # row of a sequence
    seq_r = (im_r - elem_h)/step_r + 1
    # # row of a temp sample beam
    # len_bm = seq_r*elem_h
    # column of a sequence
    seq_c = elem_w*elem_h
    # number of all sequences
    seq_n = (im_c - elem_w)/step_c + 1
    all_seq_n = img_n*seq_n

    return seq_r, seq_c, all_seq_n

def imgpair_serialize(path_a, path_b, img_n, elem_h, elem_w, step_r, step_c, mode='DisUS'):
    '''
    :param path_a: image file path a, tissue map
    :param path_b: image file path b, ultrasound
    :param img_n: image number to read
    :param elem_h: height of sequence element
    :param elem_w: width of sequence element
    :param step_r: move step in column
    :param step_c: move step in row
    :param mode: ultrasound effect removal or ultrasound simulation
    :return: the needed sequence pairs list
    '''

    print '### loading dataset from:\n %s \n %s' % (path_a, path_b)
    # check
    if mode != 'SIMULATE' and mode != 'DisUS':
        print('"mode" can only be SIMULATE or DisUS.')
        raise SystemExit
    # file counting
    cnt_f = len(fnmatch.filter(os.listdir(path_a), '*.png'))
    if img_n > cnt_f:
        print('no enough images to process!')
        raise SystemExit

    # basic image info.
    info_im = cv2.imread(path_a + '/' + str(1) + '.png', 0)
    im_r, im_c = info_im.shape

    # pre-calculate the dataset scale
    seq_r, seq_c, all_seq_n = calc_datascale(im_r, im_c, img_n, elem_h, elem_w, step_r, step_c)

    # serialize image pairs to sequence pairs
    feat_mat = np.zeros((seq_r, seq_c, all_seq_n))
    targ_mat = np.zeros((seq_r, seq_c, all_seq_n))
    seq_cnt = 0
    for im_id in range(img_n):
        # print 'serializing No. %d image' % (im_id+1)
        im_a = cv2.imread(path_a + '/' + str(im_id+1) + '.png', 0).astype(float)
        im_b = cv2.imread(path_b + '/' + str(im_id+1) + '.png', 0).astype(float)
        for c_s in range(0, im_c-elem_w+1, step_c):
            c_e = c_s + elem_w
            seq_r_cnt = 0
            for r_s in range(0, im_r-elem_h+1, step_r):
                r_e = r_s + elem_h
                # extract a block
                block_a = im_a[r_s:r_e, c_s:c_e]/255
                block_b = im_b[r_s:r_e, c_s:c_e]/255
                # fold
                beam_a = np.reshape(block_a, (1, elem_h*elem_w))
                beam_b = np.reshape(block_b, (1, elem_h*elem_w))
                # save
                if mode == 'SIMULATE':
                    feat_mat[seq_r_cnt,:,seq_cnt] = beam_a
                    targ_mat[seq_r_cnt,:,seq_cnt] = beam_b
                elif mode == 'DisUS':
                    feat_mat[seq_r_cnt,:,seq_cnt] = beam_b
                    targ_mat[seq_r_cnt,:,seq_cnt] = beam_a
                # row increases
                seq_r_cnt = seq_r_cnt + 1
            # sequence increases
            seq_cnt = seq_cnt + 1

    # transpose 2nd and 3rd dimension
    # necessary for following matrix multiplication
    feat_mat = feat_mat.transpose((0, 2, 1))
    targ_mat = targ_mat.transpose((0, 2, 1))

    print 'success!'

    return feat_mat, targ_mat


def seq3d_deserialize(seq3d, elem_h, step_r, step_c, show_F=False):
    '''
    :param seq3d: sequences to be transformed to an image
    :return: composed image
    note: this version is only suitable for single image, 9-May-2016
    '''
    # transform sequences to an image
    seq_len, seq_n, elem_w = seq3d.shape  # sequence length, sequence number, element number
    res_h = seq_len*elem_h
    res_w = elem_w/elem_h

    t_a = seq3d[:, 0, :]
    t_a = np.reshape(t_a, (res_h, res_w))
    for l in range(1, seq_n):
        temp = seq3d[:, l, :]
        temp = np.reshape(temp, (res_h, res_w))
        t_a = np.append(t_a, temp, axis=1)

    orig_r, orig_c = t_a.shape
    ''' delete overlapping now, but maybe fusion should be better '''
    # delete overlapped rows
    row_l = np.arange(orig_r).tolist()
    OvR_l = np.arange(elem_h-step_r).tolist()
    intv_r = elem_h - step_r
    for r in range(elem_h, orig_r):
        if np.mod(r, elem_h) in OvR_l:
            # fusion
            t_a[r-intv_r,:] = 0.5*t_a[r-intv_r,:] + 0.5*t_a[r,:]
            row_l.remove(r)

    t_a = t_a[np.array(row_l), :]

    # delete overlapped columns
    col_l = np.arange(orig_c).tolist()
    OvR_l = np.arange(res_w-step_c).tolist()
    intv_c = res_w - step_c
    for c in range(res_w, orig_c):
        if (np.mod(c, res_w) in OvR_l):
            # fusion
            t_a[:,c-intv_c] = 0.5*t_a[:,c-intv_c] + 0.5*t_a[:,c]
            col_l.remove(c)
    t_a = t_a[:, np.array(col_l)]

    # convert to intensity
    t_a = (t_a * 255).astype(np.uint8)

    # show
    if show_F == True:
        plt.imshow(t_a, cmap='gray', norm=plt.Normalize(vmin=0, vmax=255))
        plt.show()
    # im = img_tk.fromarray(t_a)
    # im.save("compose.png")

    return t_a


''' demo usage '''
# s_t = time.time()
# feat_mat, targ_mat = imgpair_serialize('../../data/train/img/crop_gr', '../../data/train/img/crop_us', 1, 20, 20, 10, 10, mode='SIMULATE')
# d_I = seq3d_deserialize(feat_mat, 20, 10, 10)
# # print feat_mat.shape
# e_t = time.time()
# print('The code ran for %.5f s' % (e_t - s_t))
# #
# # # for k in range(10):
# # #     plt.imshow(np.concatenate((feat_mat[:,k,:], targ_mat[:,k,:]), axis=0), cmap='gray')
# # #     plt.show()
# #
# plt.imshow(d_I, cmap='gray', norm=plt.Normalize(vmin=0, vmax=255))
# plt.show()