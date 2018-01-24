#!/usr/bin/python
import os
import cv2
import math
import fnmatch
import natsort
import numpy as np
import nibabel as nib
from scipy import ndimage


####################
# serialization
####################
def vols_serialization(dict_s, sample_idx):
    # us images
    us_mat = serialize_single_vol(im_folder=dict_s['img_a_path'], file_sfix='.nii', idx=sample_idx, cube_D=dict_s['cube_D'], ita=dict_s['ita'], norm_fact=255.0)
    # print us_mat.shape
    # print np.unique(us_mat)
    # auxiliary images
    aux_mat = serialize_single_vol(im_folder=dict_s['img_b_path'], file_sfix='-label.nii', idx=sample_idx, cube_D=dict_s['cube_D'], ita=dict_s['ita'], norm_fact=3.0)
    # print np.unique(aux_mat)
    # print aux_mat.shape
    # concatenate us and auxiliary serializations
    feat_mat = np.concatenate((us_mat, aux_mat), axis=1)
    # print feat_mat.shape

    # label images
    targ_mat = serialize_single_vol(im_folder=dict_s['seg_c_path'], file_sfix='_seg.nii', idx=sample_idx, cube_D=dict_s['cube_D'], ita=dict_s['ita'], norm_fact=1)
    targ_mat = targ_mat.astype('int')
    targ_shape = targ_mat.shape
    targ_entr_mat = np.zeros([targ_shape[0], targ_shape[1], 4]).astype('int')
    ref_t = np.arange(targ_shape[1])
    for k in range(targ_shape[0]):
        targ_entr_mat[k, ref_t, targ_mat[k, :]] = 1

    # print np.unique(targ_mat)
    # print targ_mat.shape

    return feat_mat, targ_entr_mat


def vols_serialization_bbx(dict_s, sample_idx):
    # print 'loading %d volumes...' % sample_idx
    ''' only serialize the bounding box part '''
    # auxiliary images
    file_sfix = '-label.nii'
    aux_path = os.path.join(dict_s['img_b_path'], (str(sample_idx) + file_sfix))
    aux_mat, orig_dim, bbx_loc = serialize_single_vol_bbx(img_path=aux_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_ext=10, norm_fact=3.0)
    # print np.unique(aux_mat)
    # print aux_mat.shape

    # us images
    file_sfix = '.nii'
    us_path = os.path.join(dict_s['img_a_path'], (str(sample_idx) + file_sfix))
    us_mat = serialize_single_vol(img_path=us_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_loc=bbx_loc, norm_fact=255.0)
    # print np.unique(us_mat)
    # print us_mat.shape

    # concatenate us and auxiliary serializations
    feat_mat = np.concatenate((us_mat, aux_mat), axis=1)
    # feat_mat = aux_mat
    # print feat_mat.shape

    # label images
    file_sfix = '_seg.nii'
    targ_path = os.path.join(dict_s['seg_c_path'], (str(sample_idx) + file_sfix))
    targ_mat = serialize_single_vol(img_path=targ_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_loc=bbx_loc, norm_fact=1)
    targ_mat = targ_mat.astype('int')
    targ_shape = targ_mat.shape
    targ_entr_mat = np.zeros([targ_shape[0], targ_shape[1], 4]).astype('int')
    ref_t = np.arange(targ_shape[1])
    for k in range(targ_shape[0]):
        targ_entr_mat[k, ref_t, targ_mat[k, :]] = 1

    # print np.unique(targ_mat)
    # print targ_mat.shape

    return feat_mat, targ_entr_mat


def vols_serialization_subgrid(dict_s, sample_idx):
    ''' only serialize the random grid bounding box part '''
    # print 'loading %d volumes...' % sample_idx
    # auxiliary images
    file_sfix = '_aux.nii'
    aux_path = os.path.join(dict_s['img_b_path'], (str(sample_idx) + file_sfix))
    orig_dim, bbx_loc = get_bounding_box_loc(aux_path, bbx_ext=10)
    sub_grid_loc = get_subgrid_loc(bbx_loc, subgrid_D=dict_s['grid_D'])

    # aux_mat = serialize_single_vol(img_path=aux_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_loc=sub_grid_loc, norm_fact=3.0)

    # load auxiliary maps
    aux_c0_path = os.path.join(dict_s['img_bm_path'], (str(sample_idx) + "_aux_" + str(0) + ".nii"))
    aux_c0_mat = serialize_single_vol(img_path=aux_c0_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_loc=sub_grid_loc, norm_fact=255.0)
    aux_c1_path = os.path.join(dict_s['img_bm_path'], (str(sample_idx) + "_aux_" + str(1) + ".nii"))
    aux_c1_mat = serialize_single_vol(img_path=aux_c1_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_loc=sub_grid_loc, norm_fact=255.0)
    aux_c2_path = os.path.join(dict_s['img_bm_path'], (str(sample_idx) + "_aux_" + str(2) + ".nii"))
    aux_c2_mat = serialize_single_vol(img_path=aux_c2_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_loc=sub_grid_loc, norm_fact=255.0)
    aux_c3_path = os.path.join(dict_s['img_bm_path'], (str(sample_idx) + "_aux_" + str(3) + ".nii"))
    aux_c3_mat = serialize_single_vol(img_path=aux_c3_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_loc=sub_grid_loc, norm_fact=255.0)
    # print np.unique(aux_mat)
    # print aux_mat.shape

    # us images
    file_sfix = '.nii'
    us_path = os.path.join(dict_s['img_a_path'], (str(sample_idx) + file_sfix))
    us_mat = serialize_single_vol(img_path=us_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_loc=sub_grid_loc, norm_fact=255.0)
    # print np.unique(us_mat)
    # print us_mat.shape

    # concatenate us and auxiliary serializations
    # feat_mat = np.concatenate((us_mat, aux_mat, aux_c0_mat, aux_c1_mat, aux_c2_mat, aux_c3_mat), axis=1)
    feat_mat = np.concatenate((us_mat, aux_c0_mat, aux_c1_mat, aux_c2_mat, aux_c3_mat), axis=1)
    # feat_mat = aux_mat
    # print feat_mat.shape

    # label images
    file_sfix = '_seg.nii'
    targ_path = os.path.join(dict_s['seg_c_path'], (str(sample_idx) + file_sfix))
    targ_mat = serialize_single_vol(img_path=targ_path, cube_D=dict_s['cube_D'], ita=dict_s['ita'], bbx_loc=sub_grid_loc, norm_fact=1)
    targ_mat = targ_mat.astype('int')
    targ_shape = targ_mat.shape
    targ_entr_mat = np.zeros([targ_shape[0], targ_shape[1], 4]).astype('int')
    ref_t = np.arange(targ_shape[1])
    for k in range(targ_shape[0]):
        targ_entr_mat[k, ref_t, targ_mat[k, :]] = 1

    # print np.unique(targ_mat)
    # print targ_mat.shape

    return feat_mat, targ_entr_mat


##################
##################
def serialize_single_vol(img_path, cube_D, ita, bbx_loc=None, norm_fact=255.0):
    # print '### loading sample from:\n %s' % (im_folder+ '/' + str(idx))
    vol_file = nib.load(img_path)
    vol_data = vol_file.get_data()
    vol_data = np.asarray(vol_data)

    # crop the volume according to bounding box
    if bbx_loc is not None:
        vol_data = vol_data[bbx_loc[0, 0]:bbx_loc[1, 0], bbx_loc[0, 1]:bbx_loc[1, 1], bbx_loc[0, 2]:bbx_loc[1, 2]]

    # print 'serializing No. %d image' % (idx)
    serial_mat = partition_vol2grid2seq(vol_data, cube_D, ita, norm_fact)

    # plt.imshow(feat_mat[:,0,:], cmap='gray')
    # plt.show()
    # print '# success!'

    return serial_mat


def serialize_single_vol_bbx(img_path, cube_D, ita, bbx_ext=10, norm_fact=255.0):
    # print '### loading sample from:\n %s' % (im_folder+ '/' + str(idx))
    vol_file = nib.load(img_path)
    vol_data = vol_file.get_data()
    vol_data = np.asarray(vol_data)

    # original dimension
    orig_dim = vol_data.shape

    # mask
    vol_mask = (vol_data > 0)
    vol_mask = vol_mask.astype('int')
    # locate object in the volume
    loc = ndimage.find_objects(vol_mask)[0]
    # extend the bounding box
    bbx_loc = np.array([ [loc[0].start, loc[1].start, loc[2].start], [loc[0].stop, loc[1].stop, loc[2].stop] ])
    bbx_loc[0, :] = bbx_loc[0, :] - bbx_ext
    if bbx_loc[0, 0] < 0: bbx_loc[0, 0] = 0
    if bbx_loc[0, 1] < 0: bbx_loc[0, 1] = 0
    if bbx_loc[0, 2] < 0: bbx_loc[0, 2] = 0

    bbx_loc[1, :] = bbx_loc[1, :] + bbx_ext
    if bbx_loc[1, 0] > orig_dim[0]: bbx_loc[1, 0] = orig_dim[0]
    if bbx_loc[1, 1] > orig_dim[1]: bbx_loc[1, 1] = orig_dim[1]
    if bbx_loc[1, 2] > orig_dim[2]: bbx_loc[1, 2] = orig_dim[2]

    # volume in bounding box
    vol_data_bbx = vol_data[bbx_loc[0, 0]:bbx_loc[1, 0], bbx_loc[0, 1]:bbx_loc[1, 1], bbx_loc[0, 2]:bbx_loc[1, 2]]

    # serialize
    serial_mat = partition_vol2grid2seq(vol_data_bbx, cube_D, ita, norm_fact)

    # plt.imshow(feat_mat[:,0,:], cmap='gray')
    # plt.show()
    # print '# success!'

    return serial_mat, orig_dim, bbx_loc


def get_bounding_box_loc(img_path, bbx_ext=10):
    # print '### loading sample from:\n %s' % (im_folder+ '/' + str(idx))
    vol_file = nib.load(img_path)
    vol_data = vol_file.get_data()
    vol_data = np.asarray(vol_data)

    # original dimension
    orig_dim = vol_data.shape

    # mask
    vol_mask = (vol_data > 0)
    vol_mask = vol_mask.astype('int')
    # locate object in the volume
    loc = ndimage.find_objects(vol_mask)[0]
    # extend the bounding box
    bbx_loc = np.array([[loc[0].start, loc[1].start, loc[2].start], [loc[0].stop, loc[1].stop, loc[2].stop]])
    bbx_loc[0, :] = bbx_loc[0, :] - bbx_ext
    if bbx_loc[0, 0] < 0: bbx_loc[0, 0] = 0
    if bbx_loc[0, 1] < 0: bbx_loc[0, 1] = 0
    if bbx_loc[0, 2] < 0: bbx_loc[0, 2] = 0

    bbx_loc[1, :] = bbx_loc[1, :] + bbx_ext
    if bbx_loc[1, 0] > orig_dim[0]: bbx_loc[1, 0] = orig_dim[0]
    if bbx_loc[1, 1] > orig_dim[1]: bbx_loc[1, 1] = orig_dim[1]
    if bbx_loc[1, 2] > orig_dim[2]: bbx_loc[1, 2] = orig_dim[2]

    return orig_dim, bbx_loc


def get_subgrid_loc(bbx_loc, subgrid_D=60):
    limit_e = bbx_loc[1, :] - subgrid_D
    range_x = range(bbx_loc[0, 0], limit_e[0])
    range_y = range(bbx_loc[0, 1], limit_e[1])
    range_z = range(bbx_loc[0, 2], limit_e[2])

    np.random.shuffle(range_x)
    np.random.shuffle(range_y)
    np.random.shuffle(range_z)

    sub_grid_s = np.array([range_x[0], range_y[0], range_z[0]])
    sub_grid_loc = np.array([sub_grid_s, sub_grid_s + subgrid_D])

    return sub_grid_loc



def calc_data_scale(dim, cube_D, ita):
    dim = np.asarray(dim)
    # cube number and overlap along 2 dimensions
    fold = dim / cube_D + ita
    ovlap = np.ceil(np.true_divide((fold * cube_D - dim), (fold - 1)))
    ovlap = ovlap.astype('int')

    # sequence length
    fold = np.ceil(np.true_divide((dim + (fold - 1)*ovlap), cube_D))
    fold = fold.astype('int')
    seq_r = fold[0]*fold[1]*fold[2]
    # element dimension
    seq_c = cube_D**3

    return seq_r, seq_c, fold, ovlap


def load_nii2grid(img_path, grid_D, grid_ita, bbx_loc=None):
    vol_file = nib.load(img_path)
    vol_data = vol_file.get_data()
    vol_data = np.asarray(vol_data)

    # crop the volume according to bounding box
    if bbx_loc is not None:
        vol_data = vol_data[bbx_loc[0, 0]:bbx_loc[1, 0], bbx_loc[0, 1]:bbx_loc[1, 1], bbx_loc[0, 2]:bbx_loc[1, 2]]

    grid_list = partition_vol2grid(vol_data, grid_D, grid_ita)
    return grid_list


def partition_vol2grid(vol_data, grid_D, grid_ita):
    # pre-calculate the dataset scale
    seq_r, seq_c, fold, ovlap = calc_data_scale(vol_data.shape, grid_D, grid_ita)

    # r_fold = fold[0]; c_fold = fold[1]; h_fold = fold[2]
    # r_ovlap = ovlap[0]; c_ovlap = ovlap[1]; h_ovlap = ovlap[2]

    # partition into grids
    dim = np.asarray(vol_data.shape)
    grid_list = []
    for R in range(0, fold[0]):
        r_s = R*grid_D - R*ovlap[0]
        r_e = r_s + grid_D
        if r_e >= dim[0]:
            r_s = dim[0] - grid_D
            r_e = r_s + grid_D
        for C in range(0, fold[1]):
            c_s = C*grid_D - C*ovlap[1]
            c_e = c_s + grid_D
            if c_e >= dim[1]:
                c_s = dim[1] - grid_D
                c_e = c_s + grid_D
            for H in range(0, fold[2]):
                h_s = H*grid_D - H*ovlap[2]
                h_e = h_s + grid_D
                if h_e >= dim[2]:
                    h_s = dim[2] - grid_D
                    h_e = h_s + grid_D
                # partition multiple channels
                vol_grid = vol_data[r_s:r_e, c_s:c_e, h_s:h_e]
                # print vol_grid.shape
                # save
                grid_list.append(vol_grid)

    return grid_list


def partition_vol2grid2seq(vol_data, cube_D, ita, norm_fact):
    # pre-calculate the dataset scale
    seq_r, seq_c, fold, ovlap = calc_data_scale(vol_data.shape, cube_D, ita)
    # serialize image
    serial_mat = np.zeros((seq_r, seq_c))

    # r_fold = fold[0]; c_fold = fold[1]; h_fold = fold[2]
    # r_ovlap = ovlap[0]; c_ovlap = ovlap[1]; h_ovlap = ovlap[2]

    # partition into grids
    dim = np.asarray(vol_data.shape)
    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*cube_D - R*ovlap[0]
        r_e = r_s + cube_D
        if r_e >= dim[0]:
            r_s = dim[0] - cube_D
            r_e = r_s + cube_D
        for C in range(0, fold[1]):
            c_s = C*cube_D - C*ovlap[1]
            c_e = c_s + cube_D
            if c_e >= dim[1]:
                c_s = dim[1] - cube_D
                c_e = c_s + cube_D
            for H in range(0, fold[2]):
                h_s = H*cube_D - H*ovlap[2]
                h_e = h_s + cube_D
                if h_e >= dim[2]:
                    h_s = dim[2] - cube_D
                    h_e = h_s + cube_D
                # partition multiple channels
                vol_grid = vol_data[r_s:r_e, c_s:c_e, h_s:h_e] / norm_fact
                # print vol_grid.shape
                # print cube_D**3
                seq_elem = np.reshape(vol_grid, (1, cube_D**3))
                # save
                serial_mat[p_count, :] = seq_elem

                p_count = p_count + 1

    return serial_mat


####################
# deserialization
####################
def deserialize_single_vol_bbx(vol_data_array, orig_dim, bbx_loc, cube_D, ita):
    # create final volume
    desr_vol = (np.zeros(orig_dim)).astype('uint8')
    # create bounding box volume
    bbx_dim = np.array([bbx_loc[1, 0]-bbx_loc[0, 0], bbx_loc[1, 1]-bbx_loc[0, 1], bbx_loc[1, 2]-bbx_loc[0, 2]])
    # pre-calculate the dataset scale
    seq_r, seq_c, fold, ovlap = calc_data_scale(bbx_dim, cube_D, ita)
    # reconstruct volume from grids
    label_0_mat = (np.zeros(bbx_dim)).astype('uint8')
    label_1_mat = (np.zeros(bbx_dim)).astype('uint8')
    label_2_mat = (np.zeros(bbx_dim)).astype('uint8')
    label_3_mat = (np.zeros(bbx_dim)).astype('uint8')

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*cube_D - R*ovlap[0]
        r_e = r_s + cube_D
        for C in range(0, fold[1]):
            c_s = C*cube_D - C*ovlap[1]
            c_e = c_s + cube_D
            for H in range(0, fold[2]):
                h_s = H*cube_D - H*ovlap[2]
                h_e = h_s + cube_D
                # histogram for voting
                idx_0 = (vol_data_array[p_count, :, :] == 0)
                # idx_0 = (np.asarray(idx_0)).astype('int')
                idx_1 = (vol_data_array[p_count, :, :] == 1)
                # idx_1 = (np.asarray(idx_1)).astype('int')
                idx_2 = (vol_data_array[p_count, :, :] == 2)
                # idx_2 = (np.asarray(idx_2)).astype('int')
                idx_3 = (vol_data_array[p_count, :, :] == 3)
                # idx_3 = (np.asarray(idx_3)).astype('int')

                label_0_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_0_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_0
                label_1_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_1_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_1
                label_2_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_2_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_2
                label_3_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_3_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_3

                p_count += 1

    label_mat = np.stack((label_0_mat, label_1_mat, label_2_mat, label_3_mat), axis=3)
    # print label_mat.shape
    # print 'label mat unique:'
    # print np.unique(label_mat)

    label_mat = np.argmax(label_mat, axis=3)
    # print np.unique(label_mat)

    # refill bounding box
    desr_vol[bbx_loc[0, 0]:bbx_loc[1, 0], bbx_loc[0, 1]:bbx_loc[1, 1], bbx_loc[0, 2]:bbx_loc[1, 2]] = label_mat

    return desr_vol


def deserialize_cubearray2grid(cube_array, grid_dim, cube_D, cube_ita):
    # pre-calculate the dataset scale
    seq_r, seq_c, fold, ovlap = calc_data_scale(grid_dim, cube_D, cube_ita)
    # reconstruct volume from grids
    label_0_mat = (np.zeros(grid_dim)).astype('uint8')
    label_1_mat = (np.zeros(grid_dim)).astype('uint8')
    label_2_mat = (np.zeros(grid_dim)).astype('uint8')
    label_3_mat = (np.zeros(grid_dim)).astype('uint8')

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*cube_D - R*ovlap[0]
        r_e = r_s + cube_D
        if r_e >= grid_dim[0]:
            r_s = grid_dim[0] - cube_D
            r_e = r_s + cube_D
        for C in range(0, fold[1]):
            c_s = C*cube_D - C*ovlap[1]
            c_e = c_s + cube_D
            if c_e >= grid_dim[1]:
                c_s = grid_dim[1] - cube_D
                c_e = c_s + cube_D
            for H in range(0, fold[2]):
                h_s = H*cube_D - H*ovlap[2]
                h_e = h_s + cube_D
                if h_e >= grid_dim[2]:
                    h_s = grid_dim[2] - cube_D
                    h_e = h_s + cube_D
                # histogram for voting
                idx_0 = (cube_array[p_count] == 0)
                # idx_0 = (np.asarray(idx_0)).astype('int')
                idx_1 = (cube_array[p_count] == 1)
                # idx_1 = (np.asarray(idx_1)).astype('int')
                idx_2 = (cube_array[p_count] == 2)
                # idx_2 = (np.asarray(idx_2)).astype('int')
                idx_3 = (cube_array[p_count] == 3)
                # idx_3 = (np.asarray(idx_3)).astype('int')

                label_0_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_0_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_0
                label_1_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_1_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_1
                label_2_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_2_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_2
                label_3_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_3_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_3

                p_count += 1

    label_mat = np.stack((label_0_mat, label_1_mat, label_2_mat, label_3_mat), axis=3)
    # print label_mat.shape
    # print 'label mat unique:'
    # print np.unique(label_mat)

    grid_vol = np.argmax(label_mat, axis=3)
    # print np.unique(label_mat)

    return grid_vol


def deserialize_gridarray2vol(grid_list, vol_dim, bbx_loc, grid_D, grid_ita):
    # create final volume
    desr_vol = (np.zeros(vol_dim)).astype('uint8')
    # create bounding box volume
    bbx_dim = np.array([bbx_loc[1, 0]-bbx_loc[0, 0], bbx_loc[1, 1]-bbx_loc[0, 1], bbx_loc[1, 2]-bbx_loc[0, 2]])
    # pre-calculate the dataset scale
    seq_r, seq_c, fold, ovlap = calc_data_scale(bbx_dim, grid_D, grid_ita)
    # reconstruct volume from grids
    label_0_mat = (np.zeros(bbx_dim)).astype('uint8')
    label_1_mat = (np.zeros(bbx_dim)).astype('uint8')
    label_2_mat = (np.zeros(bbx_dim)).astype('uint8')
    label_3_mat = (np.zeros(bbx_dim)).astype('uint8')

    p_count = 0
    for R in range(0, fold[0]):
        r_s = R*grid_D - R*ovlap[0]
        r_e = r_s + grid_D
        if r_e >= bbx_dim[0]:
            r_s = bbx_dim[0] - grid_D
            r_e = r_s + grid_D
        for C in range(0, fold[1]):
            c_s = C*grid_D - C*ovlap[1]
            c_e = c_s + grid_D
            if c_e >= bbx_dim[1]:
                c_s = bbx_dim[1] - grid_D
                c_e = c_s + grid_D
            for H in range(0, fold[2]):
                h_s = H*grid_D - H*ovlap[2]
                h_e = h_s + grid_D
                if h_e >= bbx_dim[2]:
                    h_s = bbx_dim[2] - grid_D
                    h_e = h_s + grid_D
                # histogram for voting
                idx_0 = (grid_list[p_count] == 0)
                # idx_0 = (np.asarray(idx_0)).astype('int')
                idx_1 = (grid_list[p_count] == 1)
                # idx_1 = (np.asarray(idx_1)).astype('int')
                idx_2 = (grid_list[p_count] == 2)
                # idx_2 = (np.asarray(idx_2)).astype('int')
                idx_3 = (grid_list[p_count] == 3)
                # idx_3 = (np.asarray(idx_3)).astype('int')

                label_0_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_0_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_0
                label_1_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_1_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_1
                label_2_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_2_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_2
                label_3_mat[r_s:r_e, c_s:c_e, h_s:h_e] = label_3_mat[r_s:r_e, c_s:c_e, h_s:h_e] + idx_3

                p_count += 1

    label_mat = np.stack((label_0_mat, label_1_mat, label_2_mat, label_3_mat), axis=3)
    # print label_mat.shape
    # print 'label mat unique:'
    # print np.unique(label_mat)

    label_mat = np.argmax(label_mat, axis=3)
    # print np.unique(label_mat)

    # refill bounding box
    desr_vol[bbx_loc[0, 0]:bbx_loc[1, 0], bbx_loc[0, 1]:bbx_loc[1, 1], bbx_loc[0, 2]:bbx_loc[1, 2]] = label_mat

    return desr_vol





