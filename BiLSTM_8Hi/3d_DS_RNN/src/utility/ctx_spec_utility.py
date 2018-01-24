#!/usr/bin/python
import os
import cv2
import numpy as np
import ConfigParser
from utility.polar2feat import polar_serialize_Im8Label, polar_serialize_single, polarshape_deserialize

def ctx_load_train_ini(ini_file):
    '''
     usage: # ctx_param_L = ctx_load_ini('/home/xyang/xinyang/Project/S_rnn_201605/recur_code/20160903_ctx_scale/context_param_a.ini')
    '''
    # initialize
    cf = ConfigParser.ConfigParser()
    cf.read(ini_file)
    # dictionary list
    ctx_param_L = []

    s = cf.sections()

    for d in range(len(s)):
        # create dictionary
        level_dict = dict(img_a_path    = cf.get(s[d], "img_a_path"),
                          img_b_path    = cf.get(s[d], "img_b_path"),
                          seg_c_path    = cf.get(s[d], "seg_c_path"),
                          info_file     = cf.get(s[d], "info_file"),
                          model_file    = cf.get(s[d], "model_file"),
                          pass_root     = cf.get(s[d], "pass_root"),
                          p_prefix      = cf.get(s[d], "p_prefix"),

                          img_n         = cf.getint(s[d], "img_n"),
                          elem_h        = cf.getint(s[d], "elem_h"),
                          elem_w        = cf.getint(s[d], "elem_w"),
                          column_s      = cf.getint(s[d], "column_s"),

                          n_input       = cf.getint(s[d], "elem_h") * cf.getint(s[d], "elem_w") * 2,
                          n_output      = cf.getint(s[d], "elem_h") * cf.getint(s[d], "elem_w"),
                          n_hidden      = cf.getint(s[d], "n_hidden"),
                          recur_type    = cf.get(s[d], "recur_type"),
                          n_layer       = cf.getint(s[d], "n_layer"),

                          minibatch     = cf.getint(s[d], "minibatch"),
                          learning_rate = cf.getfloat(s[d], "learning_rate"),
                          n_epochs      = cf.getint(s[d], "n_epochs"),
                          vld_intval    = cf.getint(s[d], "vld_intval"),
                          optimizer     = cf.get(s[d], "optimizer"),
                          earlyErr_T    = cf.getfloat(s[d], "earlyErr_T"))
        # add to list
        ctx_param_L.append(level_dict)

    return ctx_param_L


def ctx_load_test_ini(ini_file):
    # initialize
    cf = ConfigParser.ConfigParser()
    cf.read(ini_file)
    # dictionary list
    ctx_param_L = []

    s = cf.sections()

    for d in range(len(s)):
        # create dictionary
        level_dict = dict(orig_us_path  = cf.get(s[d], "orig_us_path"),
                          img_a_path    = cf.get(s[d], "img_a_path"),
                          img_b_path    = cf.get(s[d], "img_b_path"),
                          seg_c_path    = cf.get(s[d], "seg_c_path"),
                          cntr_path     = cf.get(s[d], "cntr_path"),
                          info_file     = cf.get(s[d], "info_file"),
                          model_file    = cf.get(s[d], "model_file"),
                          pass_root     = cf.get(s[d], "pass_root"),
                          p_prefix      = cf.get(s[d], "p_prefix"),
                          fusion_linear = cf.get(s[d], "fusion_linear"),
                          compare_show  =cf.get(s[d], "compare_show"),

                          img_n         = cf.getint(s[d], "img_n"),
                          elem_h        = cf.getint(s[d], "elem_h"),
                          elem_w        = cf.getint(s[d], "elem_w"),
                          column_s      = cf.getint(s[d], "column_s"),
                          view_N        =cf.getint(s[d], "view_N"))
        # add to list
        ctx_param_L.append(level_dict)

    return ctx_param_L


def ctx_load_feat8targ(ctx_dict, level):
    # load training dataset #
    # us images
    us_mat = polar_serialize_single(im_folder=ctx_dict['img_a_path'], img_n=ctx_dict['img_n'], elem_h=ctx_dict['elem_h'],
                                    elem_w=ctx_dict['elem_w'], column_s=ctx_dict['column_s'], norm_fact=255)
    # map images
    map_mat = 0
    if level == 0:
        map_shape = us_mat.shape
        map_mat = np.zeros(map_shape)
    else:
        map_mat = polar_serialize_single(im_folder=ctx_dict['img_b_path'], img_n=ctx_dict['img_n'], elem_h=ctx_dict['elem_h'],
                                         elem_w=ctx_dict['elem_w'], column_s=ctx_dict['column_s'], norm_fact=255)
    # label images
    label_mat = polar_serialize_single(im_folder=ctx_dict['seg_c_path'], img_n=ctx_dict['img_n'], elem_h=ctx_dict['elem_h'],
                                       elem_w=ctx_dict['elem_w'], column_s=ctx_dict['column_s'], norm_fact=255)
    # concatenate us and map images
    feat_mat = np.concatenate((us_mat, map_mat), axis=2)
    # feat_mat = us_mat
    targ_mat = label_mat

    return feat_mat, targ_mat


def ctx_paste2bkg(ctx_dict, map_list, linear_H=400, linear_W=400):
    pst_img_list = []
    ### paste prediction to background image ###
    bk_img = np.zeros((linear_H, linear_W))
    bk_img = bk_img.astype('uint8')
    column_s = ctx_dict['column_s']
    elem_w = ctx_dict['elem_w']
    for k in range(len(map_list)):
        map_img = bk_img.copy()
        img_temp = map_list[k].astype('uint8')
        map_img[:, column_s:column_s + elem_w] = img_temp
        pst_img_list.append(map_img)

    return pst_img_list


def ctx_save_map(ctx_dict, ctx_level, pst_img_list, ini_file):
    ### save map images ###
    root_folder = ctx_dict['pass_root']
    prefix = ctx_dict['p_prefix']
    # create folder
    dir_name = os.path.join(root_folder, (prefix + '_' + str(ctx_level)))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    print 'saving images...'
    for k in range(len(pst_img_list)):
        img_temp = pst_img_list[k].astype('uint8')
        img_path = os.path.join(dir_name, (str(k) + '.png'))
        cv2.imwrite(img_path, img_temp)


def ctx_update_ini(ctx_dict, ctx_level, ini_file):
    ### update .ini file ###
    root_folder = ctx_dict['pass_root']
    prefix = ctx_dict['p_prefix']
    # read original .ini file
    cf = ConfigParser.ConfigParser()
    cf.read(ini_file)
    s = cf.sections()
    # update the training map path for ctx_level+1
    dir_name = os.path.join(root_folder, (prefix + '_' + str(ctx_level)))
    cf.set(s[ctx_level+1], "img_b_path", dir_name)
    cf.write(open(ini_file, "w"))