#!/usr/bin/python
import ConfigParser

def load_train_ini(ini_file):
    # initialize
    cf = ConfigParser.ConfigParser()
    cf.read(ini_file)
    # dictionary list
    param_Levels = []

    s = cf.sections()
    for d in range(len(s)):
        # create dictionary
        level_dict = dict(img_a_path    = cf.get(s[d], "img_a_path"),
                          img_b_path    = cf.get(s[d], "img_b_path"),
                          img_bm_path   = cf.get(s[d], "img_bm_path"),
                          seg_c_path    = cf.get(s[d], "seg_c_path"),
                          info_file     = cf.get(s[d], "info_file"),
                          model_file    = cf.get(s[d], "model_file"),
                          err_txt       = cf.get(s[d], "err_txt"),
                          pass_root     = cf.get(s[d], "pass_root"),
                          p_prefix      = cf.get(s[d], "p_prefix"),

                          img_n         = cf.getint(s[d], "img_n"),
                          grid_D        = cf.getint(s[d], "grid_D"),
                          cube_D        = cf.getint(s[d], "cube_D"),
                          ita           = cf.getint(s[d], "ita"),

                          n_input       = cf.getint(s[d], "cube_D")**3*5,
                          n_output      = cf.getint(s[d], "cube_D")**3,
                          recur_type    = cf.get(s[d], "recur_type"),
                          minibatch     = cf.getint(s[d], "minibatch"),
                          n_hidden      = cf.getint(s[d], "n_hidden"),
                          n_layer       = cf.getint(s[d], "n_layer"),
                          n_class       = cf.getint(s[d], "n_class"),

                          learning_rate = cf.getfloat(s[d], "learning_rate"),
                          lr_interval   = cf.getint(s[d], "lr_interval"),
                          lr_ita        = cf.getint(s[d], "lr_ita"),

                          n_epochs      = cf.getint(s[d], "n_epochs"),
                          vld_intval    = cf.getint(s[d], "vld_intval"),
                          optimizer     = cf.get(s[d], "optimizer"),
                          earlyErr_T    = cf.getfloat(s[d], "earlyErr_T"))
        # add to list
        param_Levels.append(level_dict)

    return param_Levels