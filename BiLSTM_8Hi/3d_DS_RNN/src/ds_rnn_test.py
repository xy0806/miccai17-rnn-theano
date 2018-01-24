import os
import cv2
import nibabel as nib
import numpy as np
from recur_net import recur_net_c
from utility.vol_grid_io import serialize_single_vol_bbx, serialize_single_vol, deserialize_single_vol_bbx, \
                                get_bounding_box_loc, load_nii2grid, partition_vol2grid2seq, deserialize_cubearray2grid, \
                                deserialize_gridarray2vol


### Test Model ###
if __name__ == '__main__':
    # model path
    model_info_file = '../outcome/model/bilstm_info_08141538.pkl'
    model_weight_file = '../outcome/model/bilstm_model_08141538.pkl'
    # testing data
    test_us_folder = 'tmi17_pkg/dataset/test/us'
    test_auxLabel_folder = 'tmi17_pkg/dataset/alg_seg/FCN_TL_DS/aux_label'
    test_auxMap_folder = 'tmi17_pkg/dataset/alg_seg/FCN_TL_DS/aux_map'
    test_seg_folder = 'tmi17_pkg/dataset/alg_seg/RNN_BiLSTM_8Hi'
    img_n = 56
    # parameter setting
    grid_D = 50
    grid_ita = 3
    cube_D = 7
    cube_ita = 2

    # create recurrent net object
    RecurNet = recur_net_c()
    # load existing model
    RecurNet.load_exist_model(ex_info_file=model_info_file, ex_model_file=model_weight_file)
    # build the symbol model
    RecurNet.build_symbol_model(mode='TEST')

    grid_dim = np.array([grid_D, grid_D, grid_D])
    for k in range(img_n):
    # for k in range(55, 56):
        print 'testing No. %d volume...' % k

        # load testing data
        # auxiliary prediction from 3D FCN
        file_sfix = '_aux.nii'
        aux_path = os.path.join(test_auxLabel_folder, (str(k) + file_sfix))
        orig_vol_dim, bbx_loc = get_bounding_box_loc(aux_path, bbx_ext=10)
        # generate grid list from auxiliary volume
        # aux_grid_list = load_nii2grid(aux_path, grid_D, grid_ita, bbx_loc=bbx_loc)

        # load auxiliary maps
        aux_c0_path = os.path.join(test_auxMap_folder, (str(k) + "_aux_" + str(0) + ".nii"))
        aux_grid_list_c0 = load_nii2grid(aux_c0_path, grid_D, grid_ita, bbx_loc=bbx_loc)
        aux_c1_path = os.path.join(test_auxMap_folder, (str(k) + "_aux_" + str(1) + ".nii"))
        aux_grid_list_c1 = load_nii2grid(aux_c1_path, grid_D, grid_ita, bbx_loc=bbx_loc)
        aux_c2_path = os.path.join(test_auxMap_folder, (str(k) + "_aux_" + str(2) + ".nii"))
        aux_grid_list_c2 = load_nii2grid(aux_c2_path, grid_D, grid_ita, bbx_loc=bbx_loc)
        aux_c3_path = os.path.join(test_auxMap_folder, (str(k) + "_aux_" + str(3) + ".nii"))
        aux_grid_list_c3 = load_nii2grid(aux_c3_path, grid_D, grid_ita, bbx_loc=bbx_loc)

        # generate grid list from ultrasound volume
        file_sfix = '.nii'
        us_path = os.path.join(test_us_folder, (str(k) + file_sfix))
        us_grid_list = load_nii2grid(us_path, grid_D, grid_ita, bbx_loc=bbx_loc)
        # predict on each grid
        label_grid_list = []
        for g in range(len(us_grid_list)):
            us_grid_vol = us_grid_list[g]

            # aux_grid_vol    = aux_grid_list[g]
            aux_grid_vol_c0 = aux_grid_list_c0[g]
            aux_grid_vol_c1 = aux_grid_list_c1[g]
            aux_grid_vol_c2 = aux_grid_list_c2[g]
            aux_grid_vol_c3 = aux_grid_list_c3[g]
            # serialization grid to sequence
            us_mat = partition_vol2grid2seq(us_grid_vol, cube_D, cube_ita, norm_fact=255.0)

            # aux_mat    = partition_vol2grid2seq(aux_grid_vol, cube_D, cube_ita, norm_fact=3.0) #
            aux_mat_c0 = partition_vol2grid2seq(aux_grid_vol_c0, cube_D, cube_ita, norm_fact=255.0)
            aux_mat_c1 = partition_vol2grid2seq(aux_grid_vol_c1, cube_D, cube_ita, norm_fact=255.0)
            aux_mat_c2 = partition_vol2grid2seq(aux_grid_vol_c2, cube_D, cube_ita, norm_fact=255.0)
            aux_mat_c3 = partition_vol2grid2seq(aux_grid_vol_c3, cube_D, cube_ita, norm_fact=255.0)
            # concatenate us and auxiliary serializations
            # feat_mat = np.concatenate((us_mat, aux_mat, aux_mat_c0, aux_mat_c1, aux_mat_c2, aux_mat_c3), axis=1)
            feat_mat = np.concatenate((us_mat, aux_mat_c0, aux_mat_c1, aux_mat_c2, aux_mat_c3), axis=1)
            # RNN prediction
            y_label_seq = RecurNet.test_recur_model(feat_mat)

            # debug_a = np.ones(y_label_seq.shape, 'uint8')
            # y_label_seq = debug_a

            # reshape to grid array
            y_label_volarray = np.reshape(y_label_seq, (y_label_seq.shape[0], cube_D, cube_D, cube_D))
            # deserialize cube array to a grid
            grid_label_pred = deserialize_cubearray2grid(y_label_volarray, grid_dim, cube_D, cube_ita)
            # push to list
            label_grid_list.append(grid_label_pred)

        # deserialize grid array to a volume
        vol_label = deserialize_gridarray2vol(label_grid_list, orig_vol_dim, bbx_loc, grid_D, grid_ita)
        # save
        file_sfix = '_seg.nii.gz'
        save_path = os.path.join(test_seg_folder, (str(k) + file_sfix))
        ref_nii = nib.load(us_path)
        ref_affine = ref_nii.affine
        new_nii = nib.Nifti1Image(vol_label, ref_affine)
        nib.save(new_nii, save_path)


