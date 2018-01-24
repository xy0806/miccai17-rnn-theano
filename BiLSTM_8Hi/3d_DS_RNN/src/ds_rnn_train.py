from utility.ini_file_io import load_train_ini
from recur_net import recur_net_c

if __name__ == '__main__':

    # load training parameter #
    tr_ini_file = '../outcome/model/ini/tr_param.ini'
    param_Levels = load_train_ini(tr_ini_file)
    level_N = len(param_Levels)

    # dataset resize

    # train with .ini file #
    for s in range(1):
        print '============= training level #', s, '... ============='
        dict_s = param_Levels[s]
        # create recurrent net - train
        RecurNet_train = recur_net_c(recur_type=dict_s['recur_type'], n_input=dict_s['n_input'], minibatch=dict_s['minibatch'], n_hidden=dict_s['n_hidden'],
                                    n_layer=dict_s['n_layer'], n_output=dict_s['n_output'], n_class=dict_s['n_class'], optimizer=dict_s['optimizer'], learning_rate=dict_s['learning_rate'],
                                    n_epochs=dict_s['n_epochs'], vld_intval=dict_s['vld_intval'], earlyErr_T=dict_s['earlyErr_T'],
                                    info_file=dict_s['info_file'], model_file=dict_s['model_file'], err_txt=dict_s['err_txt'])
        # build recurrent model
        RecurNet_train.build_symbol_model(mode='TRAIN')
        # train recurrent model with on-the-fly sampling
        RecurNet_train.train_recur_model_online_subgrid_sample(dict_s, s)
