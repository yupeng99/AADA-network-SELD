#!/usr/bin/env python
# Helper on how to use the parser:
# https://github.com/bw2/ConfigArgParse

import argparse
import math
import os
import yaml
import shutil
from easydict import EasyDict
from datetime import datetime
import configargparse

def get_result_dir_path(experiment_description: str, root: str = './results'):
    """Returns path where to save training results of a experiment specific result.

    Args:
        root: root path of where to save
        experiment_description: "epoch=50-batch=128-arch=FCN-data=FULL"

    Create the directory, "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    Return directory path(str):
        "result-20190604_13_40_52-epoch=50-batch=128-arch=FCN-data=FULL"

    """
    from datetime import datetime
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y%m%d_%H_%M_%S")

    path = f"{experiment_description}__{date_time}"
    path = os.path.join(root, path)
    try:
        os.makedirs(path)
    except OSError:
        if os.path.exists(path):
            print("Path already exists")
        else:
            print(f"Couldn't create {path}.")
            path = root
    else:
        print(f"Save weights to {path}")
    finally:
        return path

def list_of_2d_tuples(s):
    '''
    For the argparser, this reads a string in the format:
    --max_pool 2,2 4,1 3,3
    And returns a list of tuples, as:
    [(2,2), (4,1), (3,3)]

    # Call it like this in terminal
    #--max_pool 2,2 4,1 3,3

    # debug:
    # config = parser.parse_args(["--max_pool", "2,2", "2,2", "2,2", "2,2", "3,3", "1,4"])  # test for the list_of_2d_tuples type
    '''
    try:
        yolo = iter(s.split(' '))
        for tmp in yolo:
            x,y = tmp.split(',')
            return int(x), int(y)
    except:
        raise argparse.ArgumentTypeError("Error reading parameters. Tuples must be x,y")

def get_parameters():
    #p = configargparse.ArgParser(default_config_files=['./configs/*.yaml'], config_file_parser_class=configargparse.YAMLConfigFileParser)
    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    #p = configargparse.ArgParser(default_config_files=['./configs/*.yaml'])
    p.add('-c', '--my-config', required=True, is_config_file=True, help='config file path', default='./configs/yu_dcase2022.yaml')

    # Experiment
    p.add_argument('--exp_name', help="Optional experiment name.")
    p.add_argument('--exp_group', help="Optional experiment group, useful to search for runs in the logs. This is added to the exp name.")
    p.add_argument('--seed_mode', help="Mode for random seeds.", choices=['balanced', 'random'])
    p.add_argument('--seed', type=int)
    p.add_argument('--mode', help='train or eval', choices=['train', 'valid', 'eval'])
    p.add_argument('--oracle_mode', action='store_true', help='Enables oracle mode, used to get the upper bound of the metrics.')
    p.add_argument('--debug', action='store_true', help='Enables debug mode, with short runs and no logging.')
    p.add_argument('--profiling', action='store_true', help='Enables profiling mode.')
    p.add_argument('--job_id', type=str, default='', help='Job id to append to the experiment name. Helps getting the job log.')
    p.add_argument('--task_id', type=str, default='', help='Task id when using array jobs.')
    p.add_argument('--logging_dir', help='Directory to save logs and results.')
    p.add_argument('--wandb', action='store_true', help='Enable wandb to log runs.')

    # Training arguments
    p.add_argument('--solver', type=str, choices=['DAN', 'vanilla'],  help='Solver to use.')
    p.add_argument('--num_iters', type=int, help='Num of trianing iterations.')
    p.add_argument('--batch_size', type=int, help='Batch size.')
    p.add_argument('--num_workers', type=int, help='Num workers for dataloader.')
    p.add_argument('--print_every', type=int, help='Print current status every x training iterations.')
    p.add_argument('--logging_interval', type=int, help='Validation interval')
    p.add_argument('--lr_scheduler_step', type=int, help='Step for the lr scheduler')
    p.add_argument('--lr', type=float, help='Learning rate for optimizer.')
    p.add_argument('--lr_min', type=float, help='Minimum learning rate so that the Scheduler does not go too low.')
    p.add_argument('--lr_decay_rate', type=float, help='Decay rate for the lr scheduler.')
    p.add_argument('--lr_patience_times', type=float, help='Validaiton steps patienece for the lr scheduler.')
    p.add_argument('--w_rec', type=float, default=0, help='Weight for the reconstruction loss of the generator.')
    p.add_argument('--w_rec_spec', type=float, default=0, help='Weight for the spectral reconstruction loss of the generator.')
    p.add_argument('--w_adv', type=float, default=1, help='Weight for the adversarial loss of the generator.')

    # Curriculum
    p.add_argument('--curriculum_scheduler', type=str, help='Select scheduler type for the curriculum learning', choices=['linear', 'fixed', 'loss', 'seld_metric'])
    p.add_argument('--curriculum_D_threshold_min', type=float, default=0.0, help='Update for the discrminator threshold when using curriculum learning. This adds to the D_threshold.')
    p.add_argument('--curriculum_D_threshold_max', type=float, default=0.0, help='Update for the discrminator threshold when using curriculum learning. This adds to the D_threshold.')
    p.add_argument('--curriculum_w_adv', type=float, default=0.0, help='Update for w-adv when using curriculim learning. This adds to the original w_adv value')

    # Training discrminator arguments
    p.add_argument('--D_lr', type=float, default=1e-5, help='Learning rate')
    p.add_argument('--D_lr_min', type=float, default=1e-8, help='Discriminator, minimum learning rate')
    p.add_argument('--D_lr_decay_rate', type=float, default=0.9, help='Discriminator, learning rate decay for the scheduler')
    p.add_argument('--D_lr_patience_times', type=float, default=3, help='Discriminator, learning rate patience for the scheduler')
    p.add_argument('--D_lr_weight_decay', type=float, default=0.001)
    p.add_argument('--D_lr_scheduler', type=str, choices=["lrstep", "warmup", "cosine", "cosine-restart"])
    p.add_argument('--D_lr_scheduler_step', type=int, help='Step for the lr scheduler of the discriminator')
    p.add_argument('--D_lr_scheduler_T_0', type=int, default=5000, help='T_0 parameter for the cosine-restart scheduler of the discriminator, this the period.')
    p.add_argument('--D_lr_scheduler_T_mult', type=int, default=1, help='T_mult paramter for the cosine-restaet scheduler of the discriminator, this is the period change.')
    p.add_argument('--D_grad_clip', type=float, default=-1, help='Gradient clipping for the discriminator. Set to -1 to disable.')

    p.add_argument('--G_rec_spec', type=str, default='l2', choices=['l2', 'l1'])
    p.add_argument('--G_crit', type=str, default='minmax', choices=['minmax', 'non-sat', 'ls', 'wass', 'hinge'])
    p.add_argument('--D_crit', type=str, default='minmax', choices=['minmax', 'non-sat', 'ls', 'wass', 'hinge'])
    p.add_argument('--disc_algorithm', type=str, default='vanilla', choices=['vanilla'])
    p.add_argument('--D_batches', type=int, default='0', help='Number of independent batches to train the discrminator. Set to 0 to use the same batch.')

    # Model arguments
    p.add_argument('--model', help='Model to use.')
    p.add_argument('--model_features_transform', help='Features transform to use in the model', choices=['stft_iv_base','mel_iv_base','stft_iv', 'mel_iv', 'mel_base', 'bandpass', 'none'])
    p.add_argument('--model_spatialmixup', action='store_true', help='Enables spatial mixuo as data augmentation.')
    p.add_argument('--model_augmentation', action='store_true', help='Enable data augmentation in audio domain')
    p.add_argument('--model_spec_augmentation', action='store_true', help='Enable data augmentation in spectrogram domain')
    p.add_argument('--model_rotations', action='store_true', help='Enable soundfiled rotations for audio and labels.')
    p.add_argument('--model_rotations_mode', type=str, help='Rotation mode.', choices=['azi', 'ele', 'azi-ele'])
    p.add_argument('--model_rotations_noise', action='store_true', help='Enable rotations (angular) noise.')
    p.add_argument('--model_loss_fn', help='Loss function.', choices=['adpit','mse', 'bce', 'l1'])
    p.add_argument('--model_normalization', help='Threshold for detecting events during evaluation.')
    p.add_argument('--detection_threshold', type=float, help='Threshold for detecting events during evaluation.')
    p.add_argument('--thresh_unify', type=float, help='Threshold for unify detections during evaluation')
    p.add_argument('--use_mixup', action='store_true')
    p.add_argument('--mixup_alpha', type=float)
    p.add_argument('--input_shape', nargs='+', type=int, help='Input shape for the model.')   #'input_shape': [4, 144000], when using sample cnn
    p.add_argument('--output_shape', nargs='+', type=int, help='Output shape of the model, so the predictions.')
    p.add_argument('--t_pool_size', nargs='+', type=int, help='Output shape of the model, so the predictions.')

    # Discriminator arguments
    p.add_argument('--disc', choices=['DiscriminatorModularThreshold'], help='Model to use.')
    p.add_argument('--disc_input_shape', nargs='+', type=int, default=[3, 257, 256], help='Input shape for the discriminator. Mostly for the torchsummary')
    p.add_argument('--disc_feature_maps', type=int, default=64, help='Feature maps for the first layer of the discrminator.')
    p.add_argument('--disc_final', choices=['none', 'relu'],  help='Activation function for the last layer.')
    p.add_argument('--disc_kernels', nargs='+', type=list_of_2d_tuples, help='Kernel sizes, for each block.')
    p.add_argument('--disc_strides', nargs='+', type=list_of_2d_tuples, help='Kernel strides, for each block.')
    p.add_argument('--disc_padding', nargs='+', type=int, help='Kernel padding, for each block.')
    p.add_argument('--disc_normalization', choices=['instance', 'layer', 'batch'], help='Model to use.')
    p.add_argument('--disc_block', choices=['coord', 'basic'], help='Discriminator block to use.')
    p.add_argument('--disc_conditioning', choices=['none', 'none-upsample', 'concat'], help='Model to use.')
    p.add_argument('--disc_final_multi', type=int, help='Multiplier value to match the output of the last layer to the number of units in the MLP. This should change if the number of frames changes.')
    p.add_argument('--disc_threshold_min', type=float, default=1.2, help='Minimum valiue for the threshold layer.')
    p.add_argument('--disc_threshold_max', type=float, default=1.2, help='Maximum valiue for the threshold layer.')
    p.add_argument('--disc_use_spectral_norm', action='store_true', help='Enables spectral normalization for the discriminator.')
    p.add_argument('--disc_use_threshold_norm', action='store_true', help='Enables threshold layer based on the vector norm.')
    p.add_argument('--disc_use_threshold_binarize', action='store_true', help='Enables threshold layer that set the norm of the output to either 0 or 1')
    p.add_argument('--disc_use_low_pass', action='store_true', help='Enables low pass median filter in the discriminator.')
    p.add_argument('--disc_with_r', action='store_true', help='Enables r coordinate in the discrminator coord_conv layers.')
    p.add_argument('--disc_freq_pooling', choices=['none', 'max', 'mean'], help='Frequency pooling type after the main convolutional block.')

    # Dataset arguments
    p.add_argument('--dataset_chunk_size_seconds', type=float, help='Chunk size of the input audio, in seconds. For example 1.27, or 2.55.')
    p.add_argument('--dataset_chunk_mode', choices=['random', 'fixed', 'full'])
    p.add_argument('--dataset_multi_track', action='store_true')
    p.add_argument('--dataset_backend', choices=['sony', 'baseline'], help='Backend code to parse and extract the labels from the CSVs. Important for the multitrack.')
    p.add_argument('--dataset_ignore_pad_labels', action='store_true', help='Use this when the backend=baseline, and the mnodel is samplecnn')
    p.add_argument('--dataset_trim_wavs', type=int, help='Trim wavs to this value in seconds when loading. Use -1 to load full wavs.')
    p.add_argument('--dataset_root', nargs='+', type=str)
    p.add_argument('--dataset_list_train', nargs='+')
    p.add_argument('--dataset_root_valid')
    p.add_argument('--dataset_list_valid')
    p.add_argument('--dataset_root_eval', help='Root path for the evaluation dataset. See helper.md for examples.')
    p.add_argument('--dataset_list_eval', help='File with wav filenames for the evaluation dataset. See helper.md for examples.')


    # Bilevel
    p.add_argument('--threshold', type=int, default=40, help="the threshold for augmentation")
    p.add_argument('--k_ops', type=int, default=2, help="number of augmentation applied during training")
    p.add_argument('--bi_nums', type=int, default=20000, help='when to optimzie policies')
    p.add_argument('--bi_level', action='store_true',  help='when to optimzie policies')
    p.add_argument('--search_freq', type=float, default=10000, help='exploration frequency')  #
    p.add_argument('--n_proj_layer', type=int, default=0, help="number of hidden layer in augmentation policy projection")
    p.add_argument('--temperature', type=float, default=2.0, help="temperature")
    p.add_argument('--stage_num', type=int, default=2, help="number of stage during training")
    p.add_argument('--use_aug', action='store_true')

    #add
    # p.add_argument('--finetune_mode', action='store_true', help="是否微调")
    # p.add_argument('--model_str', type=str, default='seld', help="model str")
    # p.add_argument('--optimizer', type=str, default='Adam', help="优化器")
    # p.add_argument('--weight_decay', type=float, default=1.e-6, help="weight_decay")
    # p.add_argument('--scheduler', type=str, default='WarmupLR', help="学习率调整策略")
    # p.add_argument('--scheduler_conf', type=dict, default='', help="学习率参数")
    # p.add_argument('--num_epochs', type=int, default=100, help="总epoch数")
    # p.add_argument('--each_epoch_nums_iters', type=int, default=2000, help="设置每个epoch有多少个iter")
    p.add_argument('--finetune_mode', action='store_true', help="是否微调")
    p.add_argument('--kops_linear', action='store_true', help="kops是否linear")
    p.add_argument('--model_str', type=str, default='seld', help="model str")
    p.add_argument('--optimizer', type=str, default='Adam', help="优化器")
    p.add_argument('--weight_decay', type=float, default=1.e-6, help="weight_decay")
    p.add_argument('--split', type=float, default=0.4, help="train_data_rate")
    p.add_argument('--scheduler', type=str, default='WarmupLR', help="学习率调整策略")
    # p.add_argument('--scheduler_conf', type=dict,
    #                help="学习率参数作为空格分隔的键值对，如 key1 value1 key2 value2")
    p.add_argument('--num_epochs', type=int, default=100, help="总epoch数")
    p.add_argument('--each_epoch_nums_iters', type=int, default=2000, help="设置每个epoch有多少个iter")
    p.add_argument('--pretrained_model_dir')

    p.add_argument('--fix', action='store_true', help='	是否使用固定参数数据增强')
    # Evaluation arugments
    p.add_argument('--evaluation_overlap_fraction', type=int, help='Fraction for overlap when doing the evaluation. Should be in multiples of the labels. So if the labels are 128 frames, we could use 32 here, so the hopsize would be 4 frames.')

    params = p.parse_args()

    # Set fixed values that are not defined in the config files
    params.dataset_chunk_size = round(24000 * params.dataset_chunk_size_seconds)
    params = vars(params)

    #策略网络的层数
    # params["n_proj_layer"] = 1

    if '2020' in params['dataset_root_valid']:
        params['unique_classes'] = 14
        params['output_shape'][-2] = 14
    elif '2021' in params['dataset_root_valid']:
        params['unique_classes'] = 12
        params['output_shape'][-2] = 12
    elif '2022' in params['dataset_root_valid']:
        params['unique_classes'] = 13
        params['output_shape'][-2] = 13
    elif '2023' in params['dataset_root_valid']:
        params['unique_classes'] = 13
        params['output_shape'][-2] = 13

    if 'debug' in params['exp_name']:
        params['experiment_description'] = f'{params["exp_name"]}'
    else:
        # params['experiment_description'] = f'{params["exp_group"]}-{params["exp_name"]}-{params["job_id"]}_{params["task_id"]}__' \
        #                                    f'n-work:{params["num_workers"]}_' \
        #                                    f'{params["model"]}_' \
        #                                    f'curr:{params["curriculum_scheduler"]}_' \
        #                                    f'{params["model_normalization"]}_' \
        #                                    f'{params["dataset_chunk_size"]}_' \
        #                                    f'_{datetime.now().strftime("%Y-%m-%d-%H%M%S")}'
        params['experiment_description'] = f'{params["model"]}_' \
                                           f'{params["model_loss_fn"]}_'\
                                           f'bi_level:{params["bi_level"]}_'\
                                           f'k_ops:{params["k_ops"]}_' \
                                           f'kops_linear:{params["kops_linear"]}_' \
                                           f'n_proj_layer:{params["n_proj_layer"]}_' \
                                           f'{params["dataset_chunk_size_seconds"]}_' \
                                           f'sf:{params["search_freq"]}_' \
                                           f'_{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}'
    params['logging_dir'] = f'{params["logging_dir"]}/{params["experiment_description"]}'

    if params["finetune_mode"] == True:
        params['logging_dir'] = params["pretrained_model_dir"]

    params['directory_output_results'] = f'{params["logging_dir"]}/tmp_results'

    # Save config to disk, create directories if needed
    if 'debug' in params['logging_dir'] and os.path.exists(params['logging_dir']):
        shutil.rmtree(params['logging_dir'])
    if not os.path.exists(params['logging_dir']):
        os.mkdir(params['logging_dir'])
    with open(os.path.join(params['logging_dir'], 'params.yaml'), 'w') as f:
        yaml.dump(params, f, default_flow_style=None)
    if not os.path.exists(params['directory_output_results']):
        os.mkdir(params['directory_output_results'])


    if "seld" in params["model_str"]:
        params["f_pool_size"]= [8, 4]
        params["dropout_rate"]= 0.05  # Dropout rate, constant for all layers
        params["nb_cnn2d_filt"]= 64
        params["rnn_size"]= 128
        params["nb_fnn_layers"]= 1
        params["nb_heads"]= 8
        params["nb_self_attn_layers"]= 2
        params["nb_rnn_layers"]=2
        params["fnn_size"]= 128
        # params["t_pool_size"] = [1,1,1]

    params["warmup_steps"] = 900
    params["min_lr"] = 1e-6

    if params['wandb']:
        import wandb

        wandb_config = {
            "model": params['model'],
            "model_spatialmixup": params['model_spatialmixup'],
            "model_augmentation": params['model_augmentation'],
            "model_rotations": params['model_rotations'],
            "model_features_transform": params['model_features_transform'],
            "use_mixup": params['use_mixup'],
            "model_loss_fn": params['model_loss_fn'],
            "job_id": f"{params['job_id']}_{params['task_id']}",
            "curriculum": params['curriculum_scheduler'],
            "num_workers": params['num_workers'],
            "logging_dir": params["logging_dir"],
            "dataset": params["dataset_list_train"],
            "dataset_backend": params["dataset_backend"]
        }
        wandb.init(project='seld_dcase2022_ric',
                   name=params['exp_name'],
                   tags=['debug' if params['debug'] else 'exp',
                         params['model'],
                         'augmented' if params['model_augmentation'] else 'non-augmented',
                         'rot' if params['model_rotations'] else 'non-rot'],
                   group=params['exp_group'] if (params['exp_group'] is not None or params['exp_group'] != '') else None,
                   config=wandb_config,
                   dir=params["logging_dir"],)
                   #sync_tensorboard=True)
        wandb.tensorboard.patch(save=False)

    print("")
    print("================ Experiment ================")
    print(params['experiment_description'])
    print("")

    # Print the experiment config
    ctr = 0
    for k, v in params.items():
        ctr += 1
        if ctr % 10 == 0: print(' ')
        print('{} \t {}'.format(k.ljust(15, ' '), v))
    print("")

    return EasyDict(params)


# Test
if __name__ == '__main__':
    # Preparation
    config = get_parameters()
    print(config)