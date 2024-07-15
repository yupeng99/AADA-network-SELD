import json
import warnings
import copy
import torch
import torch.nn as nn

import torch.profiler
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import higher
import os, shutil, math

import time
import wandb

from itertools import islice
from typing import List
from optimizer.scheduler import WarmupLR
from dataset.dcase_dataset import DCASE_SELD_Dataset, InfiniteDataLoader, _get_padders
from evaluation.dcase2022_metrics import cls_compute_seld_results
from evaluation.evaluation_dcase2022 import write_output_format_file, get_accdoa_labels, get_multi_accdoa_labels, determine_similar_location, all_seld_eval
from solver import SolverBasic, SolverDAN
from feature import Feature_StftPlusIV, Feature_MelPlusPhase, Feature_MelPlusIV,Feature_MelPlusIV_Base,Feature_StftPlusIV_Base
from models.projection import Projection
from models.adaptive_augmentor1 import MDAAug
from models.seldnet_model import MSELoss_ADPIT
import augmentation.spatial_mixup as spm
import torch_audiomentations as t_aug
from parameters import get_parameters
import analysis
import utils
import plots

from logger_utils.logger import setup_logger
logger = setup_logger(__name__)


def get_dataset(config):
    dataloader_train = None

    datasets_train = []
    for dset_root, dset_list in zip(config.dataset_root, config.dataset_list_train):
        dataset_tmp = DCASE_SELD_Dataset(directory_root=dset_root,
                                         list_dataset=dset_list,
                                         chunk_size=config.dataset_chunk_size,
                                         chunk_mode=config.dataset_chunk_mode,  # random
                                         trim_wavs=config.dataset_trim_wavs,
                                         multi_track=config.dataset_multi_track,
                                         num_classes=config.unique_classes,
                                         labels_backend=config.dataset_backend,  # sony
                                         pad_labels=not config.dataset_ignore_pad_labels,
                                         return_fname=False)
        datasets_train.append(dataset_tmp)
    # dataset_train = torch.utils.data.ConcatDataset(datasets_train)



    # if config.bi_level:
    #     full_dataset = torch.utils.data.ConcatDataset(datasets_train)
    #     # sss = StratifiedShuffleSplit(n_splits=1, test_size=len(search_dataset)/2, random_state=0)  # 1000 + 1000 trainset
    #     # sss = sss.split(list(range(len(search_dataset))), search_dataset.labels)
    #     # train_idx, search_idx = next(sss)
    #     # targets = [search_dataset.labels[idx] for idx in train_idx]
    #     # total_trainset = Subset(search_dataset, train_idx)
    #     # total_trainset.labels = targets
    #     # total_trainset.targets = targets
    #     # targets = [search_dataset.labels[idx] for idx in search_idx]
    #     # search_dataset = Subset(search_dataset, search_idx)
    #     # search_dataset.labels = targets
    #     # search_dataset.targets = targets
    #     print(f"len(full_dataset):{len(full_dataset)}")
    #     train_size = int(config.split * len(full_dataset))
    #     search_size = len(full_dataset) - train_size
    #     train_dataset, search_dataset = torch.utils.data.random_split(full_dataset, [train_size, search_size])
    #
    #     dataloader_full = InfiniteDataLoader(full_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
    #                                          shuffle=True, drop_last=True, pin_memory=False)
    #     dataloader_train = InfiniteDataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
    #                                       shuffle=True, drop_last=True, pin_memory=False)
    #     dataloader_search = InfiniteDataLoader(search_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
    #                                       shuffle=True, drop_last=True, pin_memory=False)
    #     print(f"dataloader_full:{len(dataloader_full)}")
    #     print(f"dataloader_train:{len(dataloader_train)}")
    #     print(f"dataloader_search:{len(dataloader_search)}")
    # else:
    if len(config.dataset_root) == 2:
        full_dataset = torch.utils.data.ConcatDataset(datasets_train)
        print(f"len(full_dataset):{len(full_dataset)}")
        train_size = int(config.split * len(full_dataset))
        end_size = len(full_dataset) - train_size
        train_dataset = torch.utils.data.random_split(full_dataset, [train_size, train_size,train_size,train_size,end_size])
        dataloader_train[len(train_dataset)] =None
        for i in range(0,len(train_dataset)):
            dataloader_train = InfiniteDataLoader(train_dataset[i], batch_size=config.batch_size, num_workers=config.num_workers,
                                                 shuffle=True, drop_last=True, pin_memory=False)
        print(f"dataloader_train:{len(dataloader_train)}")
    else:
        dataset_train = torch.utils.data.ConcatDataset(datasets_train)
        dataloader_train = InfiniteDataLoader(dataset_train, batch_size=config.batch_size,
                                                  num_workers=config.num_workers,
                                                  shuffle=True, drop_last=True, pin_memory=False)

    dataset_valid = DCASE_SELD_Dataset(directory_root=config.dataset_root_valid,
                                       list_dataset=config.dataset_list_valid,
                                       chunk_size=config.dataset_chunk_size,
                                       chunk_mode='full',
                                       trim_wavs=config.dataset_trim_wavs,
                                       multi_track=config.dataset_multi_track,
                                       num_classes=config.unique_classes,
                                       labels_backend=config.dataset_backend,
                                       pad_labels=not config.dataset_ignore_pad_labels,
                                       return_fname=True)

    # if config.bi_level:
    #     return dataloader_train, dataloader_search, dataloader_full, dataset_valid
    # else:
    return dataloader_train, dataset_valid

def get_spatial_mixup(device='cpu', p_comp=1.0):
    params = {'t_design_degree': 20,
              'G_type': 'identity',
              'use_slepian': False,
              'order_output': 1,
              'order_input': 1,
              'backend': 'basic',
              'w_pattern': 'hypercardioid'}

    transform = spm.DirectionalLoudness(t_design_degree=params['t_design_degree'],
                                        G_type=params['G_type'],
                                        use_slepian=params['use_slepian'],
                                        order_output=params['order_output'],
                                        order_input=params['order_input'],
                                        backend=params['backend'],
                                        w_pattern=params['w_pattern'],
                                        device=device,
                                        p_comp=p_comp)

    return transform

def get_rotations(device='cpu', p_comp=1.0):
    params = {'t_design_degree': 20,
              'G_type': 'identity',
              'use_slepian': False,
              'order_output': 1,
              'order_input': 1,
              'backend': 'basic',
              'w_pattern': 'hypercardioid'}

    rotation_params = {'rot_phi': 0.0,
                       'rot_theta': 0.0,
                       'rot_psi': 0.0}
    rotation_angles = [rotation_params['rot_phi'], rotation_params['rot_theta'], rotation_params['rot_psi']]

    rotation = spm.SphericalRotation(rotation_angles_rad=rotation_angles,
                                     t_design_degree=params['t_design_degree'],
                                     order_output=params['order_output'],
                                     order_input=params['order_input'],
                                     device=device,
                                     p_comp=p_comp)

    return rotation

def get_rotations_noise(device='cpu', p_comp=1.0):
    params = {'t_design_degree': 20,
              'G_type': 'identity',
              'use_slepian': False,
              'order_output': 1,
              'order_input': 1,
              'backend': 'basic',
              'w_pattern': 'hypercardioid'}

    rotation = spm.SphericalRotation(rotation_angles_rad=[0,0,0],
                                     t_design_degree=params['t_design_degree'],
                                     order_output=params['order_output'],
                                     order_input=params['order_input'],
                                     ignore_labels=True,
                                     device=device, p_comp=p_comp)

    return rotation

def get_audiomentations(p=0.5, fs=24000):
    from augmentation.spliceout import SpliceOut
    from augmentation.MyBandStopFilter import BandStopFilter
    from augmentation.MyBandPassFilter import BandPassFilter
    # Initialize augmentation callable
    apply_augmentation = t_aug.Compose(
        transforms=[
            t_aug.Gain(p=p, min_gain_in_db=-15.0, max_gain_in_db=5.0, mode='per_example', p_mode='per_example'),
            t_aug.PolarityInversion(p=p, mode='per_example', p_mode='per_example'),
            t_aug.PitchShift(p=p, min_transpose_semitones=-1.5, max_transpose_semitones=1.5, sample_rate=fs, mode='per_example', p_mode='per_example'),
            t_aug.AddColoredNoise(p=p, min_snr_in_db=6.0, max_snr_in_db=30.0, min_f_decay=-2.0, max_f_decay=2.0, sample_rate=fs, mode='per_example', p_mode='per_example'),
            BandStopFilter(p=p, min_center_frequency=400, max_center_frequency=4000, min_bandwidth_fraction=0.25, max_bandwidth_fraction=1.99, sample_rate=fs, p_mode='per_example'),
            t_aug.LowPassFilter(p=p,  min_cutoff_freq=1000, max_cutoff_freq=7500, sample_rate=fs, p_mode='per_example'),
            t_aug.HighPassFilter(p=p, min_cutoff_freq=100, max_cutoff_freq=2000, sample_rate=fs, p_mode='per_example'),
            BandPassFilter(p=p, min_center_frequency=400, max_center_frequency=4000, min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.99, sample_rate=fs, p_mode='per_example'),
            #SpliceOut(p=p, num_time_intervals=8, max_width=400, sample_rate=fs, p_mode='per_example')
        ]
    )

    return apply_augmentation

class RandomAugmentations(nn.Sequential):
    def __init__(self, fs=24000, p=1, p_comp=1, n_aug_min=2, n_aug_max=6, threshold_limiter=1):
        super().__init__()
        self.fs = fs
        self.p = p
        self.p_comp = p_comp
        self.n_aug_min = n_aug_min
        self.n_aug_max = n_aug_max
        self.threshold_limiter = threshold_limiter
        mode = 'per_example'  # for speed, we use batch processing
        p_mode = 'per_example'

        self.augmentations = t_aug.SomeOf((n_aug_min, n_aug_max), p=self.p_comp, output_type='dict',
                                      transforms=[
                                          t_aug.Gain(p=p, min_gain_in_db=-15.0, max_gain_in_db=6.0, mode=mode, p_mode=p_mode),
                                          t_aug.PolarityInversion(p=p, mode=mode, p_mode=p_mode),
                                          #t_aug.PitchShift(p=p, min_transpose_semitones=-1.5, max_transpose_semitones=1.5, sample_rate=fs,
                                          #                 mode=mode, p_mode=p_mode),
                                          t_aug.AddColoredNoise(p=p, min_snr_in_db=2.0, max_snr_in_db=30.0, min_f_decay=-2.0, max_f_decay=2.0,
                                                                sample_rate=fs, mode=mode, p_mode=p_mode),
                                          t_aug.BandStopFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
                                                               min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.1, sample_rate=fs,
                                                               p_mode=p_mode),
                                          t_aug.LowPassFilter(p=p, min_cutoff_freq=1000, max_cutoff_freq=5000, sample_rate=fs,
                                                              p_mode=p_mode),
                                          t_aug.HighPassFilter(p=p, min_cutoff_freq=250, max_cutoff_freq=1500, sample_rate=fs,
                                                               p_mode=p_mode),
                                          t_aug.BandPassFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
                                                               min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.5, sample_rate=fs,
                                                               p_mode=p_mode),
                                          #t_aug.SpliceOut(p=p, num_time_intervals=100, max_width=100, sample_rate=fs, p_mode=p_mode)
                                      ]
                                      )

    def forward(self, input):
        do_reshape = False
        if input.shape == 2:
            do_reshape = True
            input = input[None, ...]  #  audiomentations expects batches

        # Augmentations
        output = self.augmentations(input, sample_rate=self.fs)  # Returns ObjectDict
        output = output['samples']

        # Limiter
        torch.clamp(output, min=-self.threshold_limiter, max=self.threshold_limiter)

        if do_reshape:
            output = output.squeeze(0)
        return output



class RandomSpecAugmentations(nn.Sequential):
    def __init__(self, fs=24000, p=1, p_comp=1, n_aug_min=1, n_aug_max=2):
        super().__init__()
        self.fs = fs
        self.p = p
        self.p_comp = p_comp
        self.n_aug_min = n_aug_min
        self.n_aug_max = n_aug_max
        mode = 'per_example'  # for speed, we use batch processing
        p_mode = 'per_example'

        self.augmentations = t_aug.SomeOf((n_aug_min, n_aug_max), p=self.p_comp, output_type='dict',
                                      transforms=[
                                          t_aug.SpecTimeMasking(time_mask_param=24, iid_masks=True, p_proportion=0.3, p=p,
                                                                mode=mode, p_mode=p_mode),
                                          t_aug.SpecFreqMasking(freq_mask_param=24, iid_masks=True, p=p,
                                                                mode=mode, p_mode=p_mode),
                                      ]
                                      )

    def forward(self, input):
        do_reshape = False
        if input.shape == 2:
            do_reshape = True
            input = input[None, ...]  #  audiomentations expects batches

        # Augmentations
        output = self.augmentations(input)  # Returns ObjectDict
        output = output['samples']

        if do_reshape:
            output = output.squeeze(0)
        return output

class CustomFilter(nn.Sequential):
    def __init__(self, fs=24000, p=1, p_comp=1):
        super().__init__()
        self.fs = fs
        self.p = p
        self.p_comp = p_comp
        mode = 'per_batch'  # for speed, we use batch processing
        p_mode = 'per_batch'
        self.augmentations = t_aug.Compose(output_type='tensor',
                                          transforms=[
                                              t_aug.LowPassFilter(p=p, min_cutoff_freq=5000, max_cutoff_freq=5001, sample_rate=fs,
                                                                  p_mode=p_mode),
                                              t_aug.HighPassFilter(p=p, min_cutoff_freq=125, max_cutoff_freq=126, sample_rate=fs,
                                                                   p_mode=p_mode),
                                          ]
                                          )

    def forward(self, input):
        do_reshape = False
        if input.shape == 2:
            do_reshape = True
            input = input[None, ...]  # audiomentations expects batches

        # Augmentations
        output = self.augmentations(input)

        if do_reshape:
            output = output.squeeze(0)
        return output

#加载模型
def load_pretrained(model,optimizer,config,is_best = True,is_policy = False): # 没改完
    if not is_policy :
        model_dir = config["pretrained_model_dir"]
        if os.path.isdir(model_dir):
            if is_best:
                model_files = os.path.join(model_dir,"task_model","best_model")
            else :
                model_files = os.path.join(model_dir, "task_model", "last_model")
            model_file = os.path.join(model_files,config["model"]+"_model.h5")
            model_state = os.path.join(model_files,"model.state")
            model_optimizer = os.path.join(model_files,"optimizer.pt")
        assert os.path.exists(model_file), f"{model_file} 模型不存在！"
        model_dict = model.state_dict()
        model_state_dict = torch.load(model_file)
        # 特征层
        for name, weight in model_dict.items():
            if name in model_state_dict.keys():
                if list(weight.shape) != list(model_state_dict[name].shape):
                    logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                   format(name, list(model_state_dict[name].shape), list(weight.shape)))
                    model_state_dict.pop(name, None)
            else:
                logger.warning('Lack weight: {}'.format(name))
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(torch.load(model_optimizer))
        with open(model_state,'r',encoding='utf-8') as f:
            json_data = json.load(f)
            last_epoch = json_data['last_iter']
            dcase_output_val_folder = json_data['dcase_output_val_folder']
        logger.info(f' 成功加载预训练模型：{model_file}')

        return model,optimizer,last_epoch,dcase_output_val_folder
    else:
        model_dir = config["pretrained_model_dir"]
        if os.path.isdir(model_dir):
            model_files = os.path.join(model_dir, "policy_model")
            model_file = os.path.join(model_files, "project" + "_model.h5")
            model_optimizer = os.path.join(model_files, "optimizer.pt")
        assert os.path.exists(model_file), f"{model_file} 模型不存在！"
        model_dict = model.state_dict()
        model_state_dict = torch.load(model_file)
        # 特征层
        for name, weight in model_dict.items():
            if name in model_state_dict.keys():
                if list(weight.shape) != list(model_state_dict[name].shape):
                    logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                   format(name, list(model_state_dict[name].shape), list(weight.shape)))
                    model_state_dict.pop(name, None)
            else:
                logger.warning('Lack weight: {}'.format(name))
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(torch.load(model_optimizer))
        logger.info(f' 成功加载预训练模型：{model_file}')
        return model,optimizer

#保存模型   还是得加这个 dcase_output_val_folder
def save_checkpoint(model,optimizer, save_model_name,iter_id,metric_score = None,save_model_path="./save/models", dcase_output_val_folder = None,best_model=True,is_policy = False,policy_name = "projection_model"):
    state_dict = model.state_dict()
    if not is_policy:
        if best_model:
            model_path = os.path.join(save_model_path, "task_model", 'best_model')
        else:
            model_path = os.path.join(save_model_path,"task_model", 'iter_{}'.format(iter_id))
    else:
        model_path = os.path.join(save_model_path, "policy_model")
    os.makedirs(model_path, exist_ok=True)
    torch.save(optimizer.state_dict(), os.path.join(model_path, 'optimizer.pt'))
    torch.save(state_dict, os.path.join(model_path, save_model_name+'_model.h5'))

    if not is_policy:
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            data = {"last_iter": iter_id,"best_iter":metric_score[0],
                    "train_loss":metric_score[1],"val_loss":metric_score[2],
                    "ER":metric_score[3],"F":metric_score[4],"LE":metric_score[5],"LR":metric_score[6],"seld_scr":metric_score[7],"dcase_output_val_folder":dcase_output_val_folder}
            f.write(json.dumps(data))
        if not best_model:
            last_model_path = os.path.join(save_model_path, save_model_name, 'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path, save_model_name, 'epoch_{}'.format(iter_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
    logger.info('已保存模型：{}\t第{}次迭代'.format(model_path,iter_id))

def main():
    # Get config
    config = get_parameters()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Reproducibility
    utils.seed_everything(seed=config.seed, mode=config.seed_mode)  # 设置随机种子

    # Logging configuration
    writer = SummaryWriter(config.logging_dir)

    # Data
    # if config.bi_level:
    #     dataloader_train, dataloader_search, dataloader_full, dataset_valid = get_dataset(config)
    # else:
    dataloader_train, dataset_valid = get_dataset(config)




    # Solver
    if config.solver == 'vanilla':
        solver = SolverBasic(config=config, tensorboard_writer=writer)
    elif config.solver == 'DAN':
        solver = SolverDAN(config=config, tensorboard_writer=writer)
    else:
        raise ValueError(f'Solver {config.solver} not supported.')

        # model_setting
    gf_model = solver.predictor
    if config.model == "crnn10":
        h_model = Projection(in_features=gf_model.xyz_fc.in_features * 32,  ## 5.11 用32   2.55 用 16
                             n_layers=config.n_proj_layer, n_hidden=128).cuda()
    elif config.model == "baseline":
        h_model = Projection(in_features=gf_model.fnn_list[-1].in_features *501 ,  ##  * 的后面是step
                             n_layers=config.n_proj_layer, n_hidden=128).cuda()   ## gf_model.fnn_list[-1].in_features
    elif config.model.startswith("resnet-conformer"):
        h_model = Projection(in_features=gf_model.linear_out.in_features * 501,  ##  * 的后面是step
                             n_layers=config.n_proj_layer, n_hidden=128).cuda()
    h_optimizer = torch.optim.Adam(   #
        h_model.parameters(),
        lr=1e-2,
        betas=(0.9, 0.999),
        weight_decay=1e-3)
    h_scheduler = WarmupLR(optimizer=h_optimizer, warmup_steps=4000,min_lr=1e-7)


    last_epoch = 0
    if config.finetune_mode:
        gf_model, solver.optimizer_predictor, last_epoch, dcase_output_val_folder = load_pretrained(gf_model,
                                                                                                    solver.optimizer_predictor,
                                                                                                      config)
        if config.bi_level:
            h_model, h_optimizer = load_pretrained(h_model, h_optimizer, config, is_policy=True)

    mdaaug_config = {'sampling': 'prob',
                     'k_ops': config.k_ops,
                     'delta': 0.3,
                     'temp': config.temperature,  # magnitudes 参数 在explore 和 exploit 进行softmaxt的区别
                     # 'search_d': get_dataset_dimension(args.dataset),
                     # 'target_d': get_dataset_dimension(args.dataset)
                     }


    # Select features and augmentation and rotation
    augmentation_transform_spatial = None
    augmentation_transform_audio = None
    augmentation_transform_spec = None
    rotations_transform = None
    rotations_noise = None

    if config.model_features_transform == 'stft_iv':
        features_transform = Feature_StftPlusIV(nfft=512).to(device)  # mag STFT with intensity vectors
    elif config.model_features_transform == 'stft_iv_base':
        features_transform = Feature_StftPlusIV_Base(nfft=512).to(device)  # mag STFT with intensity vectors
    elif config.model_features_transform == 'mel_iv':
        features_transform = Feature_MelPlusIV().to(device)  # mel spec with intensity vectors
    elif config.model_features_transform == 'mel_iv_base':
        features_transform = Feature_MelPlusIV_Base().to(device)
    elif config.model_features_transform == 'mel_phase':
        features_transform = Feature_MelPlusPhase().to(device)  # mel spec with phase difference
    elif config.model_features_transform == 'bandpass':
        features_transform = CustomFilter().to(device)  # Custom Band pass filter to accomodate for the Eigenmike
    else:
        features_transform = None
    print(features_transform)

    if config.model_spatialmixup:
        augmentation_transform_spatial = get_spatial_mixup(device=device, p_comp=0.0).to(device)
    if config.model_augmentation:
        augmentation_transform_audio = RandomAugmentations(p_comp=0.0).to(device)
    if config.model_spec_augmentation:
        augmentation_transform_spec = RandomSpecAugmentations(p_comp=0.0).to(device)
    if config.model_rotations:
        rotations_transform = get_rotations(device=device, p_comp=0.0).to(device)
    if config.model_rotations_noise:
        rotations_noise = get_rotations_noise(device=device, p_comp=0.0).to(device)

    mdaaug = MDAAug(n_class=config.unique_classes,
                    gf_model=gf_model,
                    h_model=h_model,
                    sys_config=config,
                    iters=last_epoch,
                    features_transform = features_transform,
                    save_dir='./save_op',
                    config=mdaaug_config)

    if config['dataset_multi_track']:
        criterion = MSELoss_ADPIT()
    elif config.model_loss_fn == 'mse':
        criterion = torch.nn.MSELoss()
    elif config.model_loss_fn == 'bce':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config.model_loss_fn == 'l1':
        criterion = torch.nn.L1Loss()


    if 'samplecnn' in config.model:
        class t_transform(nn.Sequential):
            def __int__(self):
                super().__init__()
            def forward(self, input):
                out = nn.functional.interpolate(input, scale_factor=(1, 0.1), mode='nearest-exact') # 使用最近邻插值方法对数据进行插值操作
                return out
        target_transform = t_transform()
    else:
        target_transform = None
    print(target_transform)
    print(rotations_transform)
    print(augmentation_transform_spatial)
    print(augmentation_transform_audio)
    print(augmentation_transform_spec)

    # Initial loss:
    x, target = dataloader_train.dataset[0]
    if features_transform is not None:
        x = features_transform(x.unsqueeze(0).to(device))
    else:
        x = x[None, ...].to(device)
    if target_transform is not None:
        target = target_transform(target[None, ...].to(device))
    else:
        target = target[None, ...].to(device)
    solver.predictor.eval()
    print(f"x.shape:{x.shape}")
    print(f"target.shape:{target.shape}")
    # To debug
    #yolo = RandomSpecAugmentations()
    #y = yolo(x)

    out = solver.predictor(x)
    #loss = solver.loss_fns[solver.loss_names['G_rec']](out, target)
    loss = solver.loss_fns['G_rec' if config.solver == 'DAN' else 'G_rec'](out, target)
    print('Initial loss = {:.6f}'.format(loss.item()))

    # Profiling
    if config.profiling:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.logging_dir),
            profile_memory=False,
            record_shapes=True,
            with_stack=False)
        prof.start()

    # Monitoring variables
    train_loss, val_loss, seld_metrics_macro, seld_metrics_micro = 0, 0, None, None
    best_val_step_macro, best_val_loss, best_metrics_macro = 0, 0, [0, 0, 0, 0, 99]
    best_val_step_micro, best_val_loss_micro, best_metrics_micro = 0, 0, [0, 0, 0, 0, 99]
    start_time = time.time()


    seld_metrics_macro, seld_metrics_micro, val_loss, _ = validation_iteration(config, dataset=dataset_valid, iter_idx=0,
                                                                       device=device, features_transform=features_transform,
                                                                       target_transform=target_transform, solver=solver, writer=None,
                                                                       dcase_output_folder=config['directory_output_results'],
                                                                       detection_threshold=config['detection_threshold'])
    print(f'Evaluating using overlap = 1 / {config["evaluation_overlap_fraction"]}')
    print(

        'train_loss: {:0.4f}, val_loss: {:0.4f}, '
        'p_comp: {:0.3f}, '.format(
                                    train_loss, val_loss,
                                   solver.get_curriculum_params()[0]))
    print('====== micro ======')
    print(
        'micro: ER/F/LE/LR/SELD: {}, '.format(
                                              '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/\t/{:0.4f}'.format(
                                                  *seld_metrics_micro[0:5]), ))
    print('====== MACRO ======')
    print(
        'MACRO: ER/F/LE/LR/SELD: {}, '.format(
                                              '{:0.4f}/{:0.4f}/{:0.4f}/{:0.4f}/\t/{:0.4f}'.format(
                                                  *seld_metrics_macro[0:5]), ))

    print('\n MACRO Classwise results on validation data')
    print('Class\tER\t\tF\t\tLE\t\tLR\t\tSELD_score')
    seld_metrics_class_wise = seld_metrics_macro[5]
    for cls_cnt in range(config['unique_classes']):
        print('{}\t\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}\t{:0.2f}'.format(cls_cnt,
                                                                         seld_metrics_class_wise[0][cls_cnt],
                                                                         seld_metrics_class_wise[1][cls_cnt],
                                                                         seld_metrics_class_wise[2][cls_cnt],
                                                                         seld_metrics_class_wise[3][cls_cnt],
                                                                         seld_metrics_class_wise[4][cls_cnt]))
    print('================================================ \n')


def validation_iteration(config, dataset, iter_idx, solver, features_transform, target_transform: [nn.Sequential, None], dcase_output_folder, device, writer, detection_threshold=0.5):
    # Adapted from the official baseline
    nb_test_batches, test_loss = 0, 0.
    model = solver.predictor
    model.eval()
    file_cnt = 0
    overlap = 1 / config['evaluation_overlap_fraction']  # defualt should be 1  TODO, onluy works for up to 1/32 , when the labels are 128 frames long.

    outputs_for_plots = []
    print(f'Validation: {len(dataset)} fnames in dataset.')
    with torch.no_grad():
        for ctr, (audio, target, fname) in enumerate(dataset):
            # load one batch of data
            audio, target = audio.to(device), target.to(device)
            duration = dataset.durations[fname]
            print(f'Evaluating file {ctr+1}/{len(dataset)}: {fname}')
            print(f'Audio shape: {audio.shape}')
            print(f'Target shape: {target.shape}')

            warnings.warn('WARNING: Hard coded chunk size for evaluation')
            audio_padding, labels_padding = _get_padders(chunk_size_seconds=config.dataset_chunk_size / dataset._fs[fname],
                                                         duration_seconds=math.floor(duration),
                                                         overlap=overlap,
                                                         audio_fs=dataset._fs[fname],
                                                         labels_fs=100)

            # Split each wav into chunks and process them
            audio = audio_padding['padder'](audio)
            audio_chunks = audio.unfold(dimension=1, size=audio_padding['chunk_size'], step=audio_padding['hop_size']).permute((1, 0, 2))

            if config.dataset_multi_track:
                labels = labels_padding['padder'](target.permute(1,2,3,0))
                labels_chunks = labels.unfold(dimension=-1, size=labels_padding['chunk_size'], step=labels_padding['hop_size'])
                labels_chunks = labels_chunks.permute((3, 4, 0, 1, 2))
            else:
                labels = labels_padding['padder'](target)
                labels_chunks = labels.unfold(dimension=-1, size=labels_padding['chunk_size'], step=labels_padding['hop_size'])
                labels_chunks = labels_chunks.permute((2, 0, 1, 3))

            full_output = []
            full_loss = []
            full_labels = []
            if audio_chunks.shape[0] != labels_chunks.shape[0]:
                a = 1
                warnings.warn('WARNING: Possible error in padding.')
            if audio_chunks.shape[0] > labels_chunks.shape[0]:
                audio_chunks = audio_chunks[0:labels_chunks.shape[0], ...]  # Mmm... lets drop the extra audio chunk if there are no labels for it
            if audio_chunks.shape[0] < labels_chunks.shape[0]:
                audio_chunks = torch.concat([audio_chunks, torch.zeros_like(audio_chunks[0:1])])  # Mmm... lets add an empty audio slice
            tmp = torch.utils.data.TensorDataset(audio_chunks, labels_chunks)
            loader = DataLoader(tmp, batch_size=1, shuffle=False, drop_last=False)  # Loader per wav to get batches
            for ctr, (audio, labels) in enumerate(loader):
                if features_transform is not None:
                    audio = features_transform(audio)
                if target_transform is not None:
                    labels = target_transform(labels)
                output = model(audio)
                if config.oracle_mode:
                    output = torch.zeros_like(labels)  # TODO This is just to get the upper bound of the loss
                    if config.dataset_multi_track:
                        output = torch.zeros(size=(labels.shape[0], labels.shape[1], 3*3*12), device=device)  # TODO This is just to get the upper bound of the loss wih mACCDOA
                loss = solver.loss_fns[solver.loss_names[0]](output, labels)
                full_output.append(output)
                full_loss.append(loss)
                if config.oracle_mode:
                    full_labels.append(labels)  # TODO This is just to get the upper bound of the loss
                if torch.isnan(loss):
                    raise ValueError('ERROR: NaNs in loss')

            # Concatenate chunks across timesteps into final predictions
            if config.dataset_multi_track:
                output = torch.concat(full_output, dim=-2)
            else:
                if overlap == 1:
                    output = torch.concat(full_output, dim=-1)
                    if config.oracle_mode:
                        output = torch.concat(full_labels, dim=-1)   # TODO This is just to get the upper bound of the loss
                else:
                    # TODO: maybe this is ready now? at least until overlap 1/32
                    # TODO: No, it only works when validating the ground truth labels, but not the final predictions
                    # Rebuild when using overlap
                    # This is basically a folding operation, using an average of the predictions of each overlapped chunk
                    aa = len(full_output) - 1
                    if config.oracle_mode:
                        full_output = full_labels # TODO This is just to get the upper bound of the loss
                    resulton = torch.zeros(aa, labels.shape[-3], labels.shape[-2], labels_padding['full_size'] + labels_padding['padder'].padding[-3])
                    resulton = torch.zeros(aa, labels.shape[-3], labels.shape[-2],
                                           labels_padding['full_size'] + labels_padding['padder'].padding[-3] + labels_padding['hop_size'])
                    weights = torch.zeros(1, labels_padding['full_size'] + labels_padding['padder'].padding[-3] + labels_padding['hop_size'])
                    for ii in range(0, aa):
                        #print(ii)
                        start_i = ii * labels_padding['hop_size']
                        end_i = start_i + round(labels_padding['hop_size'] * 1/1)
                        end_i = start_i + round(labels_padding['chunk_size'] * 1 / 1)
                        if end_i > resulton.shape[-1]:  # Drop the last part
                            end_i = resulton.shape[-1]
                        #yolingon = full_output[ii][0]
                        try:
                            resulton[ii, :, :, start_i:end_i] = full_output[ii][0,..., 0:end_i-start_i]
                        except:
                            a = 1
                            warnings.warn('WARNING: Error while evaluating with overlap')
                        weights[:, start_i:end_i] = weights[:, start_i:end_i] + 1

                    output = torch.sum(resulton, dim=0, keepdim=True) / weights
                    if torch.any(torch.isnan(output)):
                        warnings.warn('WARNING: NaNs detected in output')

            # Apply detection threshold based on vector norm
            if config.dataset_multi_track:
                pass
            else:
                norms = torch.linalg.vector_norm(output, ord=2, dim=-3, keepdims=True)
                norms = (norms < detection_threshold).repeat(1, output.shape[-3], 1, 1)
                output[norms] = 0.0
            loss = torch.tensor([x for x in full_loss]).mean()
            outputs_for_plots.append(output)

            # Useful fo debug:
            #output.detach().cpu().numpy()[0, 0]
            #plots.plot_labels(labels.detach().cpu().numpy()[0])
            #target.detach().cpu().numpy()[0]

            # Downsample over frames:
            if config.dataset_multi_track:
                if target_transform is None:
                    output = nn.functional.interpolate(output.permute(0, 2, 1), scale_factor=(0.1), mode='nearest-exact').permute(0, 2, 1)
            else:
                if target_transform is None:
                    output = nn.functional.interpolate(output, scale_factor=(1, 0.1), mode='nearest-exact')

                # I think the baseline code needs this in [batch, frames, classes*coords]
                output = output.permute([0, 3, 1, 2])
                output = output.flatten(2, 3)

            if config['dataset_multi_track'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(
                    output.detach().cpu().numpy(), config['unique_classes'])
                sed_pred0 = cls_compute_seld_results.reshape_3Dto2D(sed_pred0)
                doa_pred0 = cls_compute_seld_results.reshape_3Dto2D(doa_pred0)
                sed_pred1 = cls_compute_seld_results.reshape_3Dto2D(sed_pred1)
                doa_pred1 = cls_compute_seld_results.reshape_3Dto2D(doa_pred1)
                sed_pred2 = cls_compute_seld_results.reshape_3Dto2D(sed_pred2)
                doa_pred2 = cls_compute_seld_results.reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), config['unique_classes'])
                sed_pred = cls_compute_seld_results.reshape_3Dto2D(sed_pred)
                doa_pred = cls_compute_seld_results.reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            tmp_name = fname.split('/')[-1]
            output_file = os.path.join(dcase_output_folder, tmp_name.replace('.wav', '.csv'))
            file_cnt += 1
            output_dict = {}
            if config['dataset_multi_track'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                sed_pred1[frame_cnt][class_cnt],
                                                                doa_pred0[frame_cnt], doa_pred1[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                sed_pred2[frame_cnt][class_cnt],
                                                                doa_pred1[frame_cnt], doa_pred2[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                sed_pred0[frame_cnt][class_cnt],
                                                                doa_pred2[frame_cnt], doa_pred0[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                           doa_pred_fc[class_cnt + config['unique_classes']],
                                                           doa_pred_fc[class_cnt + 2 * config['unique_classes']]])
                output_dict_polar = {}
                for k, v in output_dict.items():
                    ss = []
                    for this_item in v:
                        tmp = utils.cart2sph(this_item[-3], this_item[-2], this_item[-1])
                        ss.append([this_item[0], *tmp])
                    output_dict_polar[k] = ss
                write_output_format_file(config,output_file, output_dict_polar, use_cartesian=False)
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            tmp_azi, tmp_ele = utils.cart2sph(doa_pred[frame_cnt][class_cnt],
                                                           doa_pred[frame_cnt][class_cnt + config['unique_classes']],
                                                           doa_pred[frame_cnt][class_cnt + 2 * config['unique_classes']])
                            output_dict[frame_cnt].append([class_cnt, tmp_azi, tmp_ele])
                            #output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                            #                               doa_pred[frame_cnt][
                            #                                   class_cnt + config['unique_classes']],
                            #                               doa_pred[frame_cnt][
                            #                                   class_cnt + 2 * config['unique_classes']]])
                write_output_format_file(config,output_file, output_dict, use_cartesian=False)

            test_loss += loss.item()
            nb_test_batches += 1

        test_loss /= nb_test_batches

    all_test_metric_macro, all_test_metric_micro = all_seld_eval(config, directory_root=dataset.directory_root, fnames=dataset._fnames, pred_directory=dcase_output_folder)

    if writer is not None:
        writer.add_scalar('Losses/valid', test_loss, iter_idx)
        writer.add_scalar('MMacro/ER', all_test_metric_macro[0], iter_idx)
        writer.add_scalar('MMacro/F', all_test_metric_macro[1], iter_idx)
        writer.add_scalar('MMacro/LE', all_test_metric_macro[2], iter_idx)
        writer.add_scalar('MMacro/LR', all_test_metric_macro[3], iter_idx)
        writer.add_scalar('MMacro/SELD', all_test_metric_macro[4], iter_idx)

        writer.add_scalar('Mmicro/ER', all_test_metric_micro[0], iter_idx)
        writer.add_scalar('Mmicro/F', all_test_metric_micro[1], iter_idx)
        writer.add_scalar('Mmicro/LE', all_test_metric_micro[2], iter_idx)
        writer.add_scalar('Mmicro/LR', all_test_metric_micro[3], iter_idx)
        writer.add_scalar('Mmicro/SELD', all_test_metric_micro[4], iter_idx)

    return all_test_metric_macro, all_test_metric_micro, test_loss, outputs_for_plots

def validation_iteration_base(config, dataset, iter_idx, solver, features_transform,criterion, target_transform: [nn.Sequential, None], dcase_output_folder, device, writer, detection_threshold=0.5):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames

    nb_test_batches, test_loss = 0, 0.
    model = solver.predictor
    model.eval()
    file_cnt = 0
    outputs_for_plots = []
    print(f'Validation: {len(dataset)} fnames in dataset.')
    with torch.no_grad():
        for ctr, (audio, target, fname) in enumerate(dataset):

            # load one batch of data
            audio, target = audio.to(device), target.to(device)
            duration = dataset.durations[fname]
            print(f'Evaluating file {ctr + 1}/{len(dataset)}: {fname}')
            print(f'Audio shape: {audio.shape}')
            print(f'Target shape: {target.shape}')




            # feature_batch_seq_len = int(np.ceil(dataset.max_frames / float(config.dataset_chunk_size))) * config.dataset_chunk_size
            # label_batch_seq_len = int(np.ceil(dataset.max_frames / float(config.dataset_chunk_size))) * 50
            # feat_extra_frames = feature_batch_seq_len - audio.shape[1]
            # extra_feat = np.ones((audio.shape[0],feat_extra_frames)) * 1e-6
            #
            # label_extra_frames = label_batch_seq_len - target.shape[0]
            # extra_labels = np.zeros((label_extra_frames, 6, 4, 13))
            #
            # for f_row in extra_feat:
            #     self._circ_buf_feat.append(f_row)
            # for l_row in extra_labels:
            #     self._circ_buf_label.append(l_row)

            # audio_chunks = audio[:,audio.shape[1]//]
            # print(f"audio1.shape:{audio.shape}")
            if config.dataset_multi_track == False:
                audio = audio[:, :audio.shape[1] - audio.shape[1] %2400]
                if (int(audio.shape[1] / 24000 * 10) < target.shape[-1]):
                    target = target[...,:int(audio.shape[1] / 2400)]

                if (int(audio.shape[1] / 24000 * 10) > target.shape[-1]):
                    audio = audio[:, :int(target.shape[-1] * 2400)]
            else :
                audio = audio[:, :audio.shape[1] - audio.shape[1] % 2400]
                if (int(audio.shape[1] / 24000 * 10) < target.shape[0]):
                    target = target[ :int(audio.shape[1] / 2400),...]

                if (int(audio.shape[1] / 24000 * 10) > target.shape[0]):
                    audio = audio[:, :int(target.shape[0] * 2400)]
            # print(f"audio1.shape:{audio.shape}")
            tmp = torch.utils.data.TensorDataset(torch.unsqueeze(audio,0), torch.unsqueeze(target,0))
            loader = DataLoader(tmp, batch_size=1, shuffle=False, drop_last=False)
            for ctr, (audio, labels) in enumerate(loader):
                # print(f"audio2.shape:{audio.shape}")
                if features_transform is not None:
                    audio = features_transform(audio)
                # print(f"audio3.shape:{audio.shape}")

                if config.model.startswith("conformer"):
                    data_lens = torch.tensor([audio.shape[-1] for i in range(audio.shape[0])]).to(device)
                    output = solver.predictor(audio, data_lens)
                else:
                    output = model(audio)  #（1，3，13，50）


                # print(f"output:{output.shape}")
                loss = criterion(output, labels)
                # x0 = output.detach().cpu().numpy()[:, :, :1 * 13]

                # print(f"x0:{x0}")


                if config['dataset_multi_track'] is True:
                    sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(
                        output.detach().cpu().numpy(), config['unique_classes'])
                    sed_pred0 = cls_compute_seld_results.reshape_3Dto2D(sed_pred0)
                    doa_pred0 = cls_compute_seld_results.reshape_3Dto2D(doa_pred0)
                    sed_pred1 = cls_compute_seld_results.reshape_3Dto2D(sed_pred1)
                    doa_pred1 = cls_compute_seld_results.reshape_3Dto2D(doa_pred1)
                    sed_pred2 = cls_compute_seld_results.reshape_3Dto2D(sed_pred2)
                    doa_pred2 = cls_compute_seld_results.reshape_3Dto2D(doa_pred2)
                else:
                    output = output.permute([0, 3, 1, 2])
                    # output = output.flatten(2, 3)
                    output = output.reshape(output.shape[0],output.shape[1],-1)
                    # print(f"output:{output}")
                    sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), config['unique_classes'])
                    sed_pred = cls_compute_seld_results.reshape_3Dto2D(sed_pred)
                    doa_pred = cls_compute_seld_results.reshape_3Dto2D(doa_pred)
                # indices = np.argwhere(sed_pred == True)
                # print(f"sed_pred0:{indices}")
                # print(f"sed_pred0:{sed_pred}")

                # print(f"doa_pred1:{doa_pred}")

                # dump SELD results to the correspondin file
                tmp_name = fname.split('/')[-1]
                output_file = os.path.join(dcase_output_folder, tmp_name.replace('.wav', '.csv'))
                file_cnt += 1
                output_dict = {}
                if config['dataset_multi_track'] is True:
                    for frame_cnt in range(sed_pred0.shape[0]):
                        for class_cnt in range(sed_pred0.shape[1]):
                            # determine whether track0 is similar to track1
                            flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                    sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt],
                                                                    doa_pred1[frame_cnt], class_cnt, config['thresh_unify'],
                                                                    config['unique_classes'])
                            flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                    sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt],
                                                                    doa_pred2[frame_cnt], class_cnt, config['thresh_unify'],
                                                                    config['unique_classes'])
                            flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                    sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt],
                                                                    doa_pred0[frame_cnt], class_cnt, config['thresh_unify'],
                                                                    config['unique_classes'])

                            # print(f"flag_0sim1:{flag_0sim1}")
                            # print(f"flag_1sim2:{flag_1sim2}")
                            # print(f"flag_2sim0:{flag_2sim0}")


                            # unify or not unify according to flag
                            if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    if frame_cnt not in output_dict:
                                        output_dict[frame_cnt] = []
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    if frame_cnt not in output_dict:
                                        output_dict[frame_cnt] = []
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    if frame_cnt not in output_dict:
                                        output_dict[frame_cnt] = []
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                            elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                if flag_0sim1:
                                    if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                        output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                       doa_pred2[frame_cnt][
                                                                           class_cnt + config['unique_classes']],
                                                                       doa_pred2[frame_cnt][
                                                                           class_cnt + 2 * config['unique_classes']]])
                                    doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                                   doa_pred_fc[class_cnt + config['unique_classes']],
                                                                   doa_pred_fc[class_cnt + 2 * config['unique_classes']]])
                                elif flag_1sim2:
                                    if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                        output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                       doa_pred0[frame_cnt][
                                                                           class_cnt + config['unique_classes']],
                                                                       doa_pred0[frame_cnt][
                                                                           class_cnt + 2 * config['unique_classes']]])
                                    doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                                   doa_pred_fc[class_cnt + config['unique_classes']],
                                                                   doa_pred_fc[class_cnt + 2 * config['unique_classes']]])
                                elif flag_2sim0:
                                    if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                        output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                       doa_pred1[frame_cnt][
                                                                           class_cnt + config['unique_classes']],
                                                                       doa_pred1[frame_cnt][
                                                                           class_cnt + 2 * config['unique_classes']]])
                                    doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                    output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                                   doa_pred_fc[class_cnt + config['unique_classes']],
                                                                   doa_pred_fc[class_cnt + 2 * config['unique_classes']]])
                            elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                                output_dict[frame_cnt].append(
                                    [class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + config['unique_classes']],
                                     doa_pred_fc[class_cnt + 2 * config['unique_classes']]])

                    # output_dict_polar = {}
                    # for k, v in output_dict.items():
                    #     ss = []
                    #     for this_item in v:
                    #         tmp = utils.cart2sph(this_item[-3], this_item[-2], this_item[-1])
                    #         ss.append([this_item[0], *tmp])
                    #     output_dict_polar[k] = ss
                    write_output_format_file(config, output_file, output_dict, use_cartesian=True)

                else:
                    for frame_cnt in range(sed_pred.shape[0]):
                        for class_cnt in range(sed_pred.shape[1]):
                            if sed_pred[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                                                               doa_pred[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])

                    write_output_format_file(config,output_file, output_dict,use_cartesian=True)
                # print(f"output_file:{output_file}")
                # print(f"output_dict:{output_dict}")

                test_loss += loss.item()

                nb_test_batches += 1

        test_loss /= nb_test_batches

        # params = parameters.get_params('3')
        # score_obj = cls_compute_seld_results.ComputeSELDResults(params)
        # score_obj.get_SELD_Results(dcase_output_folder)


    all_test_metric_macro, all_test_metric_micro = all_seld_eval(config, directory_root=dataset.directory_root,
                                                                 fnames=dataset._fnames,
                                                                 pred_directory=dcase_output_folder)

    if writer is not None:
        writer.add_scalar('Losses/valid', test_loss, iter_idx)
        writer.add_scalar('MMacro/ER', all_test_metric_macro[0], iter_idx)
        writer.add_scalar('MMacro/F', all_test_metric_macro[1], iter_idx)
        writer.add_scalar('MMacro/LE', all_test_metric_macro[2], iter_idx)
        writer.add_scalar('MMacro/LR', all_test_metric_macro[3], iter_idx)
        writer.add_scalar('MMacro/SELD', all_test_metric_macro[4], iter_idx)

        writer.add_scalar('Mmicro/ER', all_test_metric_micro[0], iter_idx)
        writer.add_scalar('Mmicro/F', all_test_metric_micro[1], iter_idx)
        writer.add_scalar('Mmicro/LE', all_test_metric_micro[2], iter_idx)
        writer.add_scalar('Mmicro/LR', all_test_metric_micro[3], iter_idx)
        writer.add_scalar('Mmicro/SELD', all_test_metric_micro[4], iter_idx)

    return all_test_metric_macro, all_test_metric_micro, test_loss, outputs_for_plots


def evaluation(config, dataset, solver, features_transform, target_transform: [nn.Sequential, None], dcase_output_folder, device, detection_threshold=0.4):
    # Adapted from the official baseline
    # This is basically the same as validaiton, but we dont compute losses or metrics because there are no labels
    nb_test_batches, test_loss = 0, 0.
    model = solver.predictor
    model.eval()
    file_cnt = 0

    print(f'Evaluation: {len(dataset)} fnames in dataset.')
    with torch.no_grad():
        for ctr, (audio, _, fname) in enumerate(dataset):
            # load one batch of data
            audio = audio.to(device)
            duration = dataset.durations[fname]
            print(f'Evaluating file {ctr+1}/{len(dataset)}: {fname}')
            print(f'Audio shape: {audio.shape}')

            warnings.warn('WARNING: Hard coded chunk size for evaluation')
            audio_padding, labels_padding = _get_padders(chunk_size_seconds=config.dataset_chunk_size / dataset._fs[fname],
                                                         duration_seconds=math.floor(duration),
                                                         overlap=1,
                                                         audio_fs=dataset._fs[fname],
                                                         labels_fs=100)

            # Split each wav into chunks and process them
            audio = audio_padding['padder'](audio)
            audio_chunks = audio.unfold(dimension=1, size=audio_padding['chunk_size'],
                                        step=audio_padding['hop_size']).permute((1, 0, 2))

            full_output = []
            tmp = torch.utils.data.TensorDataset(audio_chunks)
            loader = DataLoader(tmp, batch_size=1, shuffle=False, drop_last=False)  # Loader per wav to get batches
            for ctr, (audio) in enumerate(loader):
                audio = audio[0]
                if features_transform is not None:
                    audio = features_transform(audio)
                output = model(audio)
                full_output.append(output)

            # Concatenate chunks across timesteps into final predictions
            if config.dataset_multi_track:
                output = torch.concat(full_output, dim=-2)
            else:
                output = torch.concat(full_output, dim=-1)

            # Apply detection threshold based on vector norm
            if config.dataset_multi_track:
                pass
            else:
                norms = torch.linalg.vector_norm(output, ord=2, dim=-3, keepdims=True)
                norms = (norms < detection_threshold).repeat(1, output.shape[-3], 1, 1)
                output[norms] = 0.0

            # Downsample over frames:
            if config.dataset_multi_track:
                if target_transform is None:
                    output = nn.functional.interpolate(output.permute(0, 2, 1), scale_factor=(0.1), mode='nearest-exact').permute(0, 2, 1)
            else:
                if target_transform is None:
                    output = nn.functional.interpolate(output, scale_factor=(1, 0.1), mode='nearest-exact')

                # I think the baseline code needs this in [batch, frames, classes*coords]
                output = output.permute([0, 3, 1, 2])
                output = output.flatten(2, 3)

            if config['dataset_multi_track'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(
                    output.detach().cpu().numpy(), config['unique_classes'])
                sed_pred0 = cls_compute_seld_results.reshape_3Dto2D(sed_pred0)
                doa_pred0 = cls_compute_seld_results.reshape_3Dto2D(doa_pred0)
                sed_pred1 = cls_compute_seld_results.reshape_3Dto2D(sed_pred1)
                doa_pred1 = cls_compute_seld_results.reshape_3Dto2D(doa_pred1)
                sed_pred2 = cls_compute_seld_results.reshape_3Dto2D(sed_pred2)
                doa_pred2 = cls_compute_seld_results.reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), config['unique_classes'])
                sed_pred = cls_compute_seld_results.reshape_3Dto2D(sed_pred)
                doa_pred = cls_compute_seld_results.reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            tmp_name = fname.split('/')[-1]
            output_file = os.path.join(dcase_output_folder, tmp_name.replace('.wav', '.csv'))
            file_cnt += 1
            output_dict = {}
            if config['dataset_multi_track'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                sed_pred1[frame_cnt][class_cnt],
                                                                doa_pred0[frame_cnt], doa_pred1[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                sed_pred2[frame_cnt][class_cnt],
                                                                doa_pred1[frame_cnt], doa_pred2[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                sed_pred0[frame_cnt][class_cnt],
                                                                doa_pred2[frame_cnt], doa_pred0[frame_cnt],
                                                                class_cnt, config['thresh_unify'],
                                                                config['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + config['unique_classes']],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + 2 * config['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + config['unique_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * config['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + config['unique_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * config['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt],
                                                           doa_pred_fc[class_cnt + config['unique_classes']],
                                                           doa_pred_fc[class_cnt + 2 * config['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt] > 0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            tmp_azi, tmp_ele = utils.cart2sph(doa_pred[frame_cnt][class_cnt],
                                                           doa_pred[frame_cnt][class_cnt + config['unique_classes']],
                                                           doa_pred[frame_cnt][class_cnt + 2 * config['unique_classes']])
                            output_dict[frame_cnt].append([class_cnt, tmp_azi, tmp_ele])
                            #output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt],
                            #                               doa_pred[frame_cnt][
                            #                                   class_cnt + config['unique_classes']],
                            #                               doa_pred[frame_cnt][
                            #                                   class_cnt + 2 * config['unique_classes']]])
            write_output_format_file(output_file, output_dict, use_cartesian=False, ignore_src_id=True)

    print('Finished evaluation')



if __name__ == '__main__':
    main()
