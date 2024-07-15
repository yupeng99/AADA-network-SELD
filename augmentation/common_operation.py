import random
import numpy as np
import torch_audiomentations as t_aug
import torch
from parameters import get_parameters
from main import get_rotations_noise,get_spatial_mixup
import augmentation.spatial_mixup as spm
config = get_parameters()
# import PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw

# random_mirror = True

#https://www.osgeo.cn/pillow/reference/ImageOps.html# 数据增强网站

mode = 'per_example'  # for speed, we use batch processing
p_mode = 'per_example'
fs = 24000
threshold_limiter=1




def identity(audio, _):
    return audio

 # transforms=[
 #              t_aug.Gain(p=p, min_gain_in_db=-15.0, max_gain_in_db=6.0, mode=mode, p_mode=p_mode),
 #              t_aug.PolarityInversion(p=p, mode=mode, p_mode=p_mode),
 #              #t_aug.PitchShift(p=p, min_transpose_semitones=-1.5, max_transpose_semitones=1.5, sample_rate=fs,
 #              #                 mode=mode, p_mode=p_mode),
 #              t_aug.AddColoredNoise(p=p, min_snr_in_db=2.0, max_snr_in_db=30.0, min_f_decay=-2.0, max_f_decay=2.0,
 #                                    sample_rate=fs, mode=mode, p_mode=p_mode),
 #              t_aug.BandStopFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
 #                                   min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.1, sample_rate=fs,
 #                                   p_mode=p_mode),
 #              t_aug.LowPassFilter(p=p, min_cutoff_freq=1000, max_cutoff_freq=5000, sample_rate=fs,
 #                                  p_mode=p_mode),
 #              t_aug.HighPassFilter(p=p, min_cutoff_freq=250, max_cutoff_freq=1500, sample_rate=fs,
 #                                   p_mode=p_mode),
 #              t_aug.BandPassFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
 #                                   min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.5, sample_rate=fs,
 #                                   p_mode=p_mode),
 #              #t_aug.SpliceOut(p=p, num_time_intervals=100, max_width=100, sample_rate=fs, p_mode=p_mode)
 #                                      ]


def gain(audio,p):
    assert 0 <= p <= 1
    apply_augmentation = t_aug.Compose(transforms=[t_aug.Gain(p=p, min_gain_in_db=-15.0, max_gain_in_db=6.0,mode=mode, p_mode=p_mode)])
    output = apply_augmentation(audio,fs)
    # output = torch.clamp(output, min=-threshold_limiter, max=threshold_limiter)
    return output

def polarityInversion(audio,p):
    assert 0 <= p <= 1
    apply_augmentation = t_aug.Compose(
        transforms=[t_aug.PolarityInversion(p=p, mode=mode, p_mode=p_mode)])
    output = apply_augmentation(audio, fs)

    # output = t_aug.PolarityInversion(p=p, mode=mode, p_mode=p_mode).apply_transform(audio,fs)["samples"]
    # output = torch.clamp(output, min=-threshold_limiter, max=threshold_limiter)
    return output

def addColoredNoise(audio,p):
    assert 0 <= p <= 1
    apply_augmentation = t_aug.Compose(
        transforms=[t_aug.AddColoredNoise(p=p, min_snr_in_db=2.0, max_snr_in_db=30.0, min_f_decay=-2.0, max_f_decay=2.0,
                                    sample_rate=fs, mode=mode, p_mode=p_mode)])
    output = apply_augmentation(audio, fs)
    # output = t_aug.AddColoredNoise(p=p, min_snr_in_db=2.0, max_snr_in_db=30.0, min_f_decay=-2.0, max_f_decay=2.0,
    #                                 sample_rate=fs, mode=mode, p_mode=p_mode).apply_transform(audio, fs)["samples"]
    # output = torch.clamp(output, min=-threshold_limiter, max=threshold_limiter)
    return output
def bandStopFilter(audio,p):
    assert 0 <= p <= 1

    apply_augmentation = t_aug.Compose(
        transforms=[t_aug.BandStopFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
                                min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.1, sample_rate=fs,
                                p_mode=p_mode)])
    output = apply_augmentation(audio, fs)
    # output =  t_aug.BandStopFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
    #                             min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.1, sample_rate=fs,
    #                             p_mode=p_mode).apply_transform(audio, fs)["samples"]
    # output = torch.clamp(output, min=-threshold_limiter, max=threshold_limiter)
    return output

def lowPassFilter(audio,p):
    assert 0 <= p <= 1
    apply_augmentation = t_aug.Compose(
        transforms=[t_aug.LowPassFilter(p=p, min_cutoff_freq=1000, max_cutoff_freq=5000, sample_rate=fs,
                                  p_mode=p_mode)])
    output = apply_augmentation(audio, fs)
    # output =  t_aug.LowPassFilter(p=p, min_cutoff_freq=1000, max_cutoff_freq=5000, sample_rate=fs,
    #                               p_mode=p_mode).apply_transform(audio, fs)["samples"]
    # output = torch.clamp(output, min=-threshold_limiter, max=threshold_limiter)
    return output

def highPassFilter(audio,p):
    assert 0 <= p <= 1
    apply_augmentation = t_aug.Compose(
        transforms=[t_aug.HighPassFilter(p=p, min_cutoff_freq=250, max_cutoff_freq=1500, sample_rate=fs,
                                   p_mode=p_mode)])
    output = apply_augmentation(audio, fs)
    # output = t_aug.HighPassFilter(p=p, min_cutoff_freq=250, max_cutoff_freq=1500, sample_rate=fs,
    #                                p_mode=p_mode).apply_transform(audio, fs)["samples"]
    # output = torch.clamp(output, min=-threshold_limiter, max=threshold_limiter)
    return output

def bandPassFilter(audio,p):
    assert 0 <= p <= 1
    apply_augmentation = t_aug.Compose(
        transforms=[t_aug.BandPassFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
                                   min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.5, sample_rate=fs,
                                   p_mode=p_mode)])
    output = apply_augmentation(audio, fs)
    # output = t_aug.BandPassFilter(p=p, min_center_frequency=400, max_center_frequency=4000,
    #                                min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.5, sample_rate=fs,
    #                                p_mode=p_mode).apply_transform(audio, fs)["samples"]
    output = torch.clamp(output, min=-threshold_limiter, max=threshold_limiter)
    return output

def rotation_noise_aug(audio,p):
    rotations_noise = get_rotations_noise(device="cuda").to("cuda")
    rotations_noise.reset_R(mode='noise')
    rotations_noise.p_comp = p
    audio, _ = rotations_noise(audio)
    return audio

def augmentation_transform_spatial_aug(audio,p):
    augmentation_transform_spatial = get_spatial_mixup(device="cuda").to("cuda")
    augmentation_transform_spatial.reset_G(G_type='spherical_cap_soft')
    augmentation_transform_spatial.p_comp = p
    audio = augmentation_transform_spatial(audio)
    return audio

AUGMENT_LIST = [
        (identity, ),  # 0
        (gain, ),  # 1
        (polarityInversion, ),  # 2
        (addColoredNoise, ),  # 3
        (bandStopFilter, ),  # 4
        (lowPassFilter, ),  # 5
        (highPassFilter,),  # 6
        (bandPassFilter,),  # 7
        (rotation_noise_aug,),  # 8
        (augmentation_transform_spatial_aug,), #9
            ]  # 16


def get_augment(name):
    augment_dict = {fn.__name__: (fn,) for fn, in AUGMENT_LIST}
    return augment_dict[name]


def apply_augment(audio, name, p):
    # audio = audio.to("cuda")
    # p = p.to("cuda")
    augment_fn = get_augment(name)[0]
    return augment_fn(audio,p)



# def rotations_aug(audio,p,target):
#     rotation_transform = get_rotations(device="cuda").to("cuda")
#     rotation_transform.reset_R(mode=config.model_rotations_mode)
#     rotation_transform.p_comp = p
#     audio, target = rotation_transform(audio, target)
#     return audio,target