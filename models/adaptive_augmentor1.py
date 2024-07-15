# 除第一个方法的其它数据增强方法 用权重 * 方法

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from augmentation.common_operation import apply_augment
from main import get_rotations

from models.utils import PolicyHistory
from config import OPS_NAMES

default_config = {'sampling': 'prob',
                  'k_ops': 1,
                  'delta': 0,
                  'temp': 1.0,
                  'search_d': 32,
                  'target_d': 32}


def perturb_param(param, delta):  # delta 0.3 param：一个DA的幅度值
    if delta <= 0:
        return param

    amt = random.uniform(0, delta)  # 0，delta之间随机产生一个数
    if random.random() < 0.5:
        return max(0, param - amt)
    else:
        return min(1, param + amt)


def stop_gradient(trans_image, magnitude):
    """
    该函数的作用是在进行图像增强操作时，先将幅度值从图像中减去，然后再将其添加回去。
    这样做的目的是在增强操作期间防止梯度流经图像，从而保持图像不受梯度的影响。
    这在一些情况下是有用的，例如当希望在训练中应用一些随机性的增强操作，但又不希望这些增强操作影响到梯度传播时的图像。
    """
    images = trans_image
    adds = 0

    images = images - magnitude
    adds = adds + magnitude
    images = images.detach() + adds  # 反向传播不会计算它的梯度
    return images

def rotations_aug(config,audio,target,p):
    rotation_transform = get_rotations(device="cuda").to("cuda")
    rotation_transform.reset_R(mode=config.model_rotations_mode)
    rotation_transform.p_comp = p
    audio, target = rotation_transform(audio, target)
    return audio,target


class MDAAug(nn.Module):
    def __init__(self, n_class, gf_model, h_model, sys_config,iters, features_transform: [nn.Sequential, None], save_dir=None,
                 config=default_config, ):
        super(MDAAug, self).__init__()
        self.ops_names = OPS_NAMES
        self.n_ops = len(self.ops_names)

        self.save_dir = save_dir
        self.gf_model = gf_model
        self.h_model = h_model
        self.n_class = n_class
        # self.resize = config['search_d'] != config['target_d']
        # self.search_d = config['search_d']
        self.k_ops = config['k_ops']
        self.current_ops = 1
        self.sampling = config['sampling']
        self.temp = config['temp']
        self.delta = config['delta']
        self.config = sys_config
        self.search_freq = sys_config.search_freq
        self.history = PolicyHistory(self.ops_names, self.save_dir, self.n_class)
        self.features_transform = features_transform
        self.iters = iters
        if os.path.exists(sys_config.logging_dir + '/aug_method') == False:
            os.makedirs(sys_config.logging_dir + '/aug_method')
        self.fs = open(sys_config.logging_dir + '/aug_method/use_aug.txt', 'a', encoding="utf-8")

    def save_history(self, class2label=None):
        self.history.save(class2label)

    def plot_history(self):
        return self.history.plot()

    def predict_aug_params(self, data, mode):
        self.gf_model.eval()
        if mode == 'exploit':
            self.h_model.train()
            T = self.temp  # 2   exploit 的 T 比 explore 的 T 大  weights/T 值就小， 经过 softmax 输出  的概率值就不明显 （区分度小）
        elif mode == 'explore':
            self.h_model.train()
            T = 1.0  # 相比于 explore 的 T  区分度更大
        a_params = self.h_model(self.gf_model.f(data.cuda()))
        # print(f"a_params.shape:{a_params.shape}") #(32,20)
        magnitudes, weights = torch.split(a_params, self.n_ops,
                                          dim=1)  # magnitudes ：[样本数,所有个DA的幅度值]  weights：[样本数,所有个DA的权重值(概率)]

        magnitudes_clone = magnitudes.clone()
        magnitudes_clone[magnitudes == float('inf')] = 1.0
        magnitudes_clone[(magnitudes == float('-inf')) | (magnitudes < 0) | torch.isnan(magnitudes)] = 0.0

        magnitudes = torch.sigmoid(magnitudes_clone)  # 对 magnitudes 值 映射到 0~1 之间

        ### weights出现了probability tensor contains either `inf`, `nan` or element < 0的问题
        # 将无穷大设置为0.999
        processed_input = weights.clone()
        processed_input[weights == float('inf')] = 0.9
        # 将无穷小和小于0的数设置为0.001
        processed_input[(weights == float('-inf')) | (weights < 0)|torch.isnan(weights)] = 0.0001

        weights = torch.nn.functional.softmax(processed_input / T, dim=-1)  # 使得 每个样本的 所有数据增强的 weights 累计和为 1

        return magnitudes, weights

    def add_history(self, data, targets):
        magnitudes, weights = self.predict_aug_params(data, 'exploit')
        for k in range(self.n_class):
            idxs = (targets == k).nonzero().squeeze()
            mean_lambda = magnitudes[idxs].mean(0).detach().cpu().tolist()
            mean_p = weights[idxs].mean(0).detach().cpu().tolist()
            std_lambda = magnitudes[idxs].std(0).detach().cpu().tolist()
            std_p = weights[idxs].std(0).detach().cpu().tolist()
            self.history.add(k, mean_lambda, mean_p, std_lambda, std_p)
            # print(k, mean_lambda, mean_p, std_lambda, std_p)

    def get_aug_valid_audios(self, audios, magnitudes,target,weights):
        """Return the augmented imgae

        Args:
            images ([Tensor]): [description]
            magnitudes ([Tensor]): [description]
        Returns:
            [Tensor]: a set of augmented validation images

        不考虑DA的概率值（weight） 所有batch 数据都根据magnitude来 使用 DA
        """
        trans_audio_list = []
        trans_target_list = []
        for i, audio in enumerate(audios):
            # Prepare transformed image for mixing
            do_reshape = False
            if len(audio.shape) == 2:
                do_reshape = True
                audio = audio[None, ...]
                target_i = target[i][None, ...]
            if weights[i][0] != 0:
            # trans_audio, target_i = rotations_aug(self.config, audio, target_i, magnitudes[i][0])
                audio, target_i = rotations_aug(self.config, audio, target_i, magnitudes[i][0])
                audio = weights[i][0] * audio
                target_i = target_i.squeeze(0)
                trans_target_list.append(target_i)
            for k, ops_name in enumerate(self.ops_names[1:]):
                trans_audio = apply_augment(audio, ops_name, magnitudes[i][k+1])
                # trans_image = self.after_transforms(trans_image)
                trans_audio = stop_gradient(trans_audio.cuda(), magnitudes[i][k+1])
                if do_reshape:
                    trans_audio = trans_audio.squeeze(0)

                trans_audio_list.append(trans_audio)  # 每一次对 原始图像 使用DA操作后 添加到list

        return torch.stack(trans_audio_list, dim=0),torch.stack(trans_target_list, dim=0)

    def explore(self, audios,target):
        """Return the mixed features

        Args:
            images ([Tensor]): [description]
        Returns:
            [Tensor]: return a batch of mixed images
        """
        features = self.features_transform(audios)
        magnitudes, weights = self.predict_aug_params(features, 'explore')
        # print(f"weights.shape:{weights.shape}\tmagnitudes:{magnitudes.shape}") #都是[32,10]
        a_audios,target = self.get_aug_valid_audios(audios, magnitudes,target,weights)  # 增强后的图像数据
        ba_audios = a_audios.reshape(len(audios), self.n_ops-1, -1)  # （数据个数，DA操作个数，每个DA操作后的数据）
        # print(f"weights.shape:{weights.shape}\tba_audios:{ba_audios.shape}") # ba_audios：[32,10,244800]  244800 = 4 * 61200



        mixed_audios = [w.matmul(feat) for w, feat in zip(weights[:,1:], ba_audios)]  # 每个对原始图像进行DA后的数据 * weight 累计和
        mixed_audios = torch.stack(mixed_audios, dim=0)  # 形成一个新的张量  (64,3072)  (batch,data)
        mixed_audios = mixed_audios.reshape(len(audios), 4, -1)  # (数据个数,通道数，12000)

        if self.features_transform is not None:
            mixed_features = self.features_transform(mixed_audios)
            # if self.augmentation_transform_spec is not None:
            #     augmentation_transform_spec = self.RandomSpecAugmentations(p_comp=magnitudes[-1])  //得在后面再写
            #     x = augmentation_transform_spec(x)

        mixed_features = self.gf_model.f(mixed_features)  # 提取特征
        return mixed_features,target

    def get_training_aug_images(self, audios, magnitudes, weights,target):

        # visualization
        if self.k_ops > 0:
            trans_audios = []  # 存储增强后的图像数据
            trans_target = []  # 存储增强后的图像数据
            # k_ops = random.randint(1,self.k_ops)
            if(self.config.kops_linear):
                k_ops = random.randint(1, self.k_ops)
                # ops_plus = int(self.config.num_iters/self.k_ops)
                # if (self.iters % ops_plus == 0 and self.iters!=0):
                #     self.current_ops = self.current_ops+1
                #     if (self.current_ops >self.k_ops):
                #         self.current_ops = self.k_ops
                #     print(f"current_ops:{self.current_ops}")
            else:

                k_ops = self.k_ops
            if self.sampling == 'prob':
                idx_matrix = torch.multinomial(weights,
                                               k_ops)  # shape(5,2)无放回采样 ，元素权重是0时，在其他元素被取完之前是不会被取到的。值越大被抽中的概率值的索引越大。 比如 idx_matrix：tensor([[15,  8],[ 5, 11], [13,  7],[14,  0],[ 9,  1]], device='cuda:0')

            elif self.sampling == 'max':
                idx_matrix = torch.topk(weights, k_ops, dim=1)[1]  # 选择最大的k_ops个索引
            if (self.iters % self.search_freq == 0):
                self.fs.write(f"iters:{self.iters}\n")
            for i, audio in enumerate(audios):  # images shape(5,3,32,32)
                do_reshape = False
                if len(audio.shape) == 2:
                    do_reshape = True
                    audio = audio[None, ...]
                    target_i = target[i][None, ...]
                # if weights[i][0] != 0:

                for idx in idx_matrix[i]:
                    if idx == 0:
                        m_pi = perturb_param(magnitudes[i][0], self.delta)
                        audio, target_i = rotations_aug(self.config, audio, target_i, m_pi)
                    else :
                        m_pi = perturb_param(magnitudes[i][idx], self.delta)  # 让一个 DA的幅度值 加或者减 delta  一定在（0，1）
                        # print(f"audio.shape:{audio.shape}")
                        audio = apply_augment(audio, self.ops_names[idx], m_pi)

                    if (self.iters % self.search_freq == 0):
                        self.fs.write("第" + str(i) + "个音频" + "augment_method:" + self.ops_names[idx] + "\tp:" + str(
                            m_pi) + "\n")
                if (self.iters % self.search_freq == 0):
                    self.fs.write("\n")
                if do_reshape:
                    audio = audio.squeeze(0)
                    target_i = target_i.squeeze(0)
                # print(f"audio.shape:{audio.shape}")
                trans_audios.append(audio)  # 将PIL 数据 变成 Tensor 放入list中
                trans_target.append(target_i)
            if (self.iters % self.search_freq == 0):
                self.fs.write("\n\n\n")
        else: #
            trans_audios = []
            trans_target = []
            for i, audio in enumerate(audios):
                if weights[i][0] != 0:
                    p = perturb_param(magnitudes[i][0], self.delta)
                    audio, target[i] = rotations_aug(self.config, audio, target[i], p)
                trans_audios.append(audio)
                trans_target.append(target[i])
        feature_aug_audios = self.features_transform(torch.stack(trans_audios, dim=0).cuda())
        self.iters = self.iters + 1
        return feature_aug_audios,torch.stack(trans_target, dim=0)

    def exploit(self, audios,target):
        # resize_imgs = F.interpolate(audios, size=self.search_d) if self.resize else audios
        if self.features_transform is not None:
            features = self.features_transform(audios)
        magnitudes, weights = self.predict_aug_params(features, 'exploit')
        features_aug_audio,target = self.get_training_aug_images(audios, magnitudes, weights,target)
        # if self.features_transform is not None:
        #     features_aug_audio = self.features_transform(aug_audios)
        return features_aug_audio,target

    def forward(self, audios, target,mode):
        if mode == 'explore':
            #  return a set of mixed augmented features
            return self.explore(audios,target)
        elif mode == 'exploit':
            #  return a set of augmented images
            return self.exploit(audios,target)
        elif mode == 'inference':
            return audios
