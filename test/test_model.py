from models.projection import Projection
from torchinfo import summary
from models.resnet_conformer1 import ResnetConformer,ResNet,BasicBlock
from models.resnet_conformer import ResnetConformer
from models.crnn import CRNN10
# summary(Projection(512,1),input_size = [32,16,512],device="cuda")

import torch
from torch_audiomentations import Compose, Gain, PolarityInversion


# Initialize augmentation callable
# apply_augmentation = Compose(
#     transforms=[
#         Gain(
#             min_gain_in_db=-15.0,
#             max_gain_in_db=5.0,
#             p=0.5,
#         ),
#         PolarityInversion(p=0.5)
#     ]
# )
#
# torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # Make an example tensor with white noise.
# # This tensor represents 8 audio snippets with 2 channels (stereo) and 2 s of 16 kHz audio.
# audio_samples = torch.rand(size=(2, 32000), dtype=torch.float32, device=torch_device) - 0.5
# print(f"audio_samples.shape:{len(audio_samples.shape)}")
# if len(audio_samples.shape) == 2:
#     print("yes")
#     do_reshape = True
#     audio_samples = audio_samples[None, ...]
#     print({audio_samples.shape})
#
# # Apply augmentation. This varies the gain and polarity of (some of)
# # the audio snippets in the batch independently.
# perturbed_audio_samples = apply_augmentation(audio_samples, sample_rate=16000)
# if do_reshape:
#     aug_audio = perturbed_audio_samples.squeeze(0)
#
# print(aug_audio.shape)

summary(ResnetConformer((10,7,500,64)),input_size = [10,7,500,64],device="cuda")
# summary(CRNN10(13,7,3),input_size = [10,7,256,512],device="cuda")

# def resnet34(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnet34-333f7ec4.pth
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)
# # resnet34(13)
# summary(resnet34(13),input_size = [10,7,500,64],device="cuda")
