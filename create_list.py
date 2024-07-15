#生成合成数据集
import os
path = "/home/yupeng/datasets/DCASE2022_SELD_synth_data"

dev_path = os.path.join(path,"foa_dev")

# 生成 train——list

train_file = open("synth_data_devtrain_all.txt",'w')
for audio_path in os.listdir(dev_path):
        train_file.write(os.path.join('.',path.split("/")[-1],"foa_dev",audio_path)+"\n")
train_file.close()


#生成官方数据集

# import os
# # path = "/home/yupeng/datasets/dcase2023"
#
# dev_path = os.path.join(path,"foa_dev")
#
# # 生成 train——list
# train_file = open("dcase2023_devtrain_all.txt",'w')
# for location_path in os.listdir(dev_path):
#     if(location_path.split("-")[1] == "train"):
#         audio_path = os.path.join(dev_path,location_path)
#         for audio in os.listdir(audio_path):
#             train_file.write(os.path.join('.',path.split("/")[-1],"foa_dev",location_path,audio)+"\n")
# train_file.close()
#
# # 生成 test——list
#
# test_file = open("dcase2023_devtest_all.txt", 'w')
# for location_path in os.listdir(dev_path):
#     if (location_path.split("-")[1] == "test"):
#         audio_path = os.path.join(dev_path, location_path)
#         for audio in os.listdir(audio_path):
#             test_file.write(os.path.join('.', path.split("/")[-1], "foa_dev", location_path, audio)+"\n")
# test_file.close()