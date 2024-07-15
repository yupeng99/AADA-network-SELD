import os
import numpy as np
import scipy.io.wavfile as wav
from collections import deque
import random
from label_feature_extraction import cls_feature_class
from utils.logger import setup_logger
# from parameters import get_params
logger = setup_logger(__name__)



class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False,use_aug = False
    ):
        self._per_file = per_file
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params['batch_size']
        self._audio_seq_len = 5*24000  # 250 帧
        self._label_seq_len = params['label_sequence_length'] # 50帧  每个0.1s
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        # self._label_dir = self._feat_cls.get_label_dir() # 提取完 label后
        # self._feat_dir = self._feat_cls.get_normalized_feat_dir()

        self._dataset_dir = params['dataset_dir']  # 获取数据集根路径
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')  # 获取数据集原音频的根目录
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)  # 得到数据集原音频文件的根路径
        # self._label_dir = "/home/yupeng/datasets/dcase2023/metadata_dev"
        self.use_aug = use_aug
        self._desc_dir = params["desc_dir"]

        self._multi_accdoa = params['multi_accdoa']

        self._audio_data_dict = {} # {'fullfilepath :data}
        self._filenames_list = list()  # ['/home/yupeng/datasets/dcase2023/foa_dev/dev-train-sony/fold3_room21_mix027.wav']
        self._nb_len_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        # self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._nb_classes = params['unique_classes']
        self._filewise_frames = {}
        self._circ_buf_feat = None
        self._circ_buf_label = None
        self._fs = params['fs']  # 采样频率 24000
        self._hop_len_s = params['hop_len_s']  # 特征的每秒的帧移长度 0.02
        self._hop_len = int(self._fs * self._hop_len_s)  # 得到特征的每秒帧移的采样点个数  0.02 * 24000 = 480

        self._label_hop_len_s = params['label_hop_len_s']  # 标签的每秒帧移长度 0.1 s
        self._label_hop_len = int(self._fs * self._label_hop_len_s)  # 得到标签的每秒帧移的采样点个数  0.1 * 24000 = 240
        self._label_frame_res = self._fs / float(self._label_hop_len)  # 每秒 标签可形成的帧数 24000 / 240 =  10
        self._nb_label_frames_1s = int(self._label_frame_res)
        self.eps = 1e-8
        self._nb_channels = 4
        self.nb_mel_bins = params["nb_mel_bins"]
        self._get_filenames_list_and_feat_label_sizes()

        logger.info(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list), self._nb_classes,
                self._nb_len_file,  self._nb_ch, self._label_len
            )
        )

        logger.info(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\taudio_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._audio_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._desc_dir, self._aud_dir
            )
        )

    def _get_filenames_list_and_feat_label_sizes(self):
        print('Computing some stats about the dataset')
        max_len, total_len, temp_audio = -1, 0, []
        for sub_folder in os.listdir(self._aud_dir):
            loc_aud_folder = os.path.join(self._aud_dir, sub_folder)
            for filename in os.listdir(loc_aud_folder):
                if int(filename[4]) in self._splits:  # check which split the file belongs to
                    full_audio_path =os.path.join(loc_aud_folder,filename)
                    self._filenames_list.append(full_audio_path)  #得到音频绝对文件路径

                    print(f"full_audio_path:{full_audio_path}")  ###

                    sr,temp_audio = wav.read(full_audio_path)
                    self._audio_data_dict[full_audio_path] = temp_audio

                    total_len += (temp_audio.shape[0] - (temp_audio.shape[0] % self._audio_seq_len))
                    nb_feat_frames = int(temp_audio.shape[0] / float(self._hop_len))
                    nb_label_frames = int(temp_audio.shape[0] / float(self._label_hop_len))
                    self._filewise_frames[filename.split('.')[0]] = [nb_feat_frames, nb_label_frames]
                    if temp_audio.shape[0] > max_len:
                        max_len = temp_audio.shape[0]

        # print(f"_filewise_frames:{self._filewise_frames}")  ####

        if len(temp_audio) != 0:
            self._nb_len_file = max_len if self._per_file else temp_audio.shape[0]
            # self._nb_ch = temp_audio.shape[1] // self._nb_mel_bins
        else:
            print('Loading audio failed')
            exit()


                # np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

        if self._per_file:
            self._batch_size = int(np.ceil(max_len / float(self._audio_seq_len)))
            print(
                '\tWARNING: Resetting batch size to {}. To accommodate the inference of longest file of {} frames in a single batch'.format(
                    self._batch_size, max_len))
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor(total_len / (self._batch_size * self._audio_seq_len)))

        self._audio_batch_seq_len = self._batch_size * self._audio_seq_len
        self._label_batch_seq_len = self._batch_size * self._label_seq_len
        return

    def get_filelist(self):
        return self._filenames_list

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
                _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), 0))
        _fid.close()

    def get_data_sizes(self):
        feat_shape = (self._batch_size, 7, int(self._audio_seq_len/480), self.nb_mel_bins)

        if self._multi_accdoa is True:
            label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3*3)
        else:
            label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3)
        return feat_shape, label_shape

    # def get_frame_stats(self):
    #
    #     if len(self._filewise_frames)!=0:
    #         return
    #
    #     logger.info('Computing frame stats:')
    #     logger.info('\t\taud_dir {}\n\t\tdesc_dir {}'.format(
    #         self._aud_dir, self._desc_dir))
    #
    #     for audio_path in self._filenames_list:
    #         if int(audio_path.split('/')[-1][4]) in self._splits:  # check which split the file belongs to
    #             with contextlib.closing(wave.open(audio_path, 'r')) as f:
    #                 file_name = audio_path.split('/')[-1]
    #                 audio_len = f.getnframes()
    #                 nb_feat_frames = int(audio_len / float(self._hop_len))
    #                 nb_label_frames = int(audio_len / float(self._label_hop_len))
    #                 self._filewise_frames[file_name.split('.')[0]] = [nb_feat_frames, nb_label_frames]
    #
    #     return
# param = get_params(argv='3')
# DataGenerator() dg = new DataGenerator(param)

    def generate_audio_label(self):
        """
        Generates batches of samples
        :return:
        """
        # 获取 源数据 文件名
        # for filename in os.listdir(self._feat_dir):
        #     if int(filename[4]) in self._splits: # check which split the file belongs to
        #         self._filenames_list.append(filename)

        if self._shuffle:
            random.shuffle(self._filenames_list)

        # Ideally this should have been outside the while loop. But while generating the test data we want the data
        # to be the same exactly for all epoch's hence we keep it here.
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()

        file_cnt = 0

        for i in range(self._nb_total_batches):
            # print(f"第{i}个batch/总共{self._nb_total_batches}batch")
            # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
            # circular buffer. If not keep refilling it.
            while len(self._circ_buf_feat) < self._audio_batch_seq_len:
                # temp_audio = np.load(os.path.join(self._aud_dir, self._filenames_list[file_cnt]))  #
                # temp_audio = []
                for audio_path in self._filenames_list:
                    # print(f"int(audio_path.split('/')[-1][4]:{int(audio_path.split('/')[-1][4])}")
                    if int(audio_path.split('/')[-1][4]) in self._splits:  # check which split the file belongs to
                        #提取audio
                        # sr,temp_audio  = wav.read(audio_path) # (序列,通道数)
                        temp_audio = self._audio_data_dict[audio_path]
                        temp_audio = temp_audio[:, :self._nb_channels] / 32768.0 + self.eps
                        #提取label
                        label_path = audio_path.replace('foa_dev','metadata_dev').replace('wav','csv')
                        # print(f"label_path:{label_path}")############
                        file_name = label_path.split('/')[-1]
                        nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]  # 标签帧数
                        desc_file_polar = self._feat_cls.load_output_format_file(label_path)
                        desc_file = self._feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar)
                        if self._multi_accdoa:
                            temp_label = self._feat_cls.get_adpit_labels_for_file(desc_file, nb_label_frames)
                        else:
                            temp_label = self._feat_cls.get_labels_for_file(desc_file, nb_label_frames)
                        # print('{}: {}'.format( file_name, temp_label.shape))


                # temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt]))

                #提取标签
                # temp_label = []
                # self.get_frame_stats()
                # for sub_folder in os.listdir(self._desc_dir):
                #     loc_desc_folder = os.path.join(self._desc_dir, sub_folder)
                #     for file_cnt, file_name in enumerate(os.listdir(loc_desc_folder)):
                #         wav_filename = '{}.wav'.format(file_name.split('.')[0])
                #         nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]  # 标签帧数
                #         desc_file_polar = self._feat_cls.load_output_format_file(
                #             os.path.join(loc_desc_folder, file_name))
                #         desc_file = self._feat_cls.convert_output_format_polar_to_cartesian(desc_file_polar)
                #         if self._multi_accdoa:
                #             temp_label = self._feat_cls.get_adpit_labels_for_file(desc_file, nb_label_frames)
                #         else:
                #             temp_label = self._feat_cls.get_labels_for_file(desc_file, nb_label_frames)
                #         print('{}: {}, {}'.format(file_cnt, file_name, temp_label.shape))

                if not self._per_file:
                    # Inorder to support variable length features, and labels of different resolution.
                    # We remove all frames in features and labels matrix that are outside
                    # the multiple of self._label_seq_len and self._feature_seq_len. Further we do this only in training.
                    temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                    temp_mul = temp_label.shape[0] // self._label_seq_len
                    temp_audio = temp_audio[:temp_mul * self._audio_seq_len,:]

                for f_row in temp_audio:
                    self._circ_buf_feat.append(f_row)
                for l_row in temp_label:
                    self._circ_buf_label.append(l_row)

                # If self._per_file is True, this returns the sequences belonging to a single audio recording
                if self._per_file:  # 按文件划分数据样本，需将不足部分用0填充
                    feat_extra_frames = self._audio_batch_seq_len - temp_audio.shape[0]
                    extra_feat = np.ones((feat_extra_frames, temp_audio.shape[1])) * 1e-6

                    label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                    if self._multi_accdoa is True:
                        extra_labels = np.zeros(
                            (label_extra_frames,  6, 4, 13))
                    else:
                        extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                    for f_row in extra_feat:
                        self._circ_buf_feat.append(f_row)
                    for l_row in extra_labels:
                        self._circ_buf_label.append(l_row)

                file_cnt = file_cnt + 1

            # Read one batch size from the circular buffer
            audio = np.zeros((self._audio_batch_seq_len, 4))
            for j in range(self._audio_batch_seq_len):
                audio[j,:] = self._circ_buf_feat.popleft()
            # audio = np.reshape(audio, (self._audio_batch_seq_len))

            label = np.zeros((self._label_batch_seq_len, 6, 4, 13))
            for j in range(self._label_batch_seq_len):
                label[j, :, :, :] = self._circ_buf_label.popleft()
            # Split to sequences
            if audio.shape[0] % self._audio_seq_len:
                audio = audio[:-(audio.shape[0] % self._audio_seq_len),:] #

            audio = audio.reshape((audio.shape[0] // self._audio_seq_len, self._audio_seq_len, audio.shape[1])) #(B,seq_len,ch)

            # audio = np.transpose(audio, (0, 2, 1, 3))  # (batch_size,channels,frames,dim)


            if label.shape[0] % self._label_seq_len:
                label = label[:-(label.shape[0] % self._label_seq_len), :, :, :]
            label = label.reshape((label.shape[0] // self._label_seq_len, self._label_seq_len, label.shape[1], label.shape[2], label.shape[3]))

            # print('audio.sahpe:{}'.format(audio.shape))
            # print('label.sahpe:{}'.format(label.shape))
            yield audio, label

# data_gen_train = DataGenerator(
#             params= get_params(argv='3'), split=[1,2,3]
#         )
# for data, target in data_gen_train.generate_audio_label():
#     print(data.shape,target.shape)
#     print(f"feat.shape:{aug_and_extraction(data).shape}")
