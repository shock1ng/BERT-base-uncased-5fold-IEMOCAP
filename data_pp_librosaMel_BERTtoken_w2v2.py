# -*- coding: utf-8 -*-
# @Time : 2023/10/19 13:14
# @Author : JohnnyYuan
# @File : data_pp_librosaMel_BERTtoken_w2v2.py


"""
Created on Tue Jan  9 20:32:28 2018

@author: shixiaohan
"""
import re
import wave
import numpy as np
import python_speech_features as ps
import soundfile as sf
import os
import glob
import pickle
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model,BertTokenizer
import librosa

rootdir = '/home/hd/SGao/IEMOCAP_full_release/'     # full_release解压

#wav2vec2
processor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2-xls-r-300m")
model = Wav2Vec2Model.from_pretrained("wav2vec2-xls-r-300m")
#BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
MAX_LEN = 128

label_list = [1, 2, 3, 4, 5]

def emo_change(x):
    if x == 'xxx' or x == 'oth':
        x = 0
    if x == 'neu':
        x = 1
    if x == 'hap':
        x = 2
    if x == 'ang':
        x = 3
    if x == 'sad':
        x = 4
    if x == 'exc':
        x = 5
    if x == 'sur':
        x = 6
    if x == 'fea':
        x = 7
    if x == 'dis':
        x = 8
    if x == 'fru':
        x = 9
    return x

def process_wav_file(wav_file, time):
    waveform, sample_rate = torchaudio.load(wav_file)
    target_length = time * sample_rate
    # 将WAV文件裁剪为目标长度
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    else:
        # 如果WAV文件长度小于目标长度，则使用填充进行扩展
        padding_length = target_length - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, padding_length))

    return waveform, sample_rate

# 拿取数据直接生成梅尔谱
def mel_spectrogram(file_path):   #返回(128,xxx)
    audio_data, sampleRate = librosa.load(file_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sampleRate)  # 这里的16000是采样率
    return mel_spec

def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()    # 获取音频文件的参数：通道数、采样宽度、频率、文件长度，是一个元组
    nchannels, sampwidth, framerate, wav_length = params[:4]
    channel = file.getnchannels()
    sampwidth = file.getsampwidth()
    framerate = file.getframerate()
    frames = file.getnframes()
    duration = frames/framerate   #帧/频率 = 时长
    wav_length = 3 * framerate    # 音频长度设定为3秒
    str_data = file.readframes(wav_length)   # 读取3秒的数据
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)  # 从0开始读取到音频长度，再除以频率得到时长
    file.close()
    return wavedata, time, framerate   # 返回音频数据、时长、采样频率

def Read_IEMOCAP_Spec():
    filter_num = 40   # 梅尔滤波器的数量，控制计算的特征维度
    train_num = 0
    train_mel_data = []
    for speaker in os.listdir(rootdir):     # 返回名称列表，这里返回的有6个文件夹名，是否返回别的文件名我不知道，文件夹有一个D开头，别的5个都是S开头的
        if (speaker[0] == 'S'):             # 当文件夹是S开头的时候进行以下操作，否则迭代
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav')    # 加上子路径，变成如“IEMOCAP_full_release/Session1/sentences/wav”
            for sess in os.listdir(sub_dir):   # 此时拿到一堆文件夹名
                if (sess[7] in ['i','s']):     # 从0开始数到第7个字符是i或者s时进行操作
                    file_dir = os.path.join(sub_dir, sess, '*.wav')    #变成如“IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/*.wav”
                    files = glob.glob(file_dir)       # glob.glob会查找并返回当前目录下以".wav"结尾的所有文件的路径
                    for filename in files:            # filename是单个文件的路径
                        wavname = filename.split("/")[-1][:-4]   #使用 / 字符将字符串 filename 分割成一个列表，返回如 ["path", "to", "file", "audio.wav"] ，然后拿取最后一个元素即'audio.wav'，再切片，最后4个字符不要，变成audio
                        # data, time, rate = read_file(filename)   # 得到音频数据、时长、采样频率
                        # mel_spec = ps.logfbank(data, rate, nfilt=filter_num)   # 输出可能是个2维数组
                        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
                            # training set
                            one_mel_data = {}
                            mel_data = mel_spectrogram(filename)   # 直接使用工具变成mel，这里的形状是(128,xxx)
                            one_mel_data['id'] = wavname
                            mel_data = np.array(mel_data)
                            one_mel_data['spec_data'] = mel_data.T      # 记住一定要转置，会变成(xxx,128)
                            train_mel_data.append(one_mel_data)
                            train_num = train_num + 1
    #print(train_num)
    return train_mel_data   # 最后这里输出的是一个列表，列表中的每个元素都由{id,spec_data}字典构成

def Read_IEMOCAP_Text():
    traindata_map_1 = []
    train_num = 0
    for speaker in os.listdir(rootdir):
        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
            text_dir = os.path.join(rootdir, speaker, 'dialog/transcriptions')
            for sess in os.listdir(text_dir):   # 目录下所有txt文件的名称
                if (sess[7] in ['i','s']):
                    data_map1_1 = []
                    textdir = text_dir + '/' + sess
                    text_map = {}
                    # with open(textdir, 'r') as text_to_read:
                    with open(textdir, 'r', encoding='unicode_escape') as text_to_read:
                        while True:
                            line = text_to_read.readline()   # 按行读取，每次读取一行
                            if not line:   # 当读不到就弹出
                                break
                            t = line.split()   # 按照空格把每个字符都分隔出来，返回列表. transp = "hello world" --> t = ["hello","world"]
                            if (t[0][0] in 'S'):   #每个txt都如： Ses02F_script01_1_M000 [010.1700-011.8100]: He saw it.  当首字母是S时开始操作
                                str = " ".join(t[2:])  # 从列表的第2号坐标开始，用空格连接。因为第0号是id，第1号是不知道啥，第2号才是文本
                                text_map['id'] = t[0]  # id赋值，t的0下标就是id.......当迭代后就是对 id 重新赋值
                                text_map['transcription'] = str  # 文本赋值.......当迭代后就是对 transcription 重新赋值

                                # 新增使用bert的tokenizer， 使用正则表达式去除标点符号
                                str = re.sub(r'[^\w\s]', '', str)
                                # 对输入文本进行tokenization并添加特殊token
                                encoding = tokenizer(str, truncation=True, padding='max_length', max_length=MAX_LEN, return_tensors='pt')
                                input_ids = encoding.get('input_ids').squeeze()
                                attention_mask = encoding.get('attention_mask').squeeze()
                                text_map['input_ids'] = input_ids
                                text_map['attention_mask'] = attention_mask

                                a = text_map.copy()    # 把当前的text_map复制一份变成a
                                data_map1_1.append(a)  # 把a塞进大列表中，由于有很多很多行，所以data_map1_1是(行数*txt文本数)个子字典构成的列表
                    traindata_map_1.append(data_map1_1)  # 把五个组的都组合起来
                    train_num = train_num + 1
    # 到这里，traindata_map_1是[[{id,trans,input_ids,att_mask},{id,trans,input_ids,att_mask}...{id,trans,input_ids,att_mask}],[s2],[s3],[s4],[s5]]   长度由2变4
    #print(train_num)

    traindata_map_2 = []
    train_num_1 = 0
    for speaker in os.listdir(rootdir):
        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
            emoevl = os.path.join(rootdir, speaker, 'dialog/EmoEvaluation')
            for sess in os.listdir(emoevl):
                if (sess[-1] in ['t']):   # 当文件名最后一个字符是t，开始操作，这样操作的话，包含 . 开头的txt也会被操作
                    data_map2_1 = []
                    emotdir = emoevl + '/' + sess
                    # emotfile = open(emotdir)
                    emot_map = {}
                    # with open(emotdir, 'r') as emot_to_read:
                    with open(emotdir, 'r', encoding='unicode_escape') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if (line[0] == '['):
                                t = line.split()
                                emot_map['id'] = t[3]
                                x = t[5] + t[6] + t[7]
                                x = re.split(r'[,[]', x)
                                y = re.split(r'[]]', x[3])
                                emot_map['emotion_v'] = float(x[1])
                                emot_map['emotion_a'] = float(x[2])
                                emot_map['emotion_d'] = float(y[0])
                                emot_map['label'] = emo_change(t[4])
                                a = emot_map.copy()
                                data_map2_1.append(a)
                    traindata_map_2.append(data_map2_1)  #[[{id, v, a, d, label},{},{},...,{}],[s2],[s3],[s4],[s5]]
                    train_num_1 = train_num_1 + 1
    #print(train_num_1)

    for i in range(len(traindata_map_1)):        # 总长 5
        for j in range(len(traindata_map_1[i])): # 单个组的长度，如s1里全部的id长度
            for x in range(len(traindata_map_2)):  # 总长 5
                for y in range(len(traindata_map_2[x])):  # 单个组的长度，如s1里全部的id长度   {id, v, a, d, label}
                    if (traindata_map_1[i][j]['id'] == traindata_map_2[x][y]['id']):     # 当1和2的id对上了，就把2的内容加给1
                        traindata_map_1[i][j]['emotion_v'] = traindata_map_2[x][y]['emotion_v']
                        traindata_map_1[i][j]['emotion_a'] = traindata_map_2[x][y]['emotion_a']
                        traindata_map_1[i][j]['emotion_d'] = traindata_map_2[x][y]['emotion_d']
                        traindata_map_1[i][j]['label'] = traindata_map_2[x][y]['label']
    # 走完上面，traindata_map_1就是[[{id, trans, ids, att_msk v, a, d, label},{},...,{}],[s2],[s3],[s4],[s5]]  # 子长扩大到8
    train_data_map = []
    for i in range(len(traindata_map_1)):  # 总长是5
        data_map_1 = []
        for x in range(len(traindata_map_1[i])): # 单个组的总长
            if (len(traindata_map_1[i][x]) == 8): # 单个id下的总长，{id, trans, ids, msk v, a, d, label}是8，等于8时开始操作
                data_map_1.append(traindata_map_1[i][x])
        train_data_map.append(data_map_1)  # 也就是把单个id下没满足长度为8的过一下筛
    return train_data_map  # [[{id, trans, ids, msk ,v, a, d, label},{},...,{}],[s2],[s3],[s4],[s5]]

def Read_IEMOCAP_Trad():
    filter_num = 40
    train_num = 0
    train_mel_data = []
    for speaker in os.listdir(rootdir):
        if (speaker[0] == 'S'):
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav')
            for sess in os.listdir(sub_dir):
                if (sess[7] in ['i','s']):
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    for filename in files:
                        wavname = filename.split("/")[-1][:-4]
                        audio_input, sample_rate = process_wav_file(filename,3)
                        #audio_input, sample_rate = sf.read(filename)
                        #input_values = processor(audio_input, sampling_rate=sample_rate,return_tensors="pt").input_values
                        input_values =processor(audio_input, return_tensors="pt").input_values    # torch.size([1,1,48000])
                        if (speaker in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']):
                            # training set
                            one_mel_data = {}
                            one_mel_data['id'] = wavname
                            one_mel_data['wav_encodings'] = input_values
                            train_mel_data.append(one_mel_data)
                            train_num = train_num + 1
    #print(train_num)
    return train_mel_data   # 返回的是列表，每个列表元素都是一个小字典{id,wav_encodings}，这里是经过Wav2vec2预处理后的encoder信息

def Seg_IEMOCAP(train_data_spec,train_data_text,train_data_trad):
    for i in range(len(train_data_text)):     # 总长是5   [{id, trans, ids, msk v, a, d, label},{},...,{}],[s2],[s3],[s4],[s5]]
        for x in range(len(train_data_text[i])):  # 取其中一组
            for y in range(len(train_data_spec)):  # (0,10039)
                if (train_data_text[i][x]['id'] == train_data_spec[y]['id']):  # 当id对上的时候把spec_data加上字典去
                    train_data_text[i][x]['spec_data'] = train_data_spec[y]['spec_data']

    for i in range(len(train_data_text)):
        for x in range(len(train_data_text[i])):
            for y in range(len(train_data_trad)):
                if (train_data_text[i][x]['id'] == train_data_trad[y]['id']):  # 同理，当id对上的时候把wav_encodings加上字典去
                    train_data_text[i][x]['wav_encodings'] = train_data_trad[y]['wav_encodings']
    num = 0
    train_data_map = []
    for i in range(len(train_data_text)):
        data_map_1 = []
        for x in range(len(train_data_text[i])):
            if (len(train_data_text[i][x]) == 10):  # 由于加上了spec_data和 wav_encodings ，长度由8变成10
                data_map_1.append(train_data_text[i][x])
                num = num + 1
        train_data_map.append(data_map_1)
    print(num)
    return train_data_map  # [[{id, trans,ids, msk, v, a, d, label, spec_data ,wav_encodings},{},...,{}],[s2],[s3],[s4],[s5]]  总长是5，子长是单个组的所有id数，再子长就是10

def Train_data(train_map):
    train_data_ALL_1 = []
    label_list= [1,2,3,4,5]
    num = 0
    for i in range(len(train_map)):  # 总长是5， [[{id, trans,ids, msk, v, a, d, label, spec_data ,wav_encodings},{},...,{}],[s2],[s3],[s4],[s5]]
        train_data = []
        for j in range(len(train_map[i])): # 这里长度是单个组下所有的id数目
            data = {}
            ####### 选择上面10个内容选择性打包 #######
            data['label'] = train_map[i][j]['label']
            data['wav_encodings'] = train_map[i][j]['wav_encodings']
            data['id'] = train_map[i][j]['id']
            data['input_ids'] = train_map[i][j]['input_ids']
            data['attention_mask'] = train_map[i][j]['attention_mask']
            data['transcription'] = train_map[i][j]['transcription']
            data['emotion_v'] = train_map[i][j]['emotion_v']
            data['emotion_a'] = train_map[i][j]['emotion_a']
            data['emotion_d'] = train_map[i][j]['emotion_d']
            data['spec_data'] = train_map[i][j]['spec_data']
            #####################################
            if(data['label'] in label_list):  # 只考虑标签是1~5的
                if(data['label'] == 5):   # 把标签为5的改为2
                    data['label'] = 2
                data['label'] = data['label'] - 1 # 标签前移，因为网络不接受（不从0开始的、不按顺序的）
                train_data.append(data)  # 保存一个组的全部数据  [{id, trans, ids, msk, v, a, d, label, spec_data ,wav_encodings},...,{}]
                num = num + 1
        train_data_ALL_1.append(train_data)  # 总长是5，子长是这个组下label为1~5内的所有id数，再子长是10

    print(len(train_data_ALL_1))  # 5   151咋来的
    print(len(train_data_ALL_1[0]))  # ? 24
    print(len(train_data_ALL_1[0][0]))  # 10
    print(num)  # 5531

    data_1 = []
    data_2 = []
    data_3 = []
    data_4 = []
    data_5 = []

    for i in range(len(train_data_ALL_1)):   # 按照组别分开保存。   这个循环体的作用是？？感觉没什么必要，上面已经把组别分开了，二次确保吗？？
        for j in range(len(train_data_ALL_1[i])):
            if (train_data_ALL_1[i][j]['id'][4] == '1'):
                data_1.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4] == '2'):
                data_2.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4]== '3'):
                data_3.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4] == '4'):
                data_4.append(train_data_ALL_1[i][j])
            if (train_data_ALL_1[i][j]['id'][4] == '5'):
                data_5.append(train_data_ALL_1[i][j])

    data = []
    data.append(data_1)
    data.append(data_2)
    data.append(data_3)
    data.append(data_4)
    data.append(data_5)
    return data

if __name__ == '__main__':
    train_data_spec = Read_IEMOCAP_Spec()   # [{id,spec_data}]
    train_data_trad = Read_IEMOCAP_Trad()   # [{id,wav2vec2embedding}]
    train_data_text = Read_IEMOCAP_Text()   # [{id, trans, ids, msk ,v, a, d, label},{},...,{}],[s2],[s3],[s4],[s5]]

    train_data_map = Seg_IEMOCAP(train_data_spec,train_data_text,train_data_trad)
    Train_data = Train_data(train_data_map)   # [[{id, trans, v, a, d, label, spec_data ,wav_encodings},{},...,{}],[s2],[s3],[s4],[s5]] 总长是5，子长是这个组下label为1~5内的所有id数，再子长是8

    file = open('Train_data_w2v2Large_BERTbaseTokenized_librosaMel.pickle', 'wb')
    pickle.dump(Train_data, file)
    file.close()