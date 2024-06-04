import soundfile as sf
import numpy as np
from scipy.signal import resample
import os
import random
import torch
import librosa
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import Dataset
import pickle
import librosa
from scipy.io import wavfile
import glob

def check_files_in_folder(folder_path, keyword):
    # 获取文件夹内所有文件的列表
    all_files = glob.glob(os.path.join(folder_path, '*'))

    # 遍历文件列表，检查是否包含指定关键词
    matching_files = [file for file in all_files if keyword in os.path.basename(file)]

    return matching_files

# def add_noise(wav_path,noisy_path,out_wav_path,SNR):
def add_noise(wav_path,noise_dir_path='/ssdhome/lixuefei/data/Noisy/noisy',SNR=None,out_wav_path=None):
     
    noise_path_list, noise_dict = get_random_noise(noise_dir_path)
    noisy_path = random.choice(noise_path_list)
    noise_name = noisy_path.split('/')[-1].replace('.wav','')
    wav_name =  out_wav_path.split('/')[-1]
    out_dir_path = out_wav_path.replace(wav_name,'')
    a = check_files_in_folder(out_dir_path,wav_name)
    
    # if :
    #     pass
    # else:
    noise_label = noise_dict[noise_name]
    # noisy_path,noise_label = get_random_noise(noise_dir_path)
    SNR = get_random_snr()
    # print('noise_path:',noisy_path)
    # print('SNR:',SNR)
    noise_name = noisy_path.split('/')[-1].replace('.wav','')
    data, fs = sf.read(wav_path)
    data = librosa.resample(y = data,orig_sr=fs,target_sr=44100)
    fs = 44100
    # print(fs)

    # 读取噪声文件
    noise, noise_fs = sf.read(noisy_path)

    # 将噪声重采样到和音频相同的采样率
    noise = resample(noise, int(len(noise) * fs / noise_fs))

    # 将噪声的数据复制，使噪声的长度和音频相同
    if len(noise) <= len(data):
        noise = np.tile(noise, len(data)//len(noise)+1)[:len(data)]
    else:
        noise = noise[:len(data)]

    #计算声音能量（功率）和噪声能量（功率）
    pow_data = (data**2).mean()
    pow_noise = (noise**2).mean()

    #给定信噪比，添加相应能量的噪音
    scale = (
            10 ** (-SNR / 20)
            * np.sqrt(pow_data)
            / np.sqrt(max(pow_noise, 1e-10))
        )

    # 将噪声混入原始音频上
    data += scale * noise

    # # 写入新的文件
    # out_wav_path = out_wav_path.split('.')[0]+'_'+noise_name+'_'+str(SNR)+'dB.wav'
    out_wav_path = out_wav_path.split('.')[0]+'.'+out_wav_path.split('.')[1]+'_'+noise_name+'_'+str(SNR)+'dB.wav'
    print(out_wav_path)
    sf.write(out_wav_path, data, fs)
    return fs, data, noise_name,noise_label

def get_random_noise(noise_dir_path):
    noise_path_list = []
    noise_dict = {}
    i=0
    for noise in os.listdir(noise_dir_path):
        if noise.endswith('wav'):
            noise_path = noise_dir_path+'/'+noise
            noise_path_list.append(noise_path)
            noise_name = noise.replace('.wav','')
            noise_dict[noise_name]=i
            i+=1
    # print(noise_path_list)
    # print(noise_dict)
    # random_noise_path = random.choice(noise_path_list)
    # random_noise_path,noise_dict[noise_name]
    return noise_path_list, noise_dict

def get_random_snr():
    SNR = [-10,-5,0,5,10,15]
    random_snr = random.choice(SNR)
    return random_snr

def get_frames(abs_wav_path,noise_dir_path='/ssdhome/lixuefei/data/Noisy/noisy',model_srate=44100,step_size= 128/44100,len_frame_time=0.064):
    sample_rate,audio, noise_name,noise_label = add_noise(abs_wav_path,noise_dir_path)
    # sample_rate, audio = wavfile.read(abs_wav_path)  # 读取音频文件，返回采样率 和 信号
    audio = audio.astype(np.float32)
    
    #分帧
    hop_length = int(sample_rate * step_size)  
    wlen = int(sample_rate * len_frame_time)
    n_frames = 1 + int((len(audio) - wlen) / hop_length)
    frames = as_strided(audio, shape=(wlen, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()
    #z-score归一化
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.std(frames, axis=1)[:, np.newaxis]
    frames[np.isnan(frames)] = 0
    ret = librosa.resample(y=frames, res_type='linear', orig_sr=sample_rate, target_sr=model_srate)
    return ret, noise_label


if __name__ == "__main__":

    wav_dir_path = 'your wav dir path'
    noise_dir_path = 'your noise dir path'
    new_wav_dir = 'the wav dir path which you will save '



    for wav in os.listdir(wav_dir_path):
     
        wav_path = wav_dir_path +'/' +wav
        new_wav_path = new_wav_dir+'/' +wav
        add_noise(wav_path,out_wav_path=new_wav_path)
        
