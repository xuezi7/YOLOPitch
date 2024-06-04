import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import librosa
import os
import re
import sys
# from dataset_org import Net_DataSet
# from dataset_wav_noise import Net_DataSet  之前模型未使用
# from dataset_stft import Net_DataSet
# from dataset_pefac import Net_DataSet
# from dataset_stft_noise import Net_DataSet
from dataset_stft_wav import Net_DataSet
from tqdm import tqdm
# from cmndf_fc_unet import YIN
# from cnn_unet_cmndf_self_11 import YIN
# from yolo_wav_stft_wo_backbone import YoloBody
from yolo_wav_stft_2bei import YoloBody
# from crepe_fdc_freq import Crepe
from formula_all import *
import time
import logging
import config

log = feature.get_logger()

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# devihen gce = 'cpu'

validation_csv_path = config.validation_csv_path
train_data = Net_DataSet(config.train_path)
validation_data = Net_DataSet(config.validation_path)
test_data = Net_DataSet(config.test_path)
batch_size = 1
train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=16)
validation_dataloader = DataLoader(validation_data, batch_size=batch_size, num_workers=16)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=16)

# resume_path = "/ssdhome/lixuefei/data/PTDB/model/model_2.pth"
#恢复模型，可以在中断的时候从保存的模型继续训练
resume_path = ""
# model = YIN().to(device)
# model = Crepe().to(device)
model = YoloBody(phi='l', pretrained=False).to(device)
if resume_path:
    model.load_state_dict(torch.load(resume_path))
print(model)
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.NLLLoss(reduction='sum')
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))

def get_label(path):
    pitch = []
    ref_cent = []
    # pitch.append(f0)
    with open(path, mode="r") as file:
        for line in file.readlines():
            
            if 'ptdb' in config.data_name :
                x = float(line.split(" ")[0])
                if x != 0:
                    cent = Convert.convert_hz_to_cent(x)
                else:
                    cent = 0
                hz=x

            elif 'ikala' in config.data_name or  'mir1k' in config.data_name:
                x = float(line.split(" ")[0])
                x = Convert.convert_semitone_to_hz(x)
                if x >= 10:
                    hz = x
                    cent = Convert.convert_hz_to_cent(x)
                else:
                    hz = 0
                    cent = 0

            elif 'mdb' in config.data_name :
                hz = float(line.split(",")[1].split("\n")[0])
                if hz > 0:
                    cent = Convert.convert_hz_to_cent(hz)
                else:
                    cent = 0

            pitch.append(hz) 
            ref_cent.append(cent)               
    return pitch,ref_cent

def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset) #dataloader.dataset 即取出train_dataloader的train_data内容
    model.train()

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # X = X.unsqueeze(0)
        # y = y.unsqueeze(0)
        # X = X.transpose(1,2)
        y = y[0].type(torch.LongTensor)
        # X = X.transpose(1,2).unsqueeze(0)
        X_wav,X_stft, y = X[0].to(device),X[1].to(device), y.to(device).squeeze(0)
        # Compute prediction error
        # X = CMNDF(X,512)
        # pred = model(X_stft,X_wav)
        pred = model(X_wav,X_stft)
        min_num = min(y.shape[0],pred.shape[0])
        pred = pred[:min_num,:]
        y = y[:min_num]
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            #查看运行进度
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def validation(dataloader, model, loss_fn,csv_path=validation_csv_path):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    total = 0
    voice_total = 0
    voice_correct = 0
    with torch.no_grad():
        sound_score_list=[]
        music_score_list=[]
        for X, Y in dataloader:
            y = Y[0].type(torch.LongTensor)
            # X = X.transpose(1,2).unsqueeze(0)
            # X = X.transpose(1,2)
            X_wav,X_stft, y = X[0].to(device),X[1].to(device), y.to(device).squeeze(0)
            # Compute prediction error
            # X = CMNDF(X,512)
            pred = model(X_wav,X_stft)
            # X, y = X.to(device), y.to(device).squeeze(0)
            # # X = CMNDF(X,512)
            # pred = model(X).squeeze(0)

            min_num = min(y.shape[0],pred.shape[0])
            pred = pred[:min_num,:]
            y = y[:min_num]
            
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            total += y.shape[0]
            zero_index = y != 0
            zero_pred = pred[zero_index]
            zero_y = y[zero_index]
            voice_correct += (zero_pred.argmax(1) == zero_y).type(torch.float).sum().item()
            voice_total += zero_index.sum().item()
            
            pitch_data = torch.max(pred,dim=1)
            pitch_data_1 = pitch_data[1].cpu()
            pitch_data = np.array(pitch_data_1)
            pitch_cent = []
            pitch=[]
            for i in pitch_data:
                if config.data_name in 'ptdb':
                    if i !=0:
                        predict_cent = Convert.convert_hz_to_cent(i)  
                        # k = Convert.convert_cent_to_hz(j)      
                    else:
                        predict_cent = 0
                        # k = 0
                    predict_hz= i

                else:
                    if i !=0:
                        predict_cent = Convert.convert_bin_to_cent(i)  
                        predict_hz = Convert.convert_cent_to_hz(predict_cent)      
                    else:
                        predict_cent = 0
                        predict_hz = 0

                pitch_cent.append(predict_cent)
                pitch.append(predict_hz)


            ref_cent_list = get_label(csv_path+'/'+Y[1][0])[1]
            label_data = get_label(csv_path+'/'+Y[1][0])[0] 

            len_pitch = len(pitch)
            len_label = len(label_data)
            pitch = pitch[:min(len_pitch,len_label)]
            label = label_data[:min(len_pitch,len_label)]
            pitch_cent = pitch_cent[:min(len_pitch,len_label)]
            label_cent = ref_cent_list[:min(len_pitch,len_label)]

            pitch = np.array(pitch)
            label = np.array(label)
            pitch_cent = np.array(pitch_cent)
            label_cent = np.array(label_cent)
            sound_score = Sound.all_mir_eval(pitch,label,threshold = 0.2)
            # print("sound_score",sound_score)
            music_score = Music.all_mir_eval(label, label_cent, pitch, pitch_cent,cent_tolerance=50)
            # print("music_score",music_score)
            sound_score_list.append(sound_score)
            music_score_list.append(music_score)
    

    test_loss /= num_batches
    correct /= total
    voice_correct /= voice_total
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"voice_correct: \n voice_Accuracy: {(100*voice_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    sound_score_numpy = np.array(sound_score_list)
    music_score_numpy = np.array(music_score_list)
    # sound_score_avg = np.sum(sound_score_numpy,axis = 0) / sound_score_numpy.shape[0]
    sound_score_avg = np.nanmean(sound_score_numpy,axis = 0)
    music_score_avg = np.sum(music_score_numpy,axis = 0) / music_score_numpy.shape[0]
    
    print("score结果:")
    print("sound_score_avg:",sound_score_avg)
    print('music_score_avg:',music_score_avg)
    log.info(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    log.info(f"voice_correct: \n voice_Accuracy: {(100*voice_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    log.info(f"score结果:")
    log.info(f"sound_score_avg:{sound_score_avg}")
    log.info(f"music_score_avg:,{music_score_avg}")

# epochs = config.epochs
epochs = 500
log.info('yolo_'+config.data_name+'_yolo_wav_stft_wo_tb')
# save_model_path = config.save_model_path
save_model_path = 'save mosel path'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    log.info(f"Epoch {t+1}\n-------------------------------")
    #加载之前的模型可用于预训练
    #注意：这里注释掉的前两行若放开，会造成训练错误
     # model = Crepe()
    # model.to(device)
    # aaa='/ssdhome/lixuefei/model/crepe/model_99.pth'
    # model.load_state_dict(torch.load(aaa))
    train(train_dataloader, model, loss_fn, optimizer)
    validation(validation_dataloader, model, loss_fn)
    log.info(f"\n\n")
    validation(test_dataloader, model, loss_fn)

    # if (t+1)%10==0 or (t+1)>=epochs-15:
    #     torch.save(model.state_dict(), save_model_path + "/" +"model_{}.pth".format(t+1))
print("Done!")

print("Saved PyTorch Model State to model.pth")










