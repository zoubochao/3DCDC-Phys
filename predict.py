# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import model
from inference_preprocess import preprocess_raw_video,detrend
import h5py
import numpy as np
from scipy.signal import butter
import scipy
import matplotlib.pyplot as plt
import math
from scipy.signal import resample

n_frame       = 10
nb_filters1   = 32 
nb_filters2   = 64
input_shape   = (36,36,3)
input_shape_1 = (36,36,n_frame,3)    #用于3D卷积模型
input_shape_2 = (36,36,3)            
kernel_size   = (3, 3)
dropout_rate1 = 0.25
dropout_rate2 = 0.5
pool_size     = (2, 2)
nb_dense      = 128
batch_size    = n_frame
fs            = 20


#获取第二种数据集的标签
def get_label(label_txt):
    with open(label_txt) as f:
        a = f.readlines()
    b = a[0].replace('\n','').split(' ')
    c = [i for i in b if i!='']
    d = []
    for i in c:
        l = float(i.split('e')[0])
        r = int(i.split('e')[1])
        if r!=0:
            l = l*(10**r)
        d.append(l)
    return np.array(d,dtype=np.float32)

def mag2db(mag):
    """Convert a magnitude to decibels (dB)
    If A is magnitude,
        db = 20 * log10(A)
    Parameters
    ----------
    mag : float or ndarray
        input magnitude or array of magnitudes
    Returns
    -------
    db : float or ndarray
        corresponding values in decibels
    """
    return 20. * np.log10(mag)


#由ppg得到心率/计算四种指标
#--------------------------------------------------------------------------------------------------
def calculate_HR(pxx_pred, frange_pred, fmask_pred, pxx_label, frange_label, fmask_label):
    pred_HR = np.take(frange_pred, np.argmax(np.take(pxx_pred, fmask_pred), 0))[0] * 60
    ground_truth_HR = np.take(frange_label, np.argmax(np.take(pxx_label, fmask_label), 0))[0] * 60
    return pred_HR, ground_truth_HR


def calculate_SNR(pxx_pred, f_pred, currHR, signal):
    currHR = currHR/60
    f = f_pred
    pxx = pxx_pred
    gtmask1 = (f >= currHR - 0.1) & (f <= currHR + 0.1)
    gtmask2 = (f >= currHR * 2 - 0.1) & (f <= currHR * 2 + 0.1)
    sPower = np.sum(np.take(pxx, np.where(gtmask1 | gtmask2)))
    if signal == 'pulse':
        fmask2 = (f >= 0.75) & (f <= 4)
    else:
        fmask2 = (f >= 0.08) & (f <= 0.5)
    allPower = np.sum(np.take(pxx, np.where(fmask2 == True)))
    SNR_temp = mag2db(sPower / (allPower - sPower))
    return SNR_temp
# %%  Processing


def calculate_metric(predictions, labels, signal='pulse', window_size=3900, fs=20, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass') # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    
    data_len = len(predictions)
    #print('data_len',data_len)
    HR_pred = []
    HR0_pred = []
    mySNR = []
    for j in range(0, data_len, window_size):
        if j == 0 and (j+window_size) > data_len:
            pred_window = predictions
            label_window = labels
        elif (j + window_size) >  data_len:
            break
        else:
            pred_window = predictions[j:j + window_size]
            label_window = labels[j:j + window_size]
        if signal == 'pulse':
            pred_window = detrend(np.cumsum(pred_window), 100)
        else:
            pred_window = np.cumsum(pred_window)

        label_window = np.squeeze(label_window)
        if bpFlag:
            pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))

        pred_window = np.expand_dims(pred_window, 0)
        label_window = np.expand_dims(label_window, 0)
        # Predictions FFT
        f_prd, pxx_pred = scipy.signal.periodogram(pred_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse':
            fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        else:
            fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
        pred_window = np.take(f_prd, fmask_pred)
        # Labels FFT
        f_label, pxx_label = scipy.signal.periodogram(label_window, fs=fs, nfft=4 * window_size, detrend=False)
        if signal == 'pulse':
            fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))  # regular Heart beat are 0.75*60 and 2.5*60
        else:
            fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))  # regular Heart beat are 0.75*60 and 2.5*60
        label_window = np.take(f_label, fmask_label)

        # MAE
        temp_HR, temp_HR_0 = calculate_HR(pxx_pred, pred_window, fmask_pred, pxx_label, label_window, fmask_label)
        temp_SNR = calculate_SNR(pxx_pred, f_prd, temp_HR_0, signal)
        HR_pred.append(temp_HR)
        HR0_pred.append(temp_HR_0)
        mySNR.append(temp_SNR)

    HR = np.array(HR_pred)
    HR0 = np.array(HR0_pred)
    mySNR = np.array(mySNR)

    MAE = np.mean(np.abs(HR - HR0))
    RMSE = np.sqrt(np.mean(np.square(HR - HR0)))
    pearson = np.corrcoef(HR, HR0)
    meanSNR = np.nanmean(mySNR)
    return MAE, RMSE,pearson ,meanSNR, HR0, HR
#——————————————————————————————————————————————————————————————————————————————————————————



#--------------------------------------------------------------------------------------------
def pre(model_weights,model_type,video_path,m_or_p,label_path=None):
    if model_type=='CAN':
        Model = model.CAN(nb_filters1, nb_filters2, input_shape)
        Model.load_weights(model_weights)
    
        dXsub = preprocess_raw_video(video_path, dim=36)
        dXsub_len = (dXsub.shape[0] // n_frame)  * n_frame
        dXsub = dXsub[:dXsub_len, :, :, :]
    
        yptest = Model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)
        return yptest
    
        # pulse_pred = yptest
        
        # pulse_pred = detrend(np.cumsum(pulse_pred), 100)
        # [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        # pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
    
        ########## Plot ##################
        # plt.subplot(211)
        # plt.plot(pulse_pred)
        # plt.title('Pulse Prediction')

    elif model_type=='MT_CAN':
        Model = model.MT_CAN(nb_filters1, nb_filters2, input_shape)
        Model.load_weights(model_weights)
    
        dXsub = preprocess_raw_video(video_path, dim=36)
        dXsub_len = (dXsub.shape[0] // n_frame)  * n_frame
        dXsub = dXsub[:dXsub_len, :, :, :]
    
        yptest = Model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)
        return yptest
    
        # pulse_pred = yptest[0]
        # pulse_pred = detrend(np.cumsum(pulse_pred), 100)
        # [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        # pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
    
        # resp_pred = yptest[1]
        # resp_pred = detrend(np.cumsum(resp_pred), 100)
        # [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
        # resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))
    
        # ########## Plot ##################
        # plt.subplot(211)
        # plt.plot(pulse_pred)
        # plt.title('Pulse Prediction')
        # plt.subplot(212)
        # plt.plot(resp_pred)
        # plt.title('Resp Prediction')
        # plt.show()
        
    elif model_type=='TS_CAN':
        Model = model.TS_CAN(n_frame, nb_filters1, nb_filters2, input_shape)
        Model.load_weights(model_weights)
    
        dXsub = preprocess_raw_video(video_path, dim=36)
        dXsub_len = (dXsub.shape[0] // n_frame)  * n_frame
        dXsub = dXsub[:dXsub_len, :, :, :]
        yptest = Model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)
        #print(yptest.shape)
        #pulse_pred = yptest
        return yptest

       
        
        # if m_or_p == 'm':
        #     f1 = h5py.File('D:\\chrome\\dataset\\1\\0\\data.hdf5', 'r')
        #     dysub = np.array(f1['pulse']) 
        #     dysub = dysub[:dXsub_len]
        #     result = calculate_metric(pulse_pred,dysub)
        #     print(result)
        
        
        '''
        pulse_pred = detrend(np.cumsum(pulse_pred), 100)
        [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
        pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
        
        ########## Plot ##################
        plt.subplot(211)
        plt.plot(pulse_pred)
        plt.title('Pulse Prediction')
        '''
    
    elif model_type=='MTTS_CAN':
        Model = model.MTTS_CAN(n_frame, nb_filters1, nb_filters2, input_shape)
        Model.load_weights(model_weights)
    
        dXsub = preprocess_raw_video(video_path, dim=36)
        dXsub_len = (dXsub.shape[0] // n_frame)  * n_frame
        dXsub = dXsub[:dXsub_len, :, :, :]
        yptest = Model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)
        return yptest
        
        
    elif model_type=='CAN_3D':
        Model = model.CAN_3D(n_frame, nb_filters1, nb_filters2, input_shape_1)
        Model.load_weights(model_weights)
    
        dXsub = preprocess_raw_video(video_path, dim=36)
        
        num_window = dXsub.shape[0] - (n_frame + 1)
        
        dXsub = np.array([dXsub[f:f+n_frame,:,:,:] for f in range(num_window)])\
                .reshape(-1,n_frame,input_shape[0],input_shape[1],6)
    
        dXsub = np.swapaxes(dXsub, 1, 3) # (-1, 36, 36, 10, 6)
        dXsub = np.swapaxes(dXsub, 1, 2) # (-1, 36, 36, 10, 6)
        yptest = Model.predict((dXsub[:, :, :, :,:3], dXsub[:, :, :,:,-3:]), batch_size=batch_size, verbose=1)
        return yptest
    
        
    elif model_type=='MT_CAN_3D':
        Model = model.MT_CAN_3D(n_frame, nb_filters1, nb_filters2, input_shape)
        Model.load_weights(model_weights)
    
        dXsub = preprocess_raw_video(video_path, dim=36)
        num_window = dXsub.shape[0] - (n_frame + 1)
        
        dXsub = np.array([dXsub[f:f+n_frame,:,:,:] for f in range(num_window)])\
                .reshape(-1,n_frame,input_shape[0],input_shape[1],6)
    
        dXsub = np.swapaxes(dXsub, 1, 3) # (-1, 36, 36, 10, 6)
        dXsub = np.swapaxes(dXsub, 1, 2) # (-1, 36, 36, 10, 6)
        yptest = Model.predict((dXsub[:, :, :, :,:3], dXsub[:, :, :,:,-3:]), batch_size=batch_size, verbose=1)
        return yptest
    
    elif model_type=='Hybrid_CAN':
        Model = model.Hybrid_CAN(n_frame, nb_filters1, nb_filters2, input_shape_1, input_shape_2)
        Model.load_weights(model_weights)
        dXsub = preprocess_raw_video(video_path, dim=36)
        num_window = dXsub.shape[0] - (n_frame + 1)
        
        dXsub = np.array([dXsub[f:f+n_frame,:,:,:] for f in range(num_window)])\
                .reshape(-1,n_frame,input_shape[0],input_shape[1],6)
    
        dXsub = np.swapaxes(dXsub, 1, 3) # (-1, 36, 36, 10, 6)
        dXsub = np.swapaxes(dXsub, 1, 2) # (-1, 36, 36, 10, 6)
        yptest = Model.predict((dXsub[:, :, :, :,:3], np.average(dXsub[:, :, :,:,-3:],axis=-2)), batch_size=batch_size, verbose=1)
        return yptest
    
    
    elif model_type=='MT_Hybrid_CAN':
        Model = model.MT_Hybrid_CAN(n_frame, nb_filters1, nb_filters2, input_shape_1, input_shape_2)
        Model.load_weights(model_weights)
        dXsub = preprocess_raw_video(video_path, dim=36)
        num_window = dXsub.shape[0] - (n_frame + 1)
        
        dXsub = np.array([dXsub[f:f+n_frame,:,:,:] for f in range(num_window)])\
                .reshape(-1,n_frame,input_shape[0],input_shape[1],6)
    
        dXsub = np.swapaxes(dXsub, 1, 3) # (-1, 36, 36, 10, 6)
        dXsub = np.swapaxes(dXsub, 1, 2) # (-1, 36, 36, 10, 6)
        yptest = Model.predict((dXsub[:, :, :, :,:3], np.average(dXsub[:, :, :,:,-3:])), batch_size=batch_size, verbose=1)
        return yptest
    
    else:
        raise ValueError('模型类型错误')
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    

    

  
if __name__=='__main__':
    
    test_txt_path = './test.txt'
    #data_dir = 'E:/dataset/cohface/cohface/cohface/'
    data_dir = 'F:/projectz/dataset/0cohface/cohface/cohface/'
    video_list = []
    with open(test_txt_path,encoding='utf8') as f:
        for i in f.readlines():
            video_list.append(data_dir + i.replace('\n', ''))
    
    model_weight = './Cohface-T-Huber.hdf5'
    m_or_p = 'm'
    model_type   = 'CAN_3D'
    HR = []
    HR0 = []
    #y_pre_r = []
    #y_true_r = []
   
    
    for path in video_list:
        video_path = path + '.avi'
        label_path = path + '.hdf5'  ###标签文件后缀
        yptest = pre(model_weight,model_type,video_path,m_or_p)
        yptest=np.average(yptest,axis=-1)
        # print('yptest的值',yptest)
        # print('ypest.shape',yptest.shape)
        
        #######################################双输出模型################################


        #数据集一
        #______________________________________________________________________
        # dXsub_len = yptest[0].shape[0]
        # f1 = h5py.File(label_path, 'r')
        # dysub = np.array(f1['pulse'])
        # dysub = resample(dysub,dXsub_len)
        # dysub = dysub[:dXsub_len]

        # dXsub_len = yptest[0].shape[0]
        # print('yptest[0]:',yptest[0])
        # print('dXsub_len:',dXsub_len)
        # dysub = get_label(label_path)
        # dysub = dysub[:dXsub_len]
        # # y_pre_pulse.append(yptest[0])
        # # y_true_pulse.append(dysub)


        
        # result = calculate_metric(yptest[0], dysub)
        # print(result)
        # HR.append(result[-1])
        # HR0.append(result[-2])
        
        # #drsub = np.array(f1['respiration'])
        # #drsub = drsub[:dXsub_len]
        # #y_pre_r.append(yptest[1])
        # #y_true_r.append(drsub)
        
        
        # #dysub = get_label(label_path)
        # #dysub = dysub[:dXsub_len]
        # #y_pre_pulse.append(yptest[0])
        # #y_true_pulse.append(dysub)





        
        
        #######################################单输出##################################


        #数据集一
        #______________________________________________________________________
        dXsub_len = yptest.shape[0]
        f1 = h5py.File(label_path, 'r')
        dysub = np.array(f1['pulse'])
        dysub = resample(dysub,dXsub_len)
        dysub = (dysub-np.mean(dysub))/np.std(dysub)
        dysub = dysub[:dXsub_len]
        #______________________________________________________________________





        #数据集二
        #______________________________________________________________________
        # dXsub_len = yptest.shape[0]
        # # print('yptest[0]:',yptest[0])
        # # print('dXsub_len:',dXsub_len)
        # dysub = get_label(label_path)
        # dysub = dysub[:dXsub_len]
        # # y_pre_pulse.append(yptest[0])
        # # y_true_pulse.append(dysub)
        #—————————————————————————————————————————————————————————————————————————




        # plt.rcParams['font.sans-serif']=['SimHei']
        # plt.rcParams['axes.unicode_minus']=False
        # plt.subplot(211)
        # plt.plot(yptest)
        # plt.title('Visualization of estimated rPPG signals')
        # plt.subplot(211)
        # plt.plot(dysub,color='r')
        # # plt.legend(["predict", "ground_truth"])
        # #plt.title('真实心率')
        # plt.show()
        

        result = calculate_metric(yptest, dysub)
        #print(result)
        print("   MAE    ：",result[0],end="\n")
        print("   RMAE   ：",result[1],end="\n")
        print(" pearson  ：",result[2],end="\n")
        print(" meanSNR  ：",result[-3],end="\n")
        print("   HR0    : ",result[-2],end="\n")
        print("   HR     : ",result[-1],end="\n")
        print('---------------------------------------------------------------------------------------')
        
        HR.append(result[-1])
        HR0.append(result[-2])
        
        


        '''
        dXsub_len = yptest.shape[0]
        f1 = h5py.File(label_path, 'r')
        dysub = np.array(f1['pulse']) 
        dysub = dysub[:dXsub_len]
        y_pre_pulse.append(yptest)
        y_true_pulse.append(dysub)
        
        #dysub = get_label(label_path)
        #dysub = dysub[:dXsub_len]
        #y_pre_pulse.append(yptest)
        #y_true_pulse.append(dysub)
        
        '''
HR = np.array(HR).reshape(-1,)
HR0 = np.array(HR0).reshape(-1,)  
   

MAE = np.mean(np.abs(HR - HR0))
RMSE = np.sqrt(np.mean(np.square(HR - HR0)))
pearson = np.corrcoef(HR, HR0) 
#meanSNR = np.nanmean(mySNR)


print("   Final MAE   :",MAE,end="\n")
print("   Final RMAE  :",RMSE,end="\n")
print(" Final pearson :",pearson,end="\n")
#print("Final meanSNR：",meanSNR,end="\n")
print("   Final HR0   :",HR0,end="\n")
print("   Final HR    :",HR,end="\n")


#print(MAE, RMSE,pearson ,meanSNR, HR0, HR)    
    
    
    
    