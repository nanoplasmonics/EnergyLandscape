import numpy as np
import function_box as f
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,kmeans,whiten
from tqdm import tqdm
import os

path_segmentation_2s = '../Dataset/41.8 degree/two_states/'
path_figure_saving = '../Figures_seg_two_states/'
if not os.path.exists(path_figure_saving):
    os.makedirs(path_figure_saving)
path_txt_saving = '../States_information/'
if not os.path.exists(path_txt_saving):
    os.makedirs(path_txt_saving)
file_names_2s = os.listdir(path_segmentation_2s)
sampling_fs = 100000
theme_color = 'black'
k = 0
if file_names_2s[k][0]!='.':
    print(file_names_2s[k])
    path_file_temp = path_segmentation_2s + file_names_2s[k]
    data = f.load_data(path_file_temp)[:,1]
    data_new = []
    dt = 0.1
    interval = dt*sampling_fs
    for i in range(int(len(data)/interval)):
        temp = data[int(i*interval):int((i+1)*interval-1)]
        data_new.append(temp)
    im_var = np.arange(int(len(data)/interval))*dt
    X = data_new
    X_whiten=whiten(X)
    centroids,_=kmeans(X_whiten,2)
    result,_=vq(X_whiten,centroids)
    state_0 = np.array([])
    state_1 = np.array([])

    for i in range(len(result)):
        if result[i] == 0:
            state_0 = np.concatenate((state_0,data[int(i*interval):int((i+1)*interval-1)]))
        if result[i] == 1:
            state_1 = np.concatenate((state_1,data[int(i*interval):int((i+1)*interval-1)]))
    ave_s0 = np.average(state_0)
    ave_s1 = np.average(state_1)

    ave_set = np.array([ave_s0,ave_s1])
    max_tag = np.argmax(ave_set)
    min_tag = np.argmin(ave_set)
    for i in range(len(result)):
        if result[i] == max_tag:
            result[i] = 1
        elif result[i] == min_tag:
            result[i] = 0
    unit_s0 = np.sort(ave_set)[0]*np.ones(int(interval))
    unit_s1 = np.sort(ave_set)[1]*np.ones(int(interval))
    unit_state0 = np.ones(int(interval))*0
    unit_state1 = np.ones(int(interval))*1
    print(unit_state1)
    print('File name is:',file_names_2s[k])
    print('The mean value for lower state is:',np.sort(ave_set)[0])
    print('The mean value for  upper state is:',np.sort(ave_set)[1])
    fit_curve = np.array([])
    state = np.array([])
    for i in range(len(result)):
        if result[i] == 0:
            fit_curve = np.concatenate((fit_curve,unit_s0))
            state = np.concatenate((state,unit_state0))
        elif result[i] == 1:
            fit_curve = np.concatenate((fit_curve,unit_s1))
            state = np.concatenate((state,unit_state1))
    print(len(result))
    print(state)
    f.data_plot(data,linewidth=1,alpha=0.5,color=theme_color,ifplot='off',multi_fig='on')
    f.data_plot(fit_curve,linewidth=2,color='red',ifplot='on',multi_fig='on')
    if len(fit_curve) != len(data):
        delta = len(data) - len(fit_curve)
        fit_curve = np.concatenate((fit_curve,np.ones(delta)*fit_curve[-1]))
        state = np.concatenate((state,np.ones(delta)*state[-1]))
    txt_info = np.vstack((data,fit_curve,state)).T

    np.savetxt(path_txt_saving+file_names_2s[k][:-4]+'_stateinfo.txt',txt_info,fmt='%.8f',delimiter=',')


