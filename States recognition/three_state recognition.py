# N-state recognition
import numpy as np
import function_box as f
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq,kmeans,whiten
from tqdm import tqdm
import os

dt = 0.05
dt_sub = 0.05
k = 0
# Basic information setting
path_segmentation_3s = '../Dataset/41.8 degree/three_states/'
path_figure_saving = '../Figures_seg_three_states/'
path_txt_saving = '../States_information/'
if not os.path.exists(path_txt_saving):
    os.makedirs(path_txt_saving)
if not os.path.exists(path_figure_saving):
    os.makedirs(path_figure_saving)
file_names_3s = os.listdir(path_segmentation_3s)
sampling_fs = 100000
theme_color = 'black'

# Data input, separation and plot
# In this part, two main categories have been classified, which are N and F states
k=k
path_file_temp = path_segmentation_3s + file_names_3s[k]
print(path_file_temp)
data = f.load_data(path_file_temp)[:,1]

data_new = []
dt = dt
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
fit_curve = np.array([])
for i in range(len(result)):
    if result[i] == 0:
        fit_curve = np.concatenate((fit_curve,unit_s0))
    elif result[i] == 1:
        fit_curve = np.concatenate((fit_curve,unit_s1))
plt.figure()
f.data_plot(data,linewidth=1,alpha=0.5,color=theme_color,ifplot='off',multi_fig='on')
f.data_plot(fit_curve,linewidth=2,color='red',ifplot='on',multi_fig='on')

# N state removal
# Here, we need to remove all the N state's data, only remind E and F state
if len(fit_curve) != len(data):
    len_delta = len(data) - len(fit_curve)
    fit_curve_suplementary = np.ones(len_delta)*fit_curve[-1]
    fit_curve = np.concatenate((fit_curve,fit_curve_suplementary))

index = np.arange(len(data))
data_with_index = np.vstack((data,index))
fit_curve_with_index = np.vstack((fit_curve,index))
removal_index = np.array([])
print('Removal data index generating:')
pbar = tqdm(total=len(fit_curve))
for i in range(len(fit_curve)):
    if fit_curve[i] == np.sort(ave_set)[1]:
        removal_index = np.append(removal_index,i)
    pbar.update(1)
pbar.close()
removal_index = removal_index.astype(int)
removal_data = np.array([])
print('Removal data generating:')
pbar = tqdm(total = len(removal_index))
for i in range(len(removal_index)):
    removal_data = np.append(removal_data,data[removal_index[i]])
    pbar.update(1)
pbar.close()
# removal_fit_curve = np.array([])
# for i in range(len(removal_index)):
#     removal_fit_curve = np.append(removal_fit_curve,fit_curve[removal_index[i]])
removal_data_with_index = np.vstack((removal_data,removal_index))
data_truncated_with_index = np.delete(data_with_index,removal_index,1)
# removal_fit_curve_with_index = np.vstack((removal_fit_curve,removal_index))

f.data_plot(data_truncated_with_index[0,:],linewidth=1,alpha=0.5,color=theme_color,ifplot='on',multi_fig='off')


# E and F state recognition
# Here, as for the truncated data sequence, we need to conduct a 2-state recognition again
# This time, the purpose is recognition of E and F state
data_new_sub = []
dt_sub = dt_sub
interval = dt_sub*sampling_fs
data_sub = data_truncated_with_index[0,:]
for i in range(int(len(data_sub)/interval)):
    temp = data_sub[int(i*interval):int((i+1)*interval-1)]
    data_new_sub.append(temp)
im_var = np.arange(int(len(data_sub)/interval))*dt
X = data_new_sub
X_whiten=whiten(X)
centroids,_=kmeans(X_whiten,2)
result_sub,_=vq(X_whiten,centroids)
state_0_sub = np.array([])
state_1_sub = np.array([])
for i in range(len(result_sub)):
    if result_sub[i] == 0:
        state_0_sub = np.concatenate((state_0_sub,data_sub[int(i*interval):int((i+1)*interval-1)]))
    if result_sub[i] == 1:
        state_1_sub = np.concatenate((state_1_sub,data_sub[int(i*interval):int((i+1)*interval-1)]))
ave_s0_sub = np.average(state_0_sub)
ave_s1_sub = np.average(state_1_sub)

ave_set_sub = np.array([ave_s0_sub,ave_s1_sub])
max_tag = np.argmax(ave_set_sub)
min_tag = np.argmin(ave_set_sub)
for i in range(len(result_sub)):
    if result_sub[i] == max_tag:
        result_sub[i] = 1
    elif result_sub[i] == min_tag:
        result_sub[i] = 0
unit_s0_sub = np.sort(ave_set_sub)[0]*np.ones(int(interval))
unit_s1_sub = np.sort(ave_set_sub)[1]*np.ones(int(interval))
fit_curve_sub = np.array([])
for i in range(len(result_sub)):
    if result_sub[i] == 0:
        fit_curve_sub = np.concatenate((fit_curve_sub,unit_s0_sub))
    elif result_sub[i] == 1:
        fit_curve_sub = np.concatenate((fit_curve_sub,unit_s1_sub))

if len(data_sub) != len(fit_curve_sub):
    len_delta = len(data_sub) - len(fit_curve_sub)
    fit_curve_suplementary = np.ones(len_delta)*fit_curve_sub[-1]
    fit_curve_sub = np.concatenate((fit_curve_sub,fit_curve_suplementary))


f.data_plot(data_sub,linewidth=1,alpha=0.5,color=theme_color,ifplot='off',multi_fig='on')
f.data_plot(fit_curve_sub,linewidth=2,color='red',ifplot='on',multi_fig='on')
fit_curve_sub_with_index = np.vstack((fit_curve_sub,data_truncated_with_index[1,:]))

# data recombination
print('Data recobination processing:')
pbar = tqdm(total=len(fit_curve_sub))
for i in range(len(fit_curve)):
    if np.any(fit_curve_sub_with_index[1,:] == i):
        temp_p = np.where(fit_curve_sub_with_index[1,:]==i)
        fit_curve[i] = fit_curve_sub_with_index[0,temp_p]
        pbar.update(1)
pbar.close()
plt.figure()
f.data_plot(data,linewidth=1,alpha=0.5,color=theme_color,ifplot='off',multi_fig='on')
f.data_plot(fit_curve,linewidth=2,color='red',ifplot='off',multi_fig='on')
plt.savefig(path_figure_saving+file_names_3s[k][:-4]+'.png',dpi=100)
plt.show()
ave_set = np.sort(np.array([np.sort(ave_set)[1],np.sort(ave_set_sub)[1],np.sort(ave_set_sub)[0]]))
print('ave should be:',ave_set)
print("Data reformation and saving:")
state = np.array([])
pbar = tqdm(total=len(fit_curve))
for i in range(len(fit_curve)):
    if fit_curve[i] == ave_set[0]:
        state = np.append(state,0)
    elif fit_curve[i] == ave_set[1]:
        state = np.append(state,1)
    elif fit_curve[i] == ave_set[2]:
        state = np.append(state,2)
    pbar.update(1)
pbar.close()
print(len(data))
print(len(fit_curve))
print(len(state))
txt_info = np.vstack((data,fit_curve,state)).T
print('Data saving...')
np.savetxt(path_txt_saving+file_names_3s[k][:-4]+'_stateinfo.txt',txt_info,fmt='%.8f',delimiter=',')
