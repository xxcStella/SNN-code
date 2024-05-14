# process input data from experiments
# requirements: 
# 1, edges of camera and markers are parallel
# 2, see pictures for details

import numpy as np
import pickle
import matplotlib.pyplot as plt
import time

global_pretime = 0
global_postime = 0

def find_clip_time(data_matrix, sum1_lim=1, sum2_lim=1):
    """find edges of the spikes.png so that the pretime and postime can be accurate;
    This is due to the robot and sensor--the spike period can shift gradually, making
    hard truncation impossible
    """
    flag1 = 0
    flag2 = 0
    pretime = 240
    postime = 900
    for t_step in range(0, 2000): # range中的时间决定了t_len的值
        sum1 = 0
        sum2 = 0
        for iy in range(data_matrix.shape[1]):
            for ix in range(data_matrix.shape[0]):
                for t in data_matrix[ix, iy]:
                    if t == t_step:
                        sum1 += 1
                    if t == 2000 - t_step + 0:
                        sum2 += 1
        if sum1 >= sum1_lim and flag1 == 0:
            pretime = t_step
            # print('pretime: ', pretime)
            flag1 = 1
        if sum2 >= sum2_lim and flag2 == 0:
            postime = 2000 - t_step + 0
            # print('postime: ', postime)
            flag2 = 1
        if flag1 == 1 and flag2 == 1:
            break

    # if postime - pretime > 380:     # 防止前面噪音太多导致误判，两者差超过阈值自动舍弃pretime的值
    #     pretime = postime - 380
    
    global global_pretime
    global global_postime
    global_pretime = pretime
    global_postime = postime

    print(pretime, postime)

    return pretime, postime  

def clip_pretime(data, t_len, left_up_x=0, left_up_y=0, right_down_x=127, right_down_y=127):
    """In experiments, pre-time is possible, use this function to clip the pre-time.
       the pre-time in this one is 250 ms.
    
    Args:
    data: original data
    file_path: used to save the new data, which is actually useless in the new version.
    coordinates: you just need to clip data within this area since in latter process you only 
            process data in this area.This can speed up.

    Return:
    data: the new data, 81*1200
    """ 
    data_matrix = data[left_up_x:right_down_x+1, left_up_y:right_down_y+1]
    pretime, postime = find_clip_time(data_matrix)

    for iy in range(right_down_y - left_up_y + 1):
        for ix in range(right_down_x - left_up_x + 1):
            tmp = []
            for e in data[iy + left_up_y, ix + left_up_x]:
                if e>pretime and e<=postime and e-pretime<t_len: # 保证不会超过时长限制
                    tmp.append(e-pretime)

            data[iy + left_up_y, ix + left_up_x] = tmp

    return data


def marker_block(block_data, t_len, temporal_compress=False, compress_time=500):
    """Each block contains h*l pixels representing one white marker (neuron), this function
       records all spike time of all pixels in one block. And record the information in 
       1*t_len neuron_spike.
    
    Args:
    block_data (ndarray): h*l shape, contains several pixels of the camera
    t_len: length of the experiment time, unit: ms. Equals to: (postime - pretime)
    temporal_compress: default False. Whether normalize the time length to a fixed value. Original \
                time length is obtained by clip_pretime function. Therefore, each piece of data is \
                accomodated.
    
    Return:
    neuron_spike (ndarray): 1*t_len, if element==0, then there is no spike in the corresponding
        time; if element>0, then there is element(s) spikes in the corresponding time
    e.g. neuron_spike=[0,1,0,3,0], there is 1 spike in 2nd ms, 3 spikes in 4th ms.
    """

    h, l = block_data.shape


    if temporal_compress==False: # traditional process (no compress time)
        neuron_spike = np.zeros((1, t_len), dtype=np.int64) # unit of t_len: ms

        for i in range(h):
            for j in range(l):
                if block_data[i,j] != []:
                    for time_stamp in block_data[i,j]:
                        # if time_stamp >=1200:
                        #     print("error, ", time_stamp)
                        neuron_spike[0, time_stamp-1] += 1
        return neuron_spike
    
    if temporal_compress==True:  # compress time
        neuron_spike_compress = np.zeros((1, compress_time), dtype=np.int64) # 500 is the compressed time length
        neuron_spike = np.zeros((1, global_postime - global_pretime + 1), dtype=np.int64)
        for i in range(h):
            for j in range(l):
                if block_data[i,j] != []:
                    for time_stamp in block_data[i,j]:
                        neuron_spike[0, time_stamp-1] += 1
        ratio = (global_postime - global_pretime + 1) / compress_time
        if ratio < 1:
            index_map = np.minimum(np.ceil(np.arange(neuron_spike.shape[1])/ratio).astype(int), neuron_spike_compress.shape[1]-1)
        else:
            index_map = np.minimum(np.floor(np.arange(neuron_spike.shape[1])/ratio).astype(int), neuron_spike_compress.shape[1]-1) 
        for idx in range(neuron_spike.shape[1]):
            neuron_spike_compress[:, index_map[idx]] += neuron_spike[:, idx]
        return neuron_spike_compress


def crop(data, t_len, temporal_compress=False, compress_time=500, left_up_x=0, left_up_y=0, right_down_x=127, right_down_y=127): # 第一批10组的参数：left_up_x=49, left_up_y=71, right_down_x=158, right_down_y=180
    """Crop the whole camera pixel matrix to get a small pixel matrix corresponding to
       the white marker matrix. Calculate each marker spike response. Then stack them
       get a 81*t_len matrix.
    
    Args:
# Revise them if needed.
    data (ndarray): spike record of experiments. The index of the pixel starts with 0.
               240 is height, 180 is width. The index increases first by width then by 
               height.
    t_len: same as in marker_block function
    others: coordinates. All coordinates starts with 0.
    
    Return:
    input (ndarray): matrix. x aixs is time (t_len). y axis is neuron index.
        the first neuron (the first block) is the first row of the input. The index 
        of neuron increases first by x axis then by y axis.
    """

# Revise them if needed.
    x_num = 16  # number of the blocks in x axis direction. Revise them if needed.
    y_num = 16  # number of the blocks in y axis direction
    # x_num = 20
    # y_num = 20

    block_x_len = int((right_down_x - left_up_x + 1) / x_num)  # Each block contains block_x_len pixels in x axis direction
    block_y_len = int((right_down_y - left_up_y + 1) / y_num)

    if temporal_compress==False:
        input = np.zeros([x_num * y_num, t_len])
    else:
        input = np.zeros([x_num * y_num, compress_time])
    print(input.shape)

    for i in range(y_num):
        for j in range(x_num):
            block_data = data[left_up_y + i*block_y_len : left_up_y + (i+1)*block_y_len, \
                left_up_x + j*block_x_len : left_up_x + (j+1)*block_x_len]

            input[i*x_num + j] = marker_block(block_data, t_len, temporal_compress, compress_time)

    return input

"""E.g.
a=np.array([[[1],   [2],   [3,5]], 
            [[1,2], [3,4], []],
            [[6],   [7],   [8,9]]])

>>> marker_block(a[1:3,1:3],10)
>>> array([[0, 0, 1, 1, 0, 0, 1, 1, 1, 0]], dtype=int64)

>>> input = crop(a,10)
>>> input
>>> array([[0., 0., 1., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 1., 0.]])
"""


def plot_input(input_response_psth):
    """plot the input spike figure similar to which is in TouchSim simulation
        (can be used to compare input vs output vs target)

    Args:
        input_response_psth (ndarray 2d): psth of the experiment input
    """
    input_t = []
    input_neuron = []

    for i, neuron in enumerate(input_response_psth):
        for t in range(np.size(neuron)):
            if neuron[t]>0.5:
                input_neuron.append(i)
                input_t.append(t)

    plt.plot(input_t, input_neuron, '.', label='input')
    plt.xlabel('time [ms]')
    plt.ylabel('neuron')
    plt.legend()
    plt.title('input spikes after process')
    plt.show()



# # Revise them if needed.
# data_path = 'F:/Files/PhD/Braille/Code/Events_Examples/Experiment_Data'
# file_name = 'taps_trial_252_pose_17_events_on'
# # input_name = 'input_trial_0_pose_0.npy'

# start = time.perf_counter()
# with open(data_path + '/' + file_name, 'rb') as file:
#     data = pickle.load(file)

# data = clip_pretime(data, 700)

# input_psth = crop(data, 700).astype(np.int64)

# # print(type(input_psth[0][0]))
# plot_input(input_psth)

# np.save(data_path + '/' + input_name, input_psth)
# print(time.perf_counter() - start)












