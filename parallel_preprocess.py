import numpy as np
import pickle
import matplotlib.pyplot as plt
from input_process import *
from shutil import copy
import os
from concurrent.futures import ProcessPoolExecutor

# 多线程加速
data_path = '/home/xxc/PhD/synsense/Data/mg_tap_9positions/'
raw_place = 'format_process'
processed_place = 'input_data'

def process_file(filename, temporal_compress=False, compress_time=350):
    with open(data_path + '/' + raw_place + '/' + filename, 'rb') as file:
        data = pickle.load(file)
    
    print(filename)
    
    data = clip_pretime(data, 2000)
    input_psth = crop(data, 2000, temporal_compress, compress_time).astype(np.int8) # 修改为int8类型，直接减小文件大小，方便运输

    np.save(data_path + '/' + processed_place + '/' + filename[5:-7] + '.npy', input_psth)


def main():
    file_names = os.listdir(data_path + '/' + raw_place)

    with ProcessPoolExecutor(max_workers=20) as executor:
        executor.map(process_file, file_names)

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(time.perf_counter() - start)