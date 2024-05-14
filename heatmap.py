import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
import samna


def main():
  # Set flags for figures
  disp_heatmap = True
  disp_spikes = True
  ssim_calc = True
 
  # Font sizes
  TITLE_SIZE = 20
  AXES_SIZE = 16
  TICKS_SIZE = 12

  # Data directory
  data_dir_name = 'Experiment_Data'
  home_dir = '/home/xxc/PhD/synsense/Code/' # TODO: you will need to change this to the location of the Event_Example folder on your computer
  data_dir = home_dir + data_dir_name
  fig_save_dir = home_dir + data_dir_name + '/figures'

  # Make folder to save figure if one doesnt already exist
  if not os.path.exists(fig_save_dir):
    os.mkdir(fig_save_dir)

  

  print('#### Pose, Tap ####')
  
  # Load data
  filename = data_dir + '/taps_trial_0_pose_0.pickle'
  infile = open(filename,'rb')
  data = pickle.load(infile)      
  infile.close()


#################################################################### HEATMAP #########################################################################################
  if disp_heatmap:
    x_size = int(data.shape[1])
    y_size = int(data.shape[0])
    
    #create array for heatmap
    data_heatmap = np.empty((y_size, x_size))
    
    # moving along x in each y count number of event timestamps in list
    for y in range(y_size):
      for x in range(x_size): 
        data_heatmap[y][x] = (len(data[y][x]))
    #reshape data to original shape
    data_heatmap = np.reshape(data_heatmap,(y_size,-1))

    plt.imshow(data_heatmap,vmax=15)
    plt.colorbar()
    plt.savefig(fig_save_dir + '/heatmap_pose_0_tap_0.png')
    plt.show()

  
  if disp_spikes:
    spikes_data = data.flatten()
    neuron_idx = 0

    for spike_train in spikes_data:
      if spike_train != []:
        y = np.ones_like(spike_train) * neuron_idx
        plt.plot(spike_train, y, 'k|', markersize=0.7)
      neuron_idx +=1

    plt.ylim = (0, len(spikes_data))
    plt.ylabel("neuron")
    plt.xlabel("time (ms)")
    plt.title("Spike data")
    plt.setp(plt.gca().get_xticklabels(), visible=True)
    plt.savefig(fig_save_dir + '/spikes.png')
    plt.show()

#################################################################### SSIM #########################################################################################
  if disp_heatmap & ssim_calc:
    zeros_heatmap = np.zeros((y_size,x_size))
    mssim_score, mssim_local_map = ssim(data_heatmap, zeros_heatmap, gaussian_weights = True, full = True, data_range=(max(np.max(data_heatmap),np.max(zeros_heatmap))-min(np.min(data_heatmap),np.min(zeros_heatmap))), use_sample_covariance = False)
    print(mssim_score)

    # rect = plt.Rectangle((200, 100), 200, 100, fill=False, edgecolor='r', linewidth=1)
    # mssim_local_map.add_patch(rect)
    
    plt.imshow(mssim_local_map)
    plt.savefig(fig_save_dir + '/mssim_local_map_pose__0_tap_0.png')
    plt.show()



 
if __name__ == '__main__':
    main()