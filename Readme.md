This is the preprocess and network training repositroy.

test folder: different test files. Can be deleted or modified as needed.

preprocess.ipynb: Used to convert the data from synsense speck camera (dobot) to previous NeuroTac data format.

heatmap.py: Check the heatmap of the data (after preprocess.ipynb)

input_process.py: Block the data into different grids (default 16*16)

parallel_preprocess.py: Parallelly process the data. Data after this process can be directly sent to the network.

Experiment_Data: Spike raster, heatmap and other info will be stored here after hearmap.py process.

snn folder: Contains 2 networks (fully-connected and SCNN). Models can be saved in the models folder.