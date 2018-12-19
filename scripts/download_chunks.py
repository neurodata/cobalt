from ndmulticore import parallel
import numpy as np
import tifffile as tf
from tifffile import imsave, imread

def multicore_handler(data, coords, channel, save_path):
    plog_file = open('progress.log', 'a')
    plog_file.write('Processing {} {} {}\n'.format(coords[0], coords[1], coords[2]))

    z_start, y_start, x_start = coords
    fname2 = 'z_{}_y_{}_x_{}'.format(z_start, y_start, x_start)
    fpath = '{}/{}_{}.tiff'.format(save_path,fname2, channel[0])
    fpath2 = '{}/{}_{}.tiff'.format(save_path,fname2, channel[1])
    imsave(fpath, data[channel[0]].astype(np.uint16))
    imsave(fpath2, data[channel[1]].astype(np.uint64))



def download_data(config_file):
    save_path = './process_folder'

    print("starting parallel process")
    parallel.run_parallel(config_file, multicore_handler, cpus=10, save_path=save_path)

