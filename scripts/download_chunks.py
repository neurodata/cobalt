import os
from ndmulticore import parallel
import numpy as np
import tifffile as tf
from tifffile import imsave, imread
from ndex.ndpull import ndpull
from ndex.ndpull import boss_resources

#def download_slices(config, save_path):
#
#    collection = config['shared']['collection']
#    experiment = config['shared']['collection']
#    channels = config['preprocessing']['channels'].split(',')
#
#    # see examples/neurodata.cfg.example to generate your own
#    config_file = 'neurodata.cfg'
#
#    # print metadata
#    meta = boss_resources.BossMeta(collection, experiment, channel)
#    token, boss_url = ndpull.get_boss_config(config_file)
#    args = ndpull.collect_input_args(
#        collection, experiment, channel, config_file, full_extent=True, res=0, outdir='./')
#    # returns a namespace as a way of passing arguments
#    result, rmt = ndpull.validate_args(args)
#
#    rmt = boss_resources.BossRemote(boss_url, token, meta)
#
#    # downloads the data
#    ndpull.download_slices(result, rmt)

def multicore_handler(data, coords, channel, save_path):
    plog_file = open('progress.log', 'a')
    plog_file.write('Processing {} {} {}\n'.format(coords[0], coords[1], coords[2]))

    z_start, y_start, x_start = coords
    fname2 = 'z_{}_y_{}_x_{}'.format(z_start, y_start, x_start)
    fpath = '{}/{}_{}.tiff'.format(save_path,fname2, channel[0])
    fpath2 = '{}/{}_{}.tiff'.format(save_path,fname2, channel[1])
    imsave(fpath, data[channel[0]].astype(np.uint16))
    imsave(fpath2, data[channel[1]].astype(np.uint64))


def main():
    #config = sys.argv[1]
    #config_p = configparser.ConfigParser()
    #config_p.read(config)
    save_path = './process_folder'
    config_file = 'neurodata.cfg'
    print("starting parallel process")
    parallel.run_parallel(config_file, multicore_handler, cpus=10, save_path=save_path)

if __name__ == "__main__":
    main()


