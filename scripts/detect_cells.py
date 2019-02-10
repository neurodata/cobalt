import json
from intern.remote.boss import BossRemote
import configparser
import sys
import glob
from sklearn import cluster
import numpy as np
from tqdm import tqdm
import os
import tifffile as tf
import argparse
import csv
from joblib import Parallel, delayed
from scipy.signal import fftconvolve
import skimage
from skimage import morphology, measure, feature
from skimage.morphology import selem
import boss_util
from configparser import ConfigParser


def write_list_to_csv(arr, csv_output_path, open_mode='w'):
    """Given a list, writes it to a CSV file"""
    with open(csv_output_path, open_mode) as csv_file:
        for item in arr:
            csv_file.write(','.join([str(x) for x in item]) + '\n')
            
def get_cells_from_csv(path_to_csv):
    detected_cells = []
    with open(path_to_csv, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            try:
                detected_cells.append([float(i[0]),float(i[1]),float(i[2]),int(i[3])])
            except Exception as e:
                detected_cells.append([float(i[0]),float(i[1]),float(i[2])])
#                 print(e)
    return np.array(detected_cells)

def read_templates(path_to_templates, ncomp=5):
    files = os.listdir(path_to_templates)
    files = np.sort([f for f in files if f.endswith('.tif') or f.endswith('.tiff')])
    templates = []
    for i in np.sort(os.listdir(path_to_templates))[:ncomp]:
        f = path_to_templates + i
        if not f.endswith('.tiff'): continue
        templates.append(tf.imread(f))
    return templates


def match_templates(chunk, identifier, templates, pad_width=16, save_path=None):
    responses = []
    padded_chunk = np.pad(chunk, pad_width, mode='reflect')
    kernel = np.ones((8,8,8))
    feature_map = np.zeros(chunk.shape)
    for template in templates:
        response = fftconvolve(padded_chunk, template, mode="same")[pad_width:-pad_width,pad_width:-pad_width,pad_width:-pad_width]
#        response = suppress_dark_blobs(response)
        response = response *  (response > 0)
        # square all responses to emphasize high responses
        response = np.square(response)
        # now convolve responses with ones 
        # the size of our 3D window.
        # in our case: ones with shape (8,8,8)
        rp = np.pad(response, pad_width, mode='reflect')
        response = fftconvolve(rp, kernel, mode='same')[pad_width:-pad_width,pad_width:-pad_width,pad_width:-pad_width]
        feature_map += response

    # feature we are using is the sum of the power of first 5 components
#    feature_map = np.sum(r_summed, axis=0)
#    print('feature map created')
    if save_path is not None:
        # linearly rescale feature map to 16-bit
        # for saving and viz purposes
        max_val = 2**16-1
        fmap_rescaled = (feature_map - feature_map.min())/(feature_map.max() - feature_map.min()) * max_val
        tf.imsave('{}/{}_feature_map2.tiff'.format(save_path, identifier), fmap_rescaled.astype('uint16'))
#        return (feature_map - feature_map.min())/(feature_map.max()-feature_map.min())
    return feature_map

def suppress_dark_blobs(feature_map, radius=10):
    mask = feature_map <= 0
    mask2 = morphology.binary_dilation(mask, selem=selem.ball(radius))
    return feature_map * np.abs(mask2 - 1)

def get_detections(feature_map, atlas, identifier, threshold, erosion_radius=30, verbose=False):
    bin_img = feature_map > threshold

    # mask out areas outside of atlas
    atlas_mask = atlas > 0
    ## ignore areas close to the boundary of the atlas
    if not np.all(atlas_mask):
        num_iter = 5
        for i in range(num_iter):
            atlas_mask = morphology.binary_erosion(atlas_mask, selem=selem.ball(erosion_radius/num_iter))
    bin_img = bin_img * atlas_mask

    labeled_img, num = measure.label(bin_img, return_num=True)
    if verbose: print('{} detected cells in {} before processing'.format(num, identifier))

    regionprops = measure.regionprops(labeled_img, feature_map)
    centroids = []
    for i in regionprops:
        if i.area > 1e6:
            if verbose: print('too big. size: {}'.format(i.area))
            continue
        p = list(i.centroid)
        centroids.append(p)
    if len(centroids) is 0: return centroids
    elif len(centroids) is 1:
        centroids = np.array(centroids).reshape(1,-1)
#    centroids = prune_clusters(centroids)
    centroids_new = []
    for i in centroids:
        p = [i[0], i[1], i[2], np.squeeze(atlas[int(i[0]),int(i[1]),int(i[2])])]
        centroids_new.append(p)
    if verbose: print('{} detected cells in {} after processing'.format(len(centroids), identifier))
    return centroids_new

def get_paired_files(path_to_chunks):
    # get centroids over whole dataset
    # threshold = 1.0
    f = np.sort(glob.glob('{}/z*.tif*'.format(path_to_chunks)))
    #print(f)
    files = []
    for i in f:
        files.append(i)
    paired_files = [files[i:i+2] for i in range(0,len(files)-1,2)]
    return np.array(paired_files)
    
def get_max_intensity(path_to_chunks, verbose=False):
    # find max intensity over all chunks
    max_val = 0
    files = os.listdir(path_to_chunks)
    files = [f for f in files if f.endswith('.tif') or f.endswith('.tiff') and 'atlas' not in f]
    for i in tqdm(files, desc='finding max intensity over all chunks'):
        if verbose: print('processing {}'.format(i))
        img = tf.imread(path_to_chunks + i)
        if max_val < img.max() < 2**16:
            max_val = img.max()
            if verbose: print("maximum value found in chunk {}".format(i))
    return max_val

def prune_clusters(centroids, bandwidth=15, verbose=False):
    clust = cluster.MeanShift(bandwidth=bandwidth).fit(np.array(centroids))
    if verbose:
        print('{} centroids before.\n{} centroids after clustering.'.format(len(centroids), len(clust.cluster_centers_)))
    return clust.cluster_centers_

def detect_cells_in_chunk(paired_file, save_path_c, threshold, templates, erosion_radius=30, max_val=54908,save_path_f=None, verbose=False, del_files=True):
    if verbose: 
        print('path_to_image: {}'.format(paired_file[0]))
        print('path_to_atlas: {}'.format(paired_file[1]))
    img = tf.imread(paired_file[0])
    atlas = tf.imread(paired_file[1])
    x = os.path.basename(paired_file[0])
    x2 = x.split('_')
    z_start = int(x2[1])
    y_start = int(x2[3])
    x_start = int(x2[5])
    identifier = '_'.join(x2[:6])
    atlas_mask = atlas > 0
    if len(np.flatnonzero(atlas_mask)) == 0:
        if verbose: print("chunk {} is outside of the registered atlas".format(identifier))
    else:
        # normalize image
        chunk = img / max_val
        if verbose: print('chunk max: {}'.format(chunk.max()))
        feature_map = match_templates(chunk, identifier, templates, save_path=save_path_f)
        centroids = get_detections(feature_map, atlas, identifier=identifier, threshold=threshold, erosion_radius=erosion_radius, verbose=verbose)
        if len(centroids) == 0: 
            pass
        else:
            # remember that c[3] is the label associated with a centroid
            centroids = [[c[0] + z_start, c[1] + y_start, c[2] + x_start, c[3]] for c in centroids]
            write_list_to_csv(centroids, save_path_c, open_mode='a')
    if del_files:
        os.remove(paired_file[0])
        os.remove(paired_file[1])

def get_labels_for_cells(centroids,rmt,coll,exp,chan):
    labels = []
    voxels = [[int(i) for i in c] for c in centroids]
    for i in tqdm(centroids, desc="getting labels for cells"):
        labels.append(boss_util.get_label(i,rmt,coll,exp,chan))
    return labels

        
def detect_cells(rmt, config):

    shared_params=config['shared']
    params=config['cell_detection']
    coll = shared_params['collection']
    exp = shared_params['experiment']
    chan = params['channel']
    
    chunks = './process_folder/'
    csv_path = './process_folder/{}_{}_{}_detected_cells.csv'.format(shared_params['collection'],shared_params['experiment'],params['channel'])
    json_path = './process_folder/{}_{}_{}_detected_cells.json'.format(shared_params['collection'],shared_params['experiment'],params['channel'])
    feature_path = None
    threshold = 1.0
    erosion_radius = 30
    jobs = 8
    verbose = 0
    path_to_templates = './entangl_templates/'

    max_val_ = get_max_intensity(chunks)
    paired_files = get_paired_files(chunks)
    templates = read_templates(path_to_templates)

    x = Parallel(n_jobs=jobs, max_nbytes=1e6, temp_folder=chunks, verbose=verbose)(delayed(detect_cells_in_chunk)(i, csv_path, threshold, templates, save_path_f=feature_path, erosion_radius=erosion_radius, max_val=max_val_, verbose=verbose) for i in tqdm(paired_files, desc='detecting cells...'))
    
    centroids = get_cells_from_csv(csv_path)
    centroids_t = prune_clusters(centroids[:,:3], verbose=verbose)
    labels = np.array(get_labels_for_cells(centroids_t,rmt,coll,exp,'atlas_50umreg'))
    centroids_tc = np.concatenate((centroids_t, labels[:,None]),axis=1)
    write_list_to_csv(centroids_tc, csv_path.replace('cells','cells_pruned'), open_mode='w')
    boss_util.save_boss_json(coll, exp, chan, csv_path.replace('cells','cells_pruned'), json_path)
    viz_link = None
    with open(json_path, 'r') as fp:
        viz_link = boss_util.get_viz_link(json.load(fp))
    return viz_link

def main():
    config = sys.argv[1]
    config_p = configparser.ConfigParser()
    config_p.read(config)
    rmt = BossRemote(config)
    viz_link = detect_cells(rmt, config_p)
    logfile = config_p['Default']['logfile']
    boss_util.write_to_logfile('cell_detection viz link: {}'.format(viz_link),logfile)


if __name__ == "__main__":
    main()
