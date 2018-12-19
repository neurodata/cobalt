import csv
import numpy as np
import json
import tifffile as tf
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import ChannelResource, ExperimentResource, CoordinateFrameResource

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
                print(e)
                continue
    return np.array(detected_cells)

def get_points_from_json(json_file, layer, return_labels=False):
    state = json.load(open(json_file, 'r'))
    labels = []
    try:
        points = [i['point'] for i in state['layers'][layer]['annotations']]
        if return_labels: 
            labels = [i['segments'][0] for i in state['layers'][layer]['annotations']]
    except:
        points = []
#        points = [i['point'] for i in state['layers']['detected_cells']['annotations']]
    if return_labels: return points, labels
    else: return points

def download_annotations(centroids, rmt,collection, experiment, channel, width=5):
    for i in centroids:
        X = int(i[0])
        Y = int(i[1])
        Z = int(i[2])
        ch_rsc = rmt.get_channel(channel,collection,experiment) 
        x_range = [X-width, X+width]
        y_range = [Y-width, Y+width]
        z_range = [Z-width, Z+width]
        anno = rmt.get_cutout(ch_rsc, 0,x_range,y_range, z_range)
        print(anno.shape)
        tf.imsave('entangl-12r_extras/z_{}_y_{}_x_{}.tiff'.format(Z,Y,X), data=anno)

def mask_out_points(path_to_csv,rmt,collection, experiment, channel):
    """
    csv should have the points in z,y,x format
    """
    new_points = []
    reader = csv.reader(open(path_to_csv, 'r'))
    counter = 0
    for i in reader:
        Z = int(float(i[0]))
        Y = int(float(i[1]))
        X = int(float(i[2]))
        ch_rsc = rmt.get_channel(channel,collection,experiment) 
        x_range = [X, X+1]
        y_range = [Y, Y+1]
        z_range = [Z, Z+1]
        anno = rmt.get_cutout(ch_rsc, 0,x_range,y_range, z_range)
#        print(anno.shape)
        if anno[0] != 0:
            new_points.append(i)
        counter += 1
    return new_points, counter

def generate_labeled_cells_json(cells, labels, voxel_size,linked_layer='atlas'):
    anno = []
    counter = 0
    for i,j in zip(cells,labels):
        x = {
            'type': 'point', 
            'id': str(counter),
            'point': [float(k) for k in i],
            'segments': [
                str(int(float(j)))
            ]
        }
        counter+=1
        anno.append(x)
    cells_layer = {
        'type': 'annotation', 
        'annotations': anno, 
        # may want to change this later for diff voxel sizes
        'voxelSize': voxel_size,
        'linkedSegmentationLayer': linked_layer
    }
    return cells_layer

def generate_cells_json(cells, voxel_size):
    anno = []
    counter = 0
    for i in cells:
        x = {
            'type': 'point', 
            'id': str(counter),
            'point': [float(k) for k in i]
        }
        counter+=1
        anno.append(x)
    cells_layer = {
        'type': 'annotation', 
        'annotations': anno, 
        # may want to change this later for diff voxel sizes
        'voxelSize': voxel_size,
    }
    return cells_layer

def generate_labeled_boss_json(coll, exp, chan, cells, labels, linked_layer="atlas"):
    url = 'boss://https://api.boss.neurodata.io/{}/{}/{}?window=0,3000'.format(coll, exp, chan)
    url2 = 'boss://https://api.boss.neurodata.io/{}/{}/{}'.format(coll, exp, 'atlas_50umreg')
#     reader = csv.reader(open(path_to_csv))
    voxel_size = [5160,5160,5160]
    cells_layer = generate_labeled_cells_json(cells, labels, voxel_size, linked_layer=linked_layer)
    exp_json = {
        "layers": {
            chan: {
              "source": url,
              "type": "image",
              "blend": "additive"
            },
            linked_layer: {
              "source": url2,
              "type": "segmentation",
              "blend": "additive"
            },
            "detected_cells": cells_layer
            
        },
        "navigation": {
            "pose": {
                "voxelSize": voxel_size,
                "voxelCoordinates": [
                  1144.7791748046875,
                  1278.600830078125,
                  751.5
                ]
            }
        },
        "perspectiveZoom": 4.779898969674949
    }
    return exp_json

def generate_boss_json(coll, exp, chan, cells, linked_layer="atlas"):
    url = 'boss://https://api.boss.neurodata.io/{}/{}/{}?window=0,5000'.format(coll, exp, chan)
    url2 = 'boss://https://api.boss.neurodata.io/{}/{}/{}'.format(coll, exp, 'atlas_50umreg')
#     reader = csv.reader(open(path_to_csv))
    voxel_size = [5160,5160,5160]
    tp_layer = generate_cells_json(cells, voxel_size)
    exp_json = {
        "layers": {
            chan: {
              "source": url,
              "type": "image",
              "blend": "additive"
            },
            linked_layer: {
              "source": url2,
              "type": "segmentation",
              "blend": "additive"
            },
            "detected_cells": tp_layer,
            
        },
        "navigation": {
            "pose": {
                "voxelSize": voxel_size,
                "voxelCoordinates": [
                  1144.7791748046875,
                  1278.600830078125,
                  751.5
                ]
            }
        },
        "perspectiveZoom": 4.779898969674949
    }
    return exp_json

def save_cell_detection_json(coll, exp, chan, csv_file, save_path):
    cells = get_cells_from_csv(csv_file)
    exp_json = generate_labeled_boss_json(coll, exp, chan, cells[:,:3][:,::-1], cells[:,-1])
    json.dump(exp_json, open(save_path, 'w'), indent=4)
def save_boss_json(coll, exp, chan, csv_file, save_path):
    cells = get_cells_from_csv(csv_file)
    exp_json = generate_boss_json(coll, exp, chan, cells[:,::-1])
    json.dump(exp_json, open(save_path, 'w'), indent=4)


import argparse
import os
import ndreg
from ndreg import preprocessor, util
import numpy as np
import SimpleITK as sitk
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import time
import requests
import configparser
from downsample.startDownsample import *


dimension = 3
vectorComponentType = sitk.sitkFloat32
vectorType = sitk.sitkVectorFloat32
affine = sitk.AffineTransform(dimension)
identityAffine = list(affine.GetParameters())
identityDirection = identityAffine[0:9]
zeroOrigin = [0] * dimension
zeroIndex = [0] * dimension

dimension = 3
vectorComponentType = sitk.sitkFloat32
vectorType = sitk.sitkVectorFloat32
affine = sitk.AffineTransform(dimension)
identityAffine = list(affine.GetParameters())
identityDirection = identityAffine[0:9]
zeroOrigin = [0] * dimension
zeroIndex = [0] * dimension

ndToSitkDataTypes = {'uint8': sitk.sitkUInt8,
                     'uint16': sitk.sitkUInt16,
                     'uint32': sitk.sitkUInt32,
                     'float32': sitk.sitkFloat32,
                     'uint64': sitk.sitkUInt64}


sitkToNpDataTypes = {sitk.sitkUInt8: np.uint8,
                     sitk.sitkUInt16: np.uint16,
                     sitk.sitkUInt32: np.uint32,
                     sitk.sitkInt8: np.int8,
                     sitk.sitkInt16: np.int16,
                     sitk.sitkInt32: np.int32,
                     sitk.sitkFloat32: np.float32,
                     sitk.sitkFloat64: np.float64,
                     }

# Boss Stuff:
def setup_experiment_boss(remote, collection, experiment):
    """
    Get experiment and coordinate frame information from the boss.
    """
    exp_setup = ExperimentResource(experiment, collection)
    try:
        exp_actual = remote.get_project(exp_setup)
        coord_setup = CoordinateFrameResource(exp_actual.coord_frame)
        coord_actual = remote.get_project(coord_setup)
        return (exp_setup, coord_actual)
    except Exception as e:
        print(e.message)


def setup_channel_boss(
        remote,
        collection,
        experiment,
        channel,
        channel_type='image',
        datatype='uint16'):
    (exp_setup, coord_actual) = setup_experiment_boss(
        remote, collection, experiment)

    chan_setup = ChannelResource(
        channel,
        collection,
        experiment,
        channel_type,
        datatype=datatype)
    try:
        chan_actual = remote.get_project(chan_setup)
        return (exp_setup, coord_actual, chan_actual)
    except Exception as e:
        print(e.message)


# Note: The following functions assume an anisotropic dataset. This is generally a bad assumption. These
# functions are stopgaps until proper coordinate frame at resulution
# support exists in intern.
def get_xyz_extents(rmt, ch_rsc, res=0, iso=True):
    boss_url = 'https://api.boss.neurodata.io/v1/'
    ds = boss_url + \
        '/downsample/{}?iso={}'.format(ch_rsc.get_cutout_route(), iso)
    headers = {'Authorization': 'Token ' + rmt.token_project}
    r_ds = requests.get(ds, headers=headers)
    response = r_ds.json()
    x_range = [0, response['extent']['{}'.format(res)][0]]
    y_range = [0, response['extent']['{}'.format(res)][1]]
    z_range = [0, response['extent']['{}'.format(res)][2]]
    spacing = response['voxel_size']['{}'.format(res)]
    return (x_range, y_range, z_range, spacing)

def get_coord_frame_details(rmt, coll, exp):
    exp_resource = ExperimentResource(exp, coll)
    coord_frame = rmt.get_project(exp_resource).coord_frame

    coord_frame_resource = CoordinateFrameResource(coord_frame)
    data = rmt.get_project(coord_frame_resource)

    max_dimensions = (data.z_stop, data.y_stop, data.x_stop)
    voxel_size = (data.z_voxel_size, data.y_voxel_size, data.x_voxel_size)
    return max_dimensions, voxel_size

def get_offset_boss(coord_frame, res=0, isotropic=False):
    return [
        int(coord_frame.x_start / (2.**res)),
        int(coord_frame.y_start / (2.**res)),
        int(coord_frame.z_start / (2.**res)) if isotropic else coord_frame.z_start]

def create_channel_resource(rmt, chan_name, coll_name, exp_name, type='image',
                            base_resolution=0, sources=[], datatype='uint16', new_channel=True):
    channel_resource = ChannelResource(chan_name, coll_name, exp_name, type=type,
                                       base_resolution=base_resolution, sources=sources, datatype=datatype)
    if new_channel:
        new_rsc = rmt.create_project(channel_resource)
        return new_rsc

    return channel_resource

def upload_to_boss(rmt, data, channel_resource, resolution=0):
    Z_LOC = 0
    size = data.shape
    for i in range(0, data.shape[Z_LOC], 16):
        last_z = i+16
        if last_z > data.shape[Z_LOC]:
            last_z = data.shape[Z_LOC]
    #    print(resolution, [0, size[2]], [0, size[1]], [i, last_z])
        rmt.create_cutout(channel_resource, resolution,
                          [0, size[2]], [0, size[1]], [i, last_z],
                          np.asarray(data[i:last_z,:,:], order='C'))
def imgDownload_boss(
        remote,
        channel_resource,
        coordinate_frame_resource,
        resolution=0,
        size=[],
        start=[],
        isotropic=False):
    """
    Download image with given token from given server at given resolution.
    If channel isn't specified the first channel is downloaded.
    """
    # TODO: Fix size and start parameters

    voxel_unit = coordinate_frame_resource.voxel_unit
    voxel_units = ('nanometers', 'micrometers', 'millimeters', 'centimeters')
    factor_divide = (1e-6, 1e-3, 1, 10)
    fact_div = factor_divide[voxel_units.index(voxel_unit)]

    spacingBoss = [
        coordinate_frame_resource.x_voxel_size,
        coordinate_frame_resource.y_voxel_size,
        coordinate_frame_resource.z_voxel_size]
    spacing = [x * fact_div for x in spacingBoss]  # Convert spacing to mm
    if isotropic:
        spacing = [x * 2**resolution for x in spacing]
    else:
        spacing[0] = spacing[0] * 2**resolution
        spacing[1] = spacing[1] * 2**resolution
        # z spacing unchanged since not isotropic

    if size == []:
        size = get_image_size_boss(
            coordinate_frame_resource, resolution, isotropic)
    if start == []:
        start = get_offset_boss(
            coordinate_frame_resource, resolution, isotropic)
#    if isotropic:
#        x_range, y_range, z_range, spacing = get_xyz_extents(
#            remote, channel_resource, res=resolution, iso=isotropic)

    # size[2] = 200
    # dataType = metadata['channels'][channel]['datatype']
    dataType = channel_resource.datatype

    # Download all image data from specified channel
    array = remote.get_cutout(
        channel_resource, resolution, [
            start[0], size[0]], [
            start[1], size[1]], [
                start[2], size[2]])

    # Cast downloaded image to server's data type
    # convert numpy array to sitk image
    img = sitk.Cast(sitk.GetImageFromArray(array), ndToSitkDataTypes[dataType])

    # Reverse axes order
    # img = sitk.PermuteAxesImageFilter().Execute(img,range(dimension-1,-1,-1))
    img.SetDirection(identityDirection)
    img.SetSpacing(spacing)

    # Convert to 2D if only one slice
    img = util.imgCollapseDimension(img)

    return img


def download_ara(rmt, resolution, type='average'):
    if resolution not in [10, 25, 50, 100]:
        print('Please provide a resolution that is among the following: 10, 25, 50, 100')
        return
    REFERENCE_COLLECTION = 'ara_2016'
    REFERENCE_EXPERIMENT = 'sagittal_{}um'.format(resolution)
    REFERENCE_COORDINATE_FRAME = 'ara_2016_{}um'.format(resolution)
    REFERENCE_CHANNEL = '{}_{}um'.format(type, resolution)

    refImg = download_image(rmt, REFERENCE_COLLECTION, REFERENCE_EXPERIMENT, REFERENCE_CHANNEL, ara_res=resolution)

    return refImg

def download_image(rmt, collection, experiment, channel, res=0, isotropic=True, ara_res=None):
    (exp_resource, coord_resource, channel_resource) = setup_channel_boss(rmt, collection, experiment, channel)
    img = imgDownload_boss(rmt, channel_resource, coord_resource, resolution=res, isotropic=isotropic)
    return img

def get_image_size_boss(coord_frame, res=0, isotropic=False):
    return [
        int(coord_frame.x_stop / (2.**res)),
        int(coord_frame.y_stop / (2.**res)),
        int(coord_frame.z_stop / (2.**res)) if isotropic else coord_frame.z_stop]

def get_viz_link(data):
    url = 'https://viz.neurodata.io/?json_url='
    json_url = 'https://json.neurodata.io/v1'
    r = requests.post(json_url,json=data)
    return url + r.json()['uri']
