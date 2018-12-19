import os
import ndreg
from ndreg import preprocessor, util
import numpy as np
import SimpleITK as sitk
from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import time
from downsample.startDownsample import *
from boss_util import *
import datetime

def correct_bias(rmt, config, scale=0.1):
    t_start_overall = time.time()

    mm_to_um = 1000.0

    # download image
    params = config['preprocessing']
    shared_params = config['shared']
    channels = params['channels']
    perms = config['permissions']


    # resolution level from 0-5
    resolution_image = 3
    image_isotropic = True

    for channel in channels.split(','):
        print('downloading experiment: {}, channel: {}...'.format(shared_params['experiment'], channel))
        t1 = time.time()
        img = download_image(rmt, shared_params['collection'], shared_params['experiment'], channel, res=resolution_image, isotropic=image_isotropic)
        print("time to download image at res {} um: {} seconds".format(img.GetSpacing()[0] * mm_to_um, time.time()-t1))
        
        img_bc = preprocessor.correct_bias_field(img, scale=scale, niters=[100,100,100,100])

        print("uploading bias corrected image back to the BOSS")
        new_channel = channel + '_bias_corrected'
        try:
            ch_rsc_bc = create_channel_resource(rmt, new_channel, shared_params['collection'], shared_params['experiment'], datatype='uint16', type='image')
            upload_to_boss(rmt, sitk.GetArrayFromImage(img_bc).astype('uint16'), ch_rsc_bc)
        except Exception as e:
            new_channel = new_channel + str(datetime.datetime.now().time())[:8].replace(':','_')
            ch_rsc_bc = create_channel_resource(rmt, new_channel, shared_params['collection'], shared_params['experiment'], datatype='uint16', type='image')
            upload_to_boss(rmt, sitk.GetArrayFromImage(img_bc).astype('uint16'), ch_rsc_bc)



        url = '{}downsample/{}/{}/{}/'.format(BOSS_URL,
                                              shared_params['collection'], shared_params['experiment'], new_channel)
        makeDownsampleRequest(url, config['Default']['token'])

