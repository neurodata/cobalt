import tempfile 

import numpy as np

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *

import configparser

import requests

import sys

# intern
boss_config_file = 'neurodata.cfg'
rmt = BossRemote(boss_config_file)

#for making manual web requests to the boss:
config = configparser.ConfigParser()
config.read(boss_config_file)
APITOKEN = config['Default']['token']

coll = 'weiler14'

experiments_process = np.loadtxt("experiments.txt", comments="#", dtype='str',ndmin=1)

s = requests.Session()

for exp in experiments_process:
    ch_names = rmt.list_channels(coll, exp)
    for ch in ch_names:
        url = 'https://api.boss.neurodata.io/latest/downsample/{}/{}/{}/'.format(
            coll, exp, ch)
        headers = { 
            'Authorization': 'Token {}'.format(APITOKEN)
        }
        r = s.post(url, data = {}, headers=headers)
        if r.status_code != 201:
            print("Error: Request failed!")
            print("Received {} from server.".format(r.status_code))
            f = tempfile.NamedTemporaryFile()
            f.write(r.content)
            import pdb; pdb.set_trace() 
            sys.exit(1)
        else:
            print("Downsample request sent for {}".format(url))