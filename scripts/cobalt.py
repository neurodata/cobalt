from intern.remote.boss import BossRemote
from correct_lavision_bias import *
from register_brain import *
from download_chunks import download_data
from detect_cells import detect_cells
from boss_util import *
import json

# script for whole cobalt pipeline

def main():
    config = 'neurodata.cfg'
    config_p = configparser.ConfigParser()
    config_p.read(config)
    rmt = BossRemote(config)

#    # correct bias in channels
#    correct_bias(rmt, config_p)
#    # assign permissions
#    
#
#    # wait until data are downsampled
#    # currently 10 min (720 seconds)
#    print('downsampling data on the BOSS...')
#    time.sleep(720) 
#
#    # register channels
#    register_data(rmt, config_p)
#
#    # detect cells
#    download_data(config)
    json_path = detect_cells(rmt, config_p)
    data = json.read(json_path)
    viz_link = get_viz_link(data)
    print('look at cell detections here: {}'.format(viz_link))
    



if __name__ == "__main__":
    main()
