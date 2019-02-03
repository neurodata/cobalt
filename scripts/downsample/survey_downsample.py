#%%
import configparser
import csv
import datetime

import requests

from intern.remote.boss import BossRemote

#%%
# intern
boss_config_file = "neurodata_boss.cfg"
rmt = BossRemote(boss_config_file)

coll_list = rmt.list_collections()
coll_list.remove('ben_dev')
print(coll_list)

boss_url = 'https://api.boss.neurodata.io/v1'
url_constr = boss_url + '/collection/{}/experiment/{}/channel/{}'

config = configparser.ConfigParser()
config.read(boss_config_file)
APITOKEN = config['Default']['token']
headers = {'Authorization': 'Token {}'.format(APITOKEN)}

s = requests.Session()

#%%
data = []
for coll in coll_list:
    exp_list = rmt.list_experiments(coll)
    for exp in exp_list:
        ch_list = rmt.list_channels(coll, exp)
        for ch in ch_list:
            url = '{}/downsample/{}/{}/{}'.format(boss_url, coll, exp, ch)
            r = s.get(url, headers=headers)
            try:
                downsample_resp = r.json()

                downsample_status = downsample_resp['status']

                extent = downsample_resp['extent']
                last_extent = extent[sorted(extent.keys())[-1]]
                first_extent = extent[sorted(extent.keys())[0]]

                data.append([coll, exp, ch, downsample_status,
                             ', '.join([str(x) for x in first_extent]),
                             ', '.join([str(x) for x in last_extent]),
                             url_constr.format(coll, exp, ch)])
            except Exception as ex:
                template = "An exception of type {} occurred."
                message = template.format(type(ex))
                data.append([coll, exp, ch, message, '',
                             url_constr.format(coll, exp, ch)])
            print(data[-1])

#%%
with open('downsample_status_{}.csv'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")), 'w', newline='') as csvfile:
    write_obj = csv.writer(csvfile)
    write_obj.writerow(['coll', 'exp', 'ch', 'downsample_status',
                        'first_extent', 'last extent', 'url'])
    write_obj.writerows(data)
print('Finished iterating over all channels')
