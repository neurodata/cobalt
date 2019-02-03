import sys
import argparse
import requests
import tempfile

BOSS_URL = 'https://api.boss.neurodata.io/latest/'


def makeDownsampleRequest(url, token):
    headers = {
        'Authorization': 'Token {}'.format(token)
    }
    r = requests.post(url, data={}, headers=headers)
    if r.status_code != 201:
        print("Error: Request failed!")
        print("Received {} from server.".format(r.status_code))
        f = tempfile.NamedTemporaryFile()
        f.write(r.content)
        sys.exit(1)
    else:
        print("Downsample request sent for {}".format(url))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('token', type=str, help='User token for the boss.')
    parser.add_argument('collection', type=str, help='Collection')
    parser.add_argument('experiment', type=str, help='Experiment')
    parser.add_argument('channel', type=str, help='Channel')

    result = parser.parse_args()
    url = '{}downsample/{}/{}/{}/'.format(BOSS_URL,
                                          result.collection, result.experiment, result.channel)
    makeDownsampleRequest(url, result.token)


if __name__ == '__main__':
    main()
