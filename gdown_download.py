import sys

import gdown
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)

def main(argv):
    (opts, args) = parser.parse_args(argv)

    if(opts.server_config == 0):
        output_dir = "/scratch1/scratch2/neil.delgallego/"
    elif(opts.server_config == 4):
        output_dir = "D:/Datasets/"
    elif(opts.server_config == 6):
        output_dir = "/home/neildelgallego/"
    else:
        output_dir = "/home/jupyter-neil.delgallego/"

    #z00
    direct_link = "https://drive.google.com/file/d/1lGpxxknJqo0rIYimixr4GZsyqqEeMjmk/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z01
    direct_link = "https://drive.google.com/file/d/1lGiXXB-BfTOlKr4HgVyzC6d4N4gk1KVq/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z02
    direct_link = "https://drive.google.com/file/d/1lHJM17Ahhni2xRwKFjSPQO-4Oxr0DKUs/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

    # z03
    direct_link = "https://drive.google.com/file/d/1lHFuvYG3gy_HZUp2CNi7UlC_j2DfHUyn/view?usp=share_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

