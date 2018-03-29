import os
import sys
import argparse

from utils import indoor3d_util

#---- input arguments
parser = argparse.ArgumentParser(description='Process input arguments.')

parser.add_argument('--raw_data_dir',
                    help='directory where raw dataset is stored')

parser.add_argument('--output_folder', default='./stanford_indoor3d',
                    help='directory of store processed stanford indoor3d dataset')

args = parser.parse_args()


#---- global variable
#Stanford3dDataset_DIR = 'Stanford3dDataset_v1.2_Aligned_Version'
Stanford3dDataset_DIR = args.raw_data_dir

anno_paths = [line.rstrip() for line in open(os.path.join('./', 'utils/meta/anno_paths.txt'))]
#anno_paths = [os.path.join(indoor3d_util.DATA_PATH, p) for p in anno_paths]
anno_paths = [os.path.join(Stanford3dDataset_DIR, p) for p in anno_paths]

output_folder = args.output_folder


if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
for anno_path in anno_paths:
    print(anno_path)
    #try:
    elements = anno_path.split('/')
    out_filename = elements[-3]+'_'+elements[-2]+'.npy' # Area_1_hallway_1.npy
    indoor3d_util.collect_point_label(anno_path, os.path.join(output_folder, out_filename), 'numpy')
    #except:
    #    print(anno_path, 'ERROR!!')
