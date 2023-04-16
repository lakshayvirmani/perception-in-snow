import json
import os
import shutil
import random
from cadc_annotation_helper import build_objects_from_3d_annotation


inp_data_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/data'

out_train_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/leftImg8bit/train'
out_val_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/leftImg8bit/val'
out_test_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/leftImg8bit/test'

output_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/'

train_file = os.path.join(output_dir, 'train_lidar_files.txt')
val_file = os.path.join(output_dir, 'val_lidar_files.txt')
test_file = os.path.join(output_dir, 'test_lidar_files.txt')

def collect_files(dir, output_file):
    input_files = [file for file in os.listdir(dir) if '.png' in file]
    output_files = []
    
    for file in input_files:
        split_units = file.split('__')
        last_unit = split_units[-1].split('_')[0] + '.bin'
        prefix = '/'.join(split_units[1:-1])
        prefix = prefix.replace('image_00', 'lidar_points')
        output_filename = prefix + '/' + last_unit
        assert os.path.exists(os.path.join(inp_data_dir, output_filename))
        output_files.append(output_filename)

    open(output_file, 'w').write('\n'.join(output_files))
        
collect_files(out_train_img_dir, train_file)
collect_files(out_val_img_dir, val_file)
collect_files(out_test_img_dir, test_file)
