import json
import os
import shutil
import random
from cadc_annotation_helper import build_objects_from_3d_annotation

label_box_export_file = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/export-2023-04-12T22_28_24.384Z.json'

labels = json.load(open(label_box_export_file, 'r'))

json_formatted_str = json.dumps(labels, indent=2)

#print(json_formatted_str)

inp_annotation_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/'
inp_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/data_renamed_and_copied'

out_train_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/leftImg8bit/train'
out_val_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/leftImg8bit/val'
out_test_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/leftImg8bit/test'

out_train_label_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/gtFine/train'
out_val_label_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/gtFine/val'
out_test_label_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/gtFine/test'

os.makedirs(out_train_img_dir, exist_ok=True)
os.makedirs(out_val_img_dir, exist_ok=True)
os.makedirs(out_test_img_dir, exist_ok=True)

os.makedirs(out_train_label_dir, exist_ok=True)
os.makedirs(out_val_label_dir, exist_ok=True)
os.makedirs(out_test_label_dir, exist_ok=True)

print('Number of labelled images:', len(labels))

for data in labels:
    label = {}
    
    label['imgHeight'] = 1024
    label['imgWidth'] = 1280
    
    #print(data['External ID'])
    org_file_name = data['External ID']
    img_file_name = data['External ID'][:-4] + '_leftImg8bit.png'
    label_file_name = org_file_name[:-4] + '_gtFine_polygons.json'
    
    #print(file_name)
    
    #print(data['Label']['objects'])
    objects = []
    
    for cur in data['Label']['objects']:
        #print(cur)
        object = {}
        
        object['label'] = cur['title']
        
        polygon = []
        for coord in cur['polygon']:
            polygon.append([coord['x'], coord['y']])
            
        object['polygon'] = polygon
        
        objects.append(object)
    
    annotations_file = '/'.join(os.path.join(inp_annotation_dir, org_file_name.replace('__', '/')).split('/')[:-4]) + '/3d_ann.json'
    annotations_data = json.load(open(annotations_file, 'r'))

    frame = int(org_file_name.replace('__', '/')[-6:-4])
    
    calib_path = '/'.join(os.path.join(inp_annotation_dir, org_file_name.replace('__', '/')).split('/')[:-5]) + '/calib'
    
    cuboids = annotations_data[frame]['cuboids']
    
    objects.extend(build_objects_from_3d_annotation(cuboids, calib_path))
    
    label['objects'] = objects

    rand_val = random.random()
    if rand_val < 0.8:
        shutil.copyfile(os.path.join(inp_img_dir, org_file_name), os.path.join(out_train_img_dir, img_file_name))
        json.dump(label, open(os.path.join(out_train_label_dir, label_file_name), 'w'))
    elif rand_val >= 0.8 and rand_val < 0.9:
        shutil.copyfile(os.path.join(inp_img_dir, org_file_name), os.path.join(out_val_img_dir, img_file_name))
        json.dump(label, open(os.path.join(out_val_label_dir, label_file_name), 'w'))
    else:
        shutil.copyfile(os.path.join(inp_img_dir, org_file_name), os.path.join(out_test_img_dir, img_file_name))
        json.dump(label, open(os.path.join(out_test_label_dir, label_file_name), 'w'))
    
    #break


