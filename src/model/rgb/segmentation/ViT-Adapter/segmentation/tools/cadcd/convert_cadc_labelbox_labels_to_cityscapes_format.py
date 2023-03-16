import json
import os
import shutil
import random

label_box_export_file = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/export-2023-03-16T16_10_32.997Z.json'

labels = json.load(open(label_box_export_file, 'r'))

json_formatted_str = json.dumps(labels, indent=2)

print(json_formatted_str)

inp_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/data_renamed_and_copied'

out_train_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/leftImg8Bit/train'
out_val_img_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/leftImg8Bit/val'

out_train_label_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/gtFine/train'
out_val_label_dir = '/Users/lakshayvirmani/Desktop/Projects/cadc_devkit/processed_data/cadcd/gtFine/val'

os.makedirs(out_train_img_dir, exist_ok=True)
os.makedirs(out_val_img_dir, exist_ok=True)

os.makedirs(out_train_label_dir, exist_ok=True)
os.makedirs(out_val_label_dir, exist_ok=True)

print('Number of labelled images:', len(labels))

for data in labels:
    label = {}
    
    label['imgHeight'] = 1024
    label['imgWidth'] = 1280
    
    #print(data['External ID'])
    img_file_name = data['External ID']
    label_file_name = img_file_name[:-4] + '_gtFine_polygons.json'
    
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
    
    label['objects'] = objects
    #print(label)
    
    if random.random() < 0.8:
        shutil.copyfile(os.path.join(inp_img_dir, img_file_name), os.path.join(out_train_img_dir, img_file_name))
        json.dump(label, open(os.path.join(out_train_label_dir, label_file_name), 'w'))
    else:
        shutil.copyfile(os.path.join(inp_img_dir, img_file_name), os.path.join(out_val_img_dir, img_file_name))
        json.dump(label, open(os.path.join(out_val_label_dir, label_file_name), 'w'))
    
    #break


