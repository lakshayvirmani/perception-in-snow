import os
import shutil

src_dir = 'data'
dst_dir = 'data_renamed_and_copied'

collected_files = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if '2018_03_06' not in root and file.endswith(".png") and int(file[-6:-4]) % 5 == 0:
             collected_files.append(os.path.join(root, file))

print(len(collected_files))
print(collected_files[:10])

        
for i in range(len(collected_files)):
    shutil.copyfile(collected_files[i], os.path.join(dst_dir, collected_files[i].replace('/', '__')))
