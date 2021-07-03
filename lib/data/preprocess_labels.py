from shutil import copyfile
import os
import re
from itertools import chain

ANNOTATORS = list(map(str, range(8)))
NEW_FOLDER = './data/all_labels'
AGGREGATED_FOLDER = './data/aggregated_labels'
ANNOTATORS_FOLDER = './data'

full_path_list = lambda folder: [
    os.path.join(folder, file) for file in os.listdir(folder)
]

def organize_labels():
    agg_folder_ids = [
        file.split('_')[0] for file in os.listdir(AGGREGATED_FOLDER)
    ]
    annotators_files = {
        ann: [
                file for file in full_path_list(os.path.join(ANNOTATORS_FOLDER, ann))
                    if file.endswith('xml')
            ]
                for ann in ANNOTATORS
    }
    written_image_ids = []
    for ann, files in annotators_files.items():
        for file_name in files:
            image_id = file_name.split('/')[-1].split('.')[0]
            if image_id in written_image_ids:
                continue
            new_file_name = os.path.join(NEW_FOLDER, f'{image_id}.xml')
            written_image_ids.append(image_id)
            if image_id in agg_folder_ids:
                file_name = os.path.join(AGGREGATED_FOLDER, file_name.split('/')[-1].replace('.xml', '_aggregated.xml'))
                assert not 'aggregated_labels' in new_file_name and 'all_labels' in new_file_name
                copyfile(file_name, new_file_name)
            else:
                with open(file_name, 'r') as fd:
                    txt = fd.read()
                new_txt = re.sub(r'<name>\w*</name>', f'<name>{ann}</name>', txt)
                assert 'all_labels' in new_file_name
                with open(new_file_name, 'w') as fd:
                    fd.write(new_txt)
    print(f'A total of {len(written_image_ids)} unique images are written')
if __name__ == '__main__':
    organize_labels()