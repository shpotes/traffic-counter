import os
import pandas as pd
import xml.etree.ElementTree as ET

from tqdm import tqdm


def split(x):
    """deterministic train/test/validation split"""
    hs = hash(x.video + x.frame) % 100  + 1
    if hs < 80:
        return 'train'
    elif hs < 90:
        return 'test'
    return 'val'

def create_metadata(dirr):
    """create metadata file from raw xml annotations"""
    data = []
    for seq in tqdm(os.listdir(dirr + '/DETRAC-Train-Annotations-XML/')):
        tree = ET.parse('DETRAC-Train-Annotations-XML/' + seq)
        root = tree.getroot()
        vid = root.attrib['name']

        for frame in iter(root):
            if frame.tag == 'frame':
                frame_number = frame.attrib['num']
                for target in frame[0]:
                    box, attrib = tuple(iter(target))[:2]
                    cat = attrib.attrib['vehicle_type']
                    box = list(map(float, box.attrib.values()))
                    tmp = [vid, frame_number] + box + [cat]
                    data.append(tmp)

    data = pd.DataFrame(data)
    data.columns = ['video', 'frame',
                    'left', 'top', 'wigth', 'height',
                    'category']

    data['split'] = data.T.apply(split)

    data['filename'] = data.T.apply(lambda x: 'Insight-MVT_Annotation_Train/%s/img%05d.jpg' % (x.video, int(x.frame)))
    
    data.to_csv('metadata.csv', index=False)

def refactor_dataset(dirr):
    pass    

def create_small(dirr, small_dirr):
    pass

if __name__ == '__main__':
    create_metadata('raw')
