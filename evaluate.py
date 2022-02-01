import os
import numpy as np
from PIL import Image

from evaluation import QudrilateralFinder, CornerExtractor

# imports from my other projects:
import os, sys
sys.path.append(os.path.abspath('z_ref_doc_scanner'))  #TODO - why importing fails if i put it under referneces folder?
from z_ref_doc_scanner.dataset import Dataset
from z_ref_doc_scanner.utils import draw_qudrilateral, IOU

v0 = [  'v0',
        'results/trained_models/document/22122021_document_smartdoc/nonamedocument_resnet.pb',
        'results/trained_models/corner/26122021_corner_smartdoc/nonamecorner_resnet.pb'
        ]

v1 = [  'v1',
        'results/trained_models/document/1212022_document_v1/nonamemy_document_resnet.pb',
        'results/trained_models/corner/17122022_corner_mycorners/nonamemy_corner_resnet.pb'
        ]

v2 = [  'v2',
        'results/trained_models/document/1212022_document_v1/nonamemy_document_resnet.pb',
        'results/trained_models/corner/my_corner_v2_Jan31_13-11-58/my_corner_v2_resnet.pb'
        ]

if __name__ == '__main__':
    #TODO - convert evaluation to a function
    v = v2
    data_type = 'low' #TODO - refactor code to be able to load two datasets together low&high
    # output_path = f'results/{v[0]}/{data_type}'
    output_path = 'results/debug'
    img_suffix = ''
    data_dir = f'z_ref_doc_scanner/data/self_collected/{data_type}-level-camera/stills/'
    
    os.makedirs(output_path, exist_ok=True)
    
    ds = Dataset(data_dir, ignore=True)
    quadrilateral_finder = QudrilateralFinder(v[1], v[2])
    quadrilateral_finder_coarse = CornerExtractor(v[1])

    iou = []
    for i in [0]:#range(len(ds)):
        im, quad_true = ds.readimage(i)
        img_name = ds.get_name(i)
        
        quad_pred_coarse = quadrilateral_finder_coarse.find_qudrilateral(im)
        quad_pred = quadrilateral_finder.find_qudrilateral(im)
        
        oim = Image.fromarray(draw_qudrilateral(quad_pred, np.array(im)))
        oim = Image.fromarray(draw_qudrilateral(quad_pred_coarse, np.array(oim), color=(0,0,255), thickness=1))
        oim.save(f'{output_path}/{img_name}{img_suffix}.jpg')
        iou.append(IOU(quad_true, quad_pred)) #TODO - understand their calcualtion for IOU and what is the difference
        
        print(f'{img_name} --- iou={iou[-1]:.02f}')
        print(quad_pred)
        assert iou[0] == 0.5267646714661619
    #TODO - write log file with all the details.
    #TODO - add timing
    print(np.mean(np.array(iou)))
