import os
import numpy as np
from PIL import Image, ImageDraw

from evaluation import QudrilateralFinder, CornersCoarseEstimation

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
    #TODO - convert evaluation to a function??
    v = v0
    output_path = f'results/{v[0]}_only/'
    # output_path = 'results/debug'
    img_suffix = ''
    data_dir1 = f'z_ref_doc_scanner/data/self_collected/low-level-camera/stills/'
    data_dir2 = f'z_ref_doc_scanner/data/self_collected/high-level-camera/stills/'
    
    os.makedirs(output_path, exist_ok=True)
    
    ds1 = Dataset.from_directory(data_dir1, ignore=True)
    ds2 = Dataset.from_directory(data_dir2, ignore=True)
    ds = ds1 + ds2

    quadrilateral_finder = QudrilateralFinder(v[1], v[2])
    quadrilateral_finder_coarse = CornersCoarseEstimation(v[1])

    iou = []
    for i in range(len(ds)):
        im, quad_true = ds.readimage(i)
        img_name = ds.get_name(i)
        
        quad_pred = quadrilateral_finder.find_qudrilateral(im)
        quad_pred_coarse = quadrilateral_finder.quad_pred_coarse
        patches_coords = quadrilateral_finder.patches_coords
        
        ImageDraw.Draw(im).polygon(quad_pred, outline='red')
        ImageDraw.Draw(im).polygon(quad_pred_coarse, outline='blue')
        [ImageDraw.Draw(im).rectangle(patches_coords[i], outline='green', width=2) for i in range(4)]
        out_name = f'{output_path}/{img_name}{img_suffix}.jpg'
        im.save(out_name)
        iou.append(IOU(quad_true, quad_pred)) #TODO - understand their calcualtion for IOU
        
        print(f'{out_name} --- iou={iou[-1]:.02f}')
        print(quad_pred)

        # next line is for debugging (refactoring, etc). iou belongs to image 'low-level-camera/stills/00000.jpg'
        # assert iou[0] == 0.5232468134613789

    #TODO - write log file with all the details.
    #TODO - add timing
    print(np.mean(np.array(iou)))
