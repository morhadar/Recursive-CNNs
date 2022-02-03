import os
import numpy as np
from PIL import ImageDraw

from evaluation import QudrilateralFinder
# imports from my other projects:
import os, sys
sys.path.append(os.path.abspath('z_ref_doc_scanner'))  #TODO - why importing fails if i put it under referneces folder?
from z_ref_doc_scanner.dataset import Dataset
from z_ref_doc_scanner.utils import IOU

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
    v = v2
    output_path = f'results/{v[0]}/'
    # output_path = 'results/debug'
    img_suffix = ''
    
    os.makedirs(output_path, exist_ok=True)
    
    ds = Dataset.from_directory(f'z_ref_doc_scanner/data/self_collected/low-level-camera/stills/', ignore=True) + \
         Dataset.from_directory(f'z_ref_doc_scanner/data/self_collected/high-level-camera/stills/', ignore=True)

    qf = QudrilateralFinder(v[1], v[2])
    qf0 = QudrilateralFinder(v0[1], v0[2])

    iou = []
    iou_model2_only = []
    for i in range(len(ds)):
        im, quad_true = ds.readimage(i)
        img_name = ds.get_name(i)
        
        quad_pred = qf.find_quad(im)
        patches_coords = qf.patches_coords
        quad_pred_model2_only = qf.find_quad_model2_only(im)
        quad_coarse = qf.quad_coarse
        
        # ImageDraw.Draw(im).polygon(quad_pred, outline='red')
        # ImageDraw.Draw(im).polygon(quad_pred_model2_only, outline='blue')
        
        ImageDraw.Draw(im).polygon(quad_pred, outline='red')
        ImageDraw.Draw(im).polygon(quad_pred_model2_only, outline='blue')
        
        #TODO - results for coarse model only
        # quad_pred0 = qf0.find_quad(im)
        # ImageDraw.Draw(im).polygon(quad_coarse, outline='yellow') #TODO - print circles instead
        # [ImageDraw.Draw(im).rectangle(patches_coords[i], outline='yellow', width=2) for i in range(4)]
        
        out_name = f'{output_path}/{img_name}{img_suffix}.jpg'
        im.save(out_name)
        #TODO - understand their calcualtion for IOU
        iou.append(IOU(quad_true, quad_pred))
        # iou.append(IOU(quad_true, quad_coarse))
        iou_model2_only.append(IOU(quad_true, quad_pred_model2_only)) 
        
        print(f'{out_name} --- iou={iou[-1]:.02f}')
        print(quad_pred)

        # next line is for debugging (refactoring, etc). iou[0] belongs to image 'low-level-camera/stills/00000.jpg' with v2 models.
        # assert iou[0] == 0.5232468134613789
    #TODO - write log file with all the details. ->even better concate it in df and print it to csv
    #TODO - add timing
    print(f'iou={np.mean(np.array(iou))}')
    print(f'iou_model2_only={np.mean(np.array(iou_model2_only))}')
