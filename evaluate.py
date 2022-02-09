import os
import numpy as np
from PIL import ImageDraw

from evaluation import QudrilateralFinder
from utils import draw_circle_pil
# imports from my other projects:
import os, sys
sys.path.append(os.path.abspath('z_ref_doc_scanner'))  #TODO - why importing fails if i put it under referneces folder?
from z_ref_doc_scanner.dataset import Dataset
from z_ref_doc_scanner.utils import IOU, plot_mesh

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

v22 = [  'v22',
        'results/trained_models/document/22122021_document_smartdoc/nonamedocument_resnet.pb',
        'results/trained_models/corner/my_corner_v2_Jan31_13-11-58/my_corner_v2_resnet.pb'
        ]

v3 = ['v3',
        None,
        'results/trained_models/corner/my_corner_v2_topleft_training_Feb07_12-34-53/my_corner_v2_topleft_training_resnet.pb'
    ]

v33 = ['v3_no_color',
        None,
        'results/trained_models/corner/my_corner_v2_topleft_training_no_color_jitter_Feb08_10-28-42/my_corner_v2_topleft_training_no_color_jitter_resnet.pb'
    ]

if __name__ == '__main__':
    v = v0
    output_path = f'results/{v[0]}/'
    # output_path = 'results/debug'
    img_suffix = ''
    
    os.makedirs(output_path, exist_ok=True)
    
    ds = Dataset.from_directory(f'z_ref_doc_scanner/data/self_collected/low-level-camera/stills/', ignore=True) + \
         Dataset.from_directory(f'z_ref_doc_scanner/data/self_collected/high-level-camera/stills/', ignore=True)

    qf = QudrilateralFinder(v[1], v[2])

    imgs = []
    iou = []
    iou_coarse = []
    for i in range(len(ds)):
        im, quad_true = ds.readimage(i)
        
        quad_pred = qf.find_quad(im)
        # quad_pred = qf.find_quad_model2_only(im)
        # quad_pred = qf.find_quad_model2_only_by_top_left(im)
        
        im_out = im.copy()
        ImageDraw.Draw(im_out).polygon(quad_pred, outline='red') 
        
        patches_coords = qf.patches_coords
        quad_coarse = qf.quad_coarse
        [draw_circle_pil(im_out, xy, radious=10, width=50, outline='yellow') for xy in quad_coarse]
        patches_coords = [[(coords[0][0] +4, coords[0][1] + 4), (coords[1][0] - 4, coords[1][1] - 4)] for coords in patches_coords] #shrink rectangles for better vizualization
        [ImageDraw.Draw(im_out).rectangle(patches_coords[i], outline='yellow', width=1) for i in range(4)]
        

        img_name = ds.get_name(i)
        out_name = f'{output_path}/{img_name}{img_suffix}'
        im_out.save(f'{out_name}.jpg')
        
        imgs.append(im_out)
        iou.append(IOU(quad_true, quad_pred))
        iou_coarse.append(IOU(quad_true, quad_coarse))
        
        print(f'{out_name} --- iou={iou[-1]:.02f}')

        # next line is for debugging (refactoring, etc). iou[0] belongs to image 'low-level-camera/stills/00000.jpg' with v2 models.
        # assert iou[0] == 0.5232468134613789
    #TODO - write log file with all the details. ->even better concate it in df and print it to csv
    #TODO - add timing
    #TODO - understand their calcualtion for IOU    

    fig = plot_mesh(imgs, [5,8], title=output_path)
    fig.savefig(f'{output_path}/all_images.jpg')
    
    print(f'iou={np.mean(np.array(iou))}')
    print(f'iou_coarse={np.mean(np.array(iou_coarse))}')
