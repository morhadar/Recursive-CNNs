import os
import numpy as np
from PIL import ImageDraw
import pandas as pd

from evaluation import QudrilateralFinder
from utils import draw_circle_pil, mesh_imgs, draw_polygon_pil, IOU
from dataprocessor import Dataset

v0 = [  'v0',
        'trained_models/document/22122021_document_smartdoc/nonamedocument_resnet.pb',
        'trained_models/corner/26122021_corner_smartdoc/nonamecorner_resnet.pb'
        ]

v1 = [  'v1',
        'trained_models/document/1212022_document_v1/nonamemy_document_resnet.pb',
        'trained_models/corner/17122022_corner_mycorners/nonamemy_corner_resnet.pb'
        ]


v2_0 = [  'v2_0',
        'trained_models/document/22122021_document_smartdoc/nonamedocument_resnet.pb',
        'trained_models/corner/my_corner_v2_Jan31_13-11-58/my_corner_v2_resnet.pb'
        ]

v2_1 = [  'v2_1',
        'trained_models/document/1212022_document_v1/nonamemy_document_resnet.pb',
        'trained_models/corner/my_corner_v2_Jan31_13-11-58/my_corner_v2_resnet.pb'
        ]

v2 = [  'v2',
        None,
        # 'trained_models/corner/my_corner_v2_Jan31_13-11-58/my_corner_v2_resnet.pb'
        '/home/mhadar/projects/doc_scanner/results/v2_corner_Feb23_15-16-51/v2_corner_resnet.pb'
        # 'trained_models/corner/v3_Feb14_08-50-21/v3_resnet.pb'
        ]

v2c = [  'v2c',
        None,
        'trained_models/corner/v2c_Feb13_13-42-45/v2c_resnet.pb'
        # 'trained_models/corner/v2b_Feb13_10-34-41/v2b_resnet.pb'
        ]

v3 = ['v3',
        None,
        # 'trained_models/corner/v3b_Feb15_13-11-47/v3b_resnet.pb'
        # 'trained_models/corner/v3_Feb16_11-34-18/v3_resnet.pb' #fixed 80%-20%
        # 'trained_models/corner/v3_Feb14_08-50-21/v3_resnet.pb' #iou=0.99
        '/home/mhadar/projects/doc_scanner/trained_models/corner/v2_corner_topleft_Feb24_11-44-39/v2_corner_topleft_resnet.pb'
    ]

if __name__ == '__main__':
    v = v3
    output_path = f'results/{v[0]}'
    # output_path = 'results/debug'
    # output_path = '/home/mhadar/projects/doc_scanner/results/z_ref/'
    
    img_suffix = ''
    
    os.makedirs(output_path, exist_ok=True)
    
    ds = Dataset.from_directory(f'/home/mhadar/projects/doc_scanner/data/self_collected/low-level-camera/stills', ignore=True) + \
         Dataset.from_directory(f'/home/mhadar/projects/doc_scanner/data/self_collected/high-level-camera/stills', ignore=True)
        #  Dataset.from_directory('/home/mhadar/projects/doc_scanner/data/data_generator/sandbox')#TODO - this way it not gurnateed to run images from testset. (it will probably include images from trainset as well)
    
    N = len(ds)
    qf = QudrilateralFinder(v[1], v[2])

    imgs = []
    iou = []
    iou_coarse = []
    df = pd.DataFrame(columns=['img_name', 'iou', 'quad_true', 'quad_pred'], index=range(N))
    for i in range(N):
        im, quad_true = ds.read_sample(i)
        
        # quad_pred = qf.find_quad(im)
        # quad_pred = qf.find_quad_model2_only(im)
        quad_pred = qf.find_quad_model2_only_by_top_left(im)
        
        im_out = im.copy()
        draw_polygon_pil(im_out, quad_pred, outline='red', width=3)
        
        # coarse info:
        # patches_coords = qf.patches_coords
        # quad_coarse = qf.quad_coarse
        # [draw_circle_pil(im_out, xy, radious=10, width=50, outline='yellow') for xy in quad_coarse]
        # patches_coords = [[(coords[0][0] +4, coords[0][1] + 4), (coords[1][0] - 4, coords[1][1] - 4)] for coords in patches_coords] #shrink rectangles for better vizualization
        # [ImageDraw.Draw(im_out).rectangle(patches_coords[i], outline='yellow', width=1) for i in range(4)]
        # iou_coarse.append(round(IOU(quad_true, quad_coarse), 2))
        

        img_name = ds.get_name(i)
        out_name = f'{output_path}/{img_name}{img_suffix}'
        im_out.save(f'{out_name}.jpg')
        
        imgs.append(im_out)
        iou.append(round(IOU(quad_true, quad_pred), 2))
        
        print(f'{out_name} --- iou={iou[-1]:.02f} -- {quad_pred}')
        df.img_name[i] = img_name
        df.iou[i] = iou[-1]
        df.quad_true[i] = quad_true
        df.quad_pred[i] = quad_pred

        # next line is for debugging (refactoring, etc). iou[0] belongs to image 'low-level-camera/stills/00000.jpg' with v2 models.
        # assert iou[0] == 0.5232468134613789
    #TODO - add timing
    #TODO - understand their calcualtion for IOU    
    
    iou_str = [f'{df.img_name[i]} --- iou={df.iou[i]:.02f}' for i in range(N)]
    mesh = mesh_imgs(imgs[:29], [5,6], titles=iou_str[:29]) #TODO - instead of writing 29 and 30. take this numbers from dataset.
    mesh.save(f'{output_path}/all_images.jpg')
    mesh_high = mesh_imgs(imgs[30:39], [3,3], titles=iou_str[30:39])
    mesh_high.save(f'{output_path}/high_all_images.jpg')
    # mesh_val = mesh_imgs(imgs[39:], [3,3], titles=iou_str[39:])
    # mesh_val.save(f'{output_path}/val_all_images.jpg')

    mean_iou = round(np.mean(iou), 3)
    df = df.append({'img_name': 'summary', 'iou': mean_iou}, ignore_index=True)
    df.to_csv(f'{output_path}/summary.csv', float_format='{:,.2f}'.format)
    
    print(f'iou={mean_iou}')
    # print(f'iou_coarse={np.mean(np.array(iou_coarse))}')
