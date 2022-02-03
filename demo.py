from PIL import Image
import numpy as np
import os
import argparse
from evaluation import QudrilateralFinder

# imports from my other projects:
import os, sys
sys.path.append(os.path.abspath('z_ref_doc_scanner'))  #TODO - why importing fails if i put it under referneces folder?
from z_ref_doc_scanner.dataset import Dataset
from z_ref_doc_scanner.utils import draw_qudrilateral, IOU

def args_processor():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--imagePath", default="../058.jpg", help="Path to the document image") #TODO - enbale this 
    # parser.add_argument("-o", "--outputPath", default="results/debug/", help="Path to store the result")
    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor", default="0.85")
    #document:
    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        # default="results/trained_models/document/22122021_document_smartdoc/nonamedocument_resnet.pb")
                        default="results/trained_models/document/1212022_document_v1/nonamemy_document_resnet.pb")
    #corner:
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                        # default="results/trained_models/corner/26122021_corner_smartdoc/nonamecorner_resnet.pb")
                        default="results/trained_models/corner/17122022_corner_mycorners/nonamemy_corner_resnet.pb")
    return parser.parse_args()

if __name__ == "__main__":
    args = args_processor()
    output_path = 'results/my_results_bgr'
    img_suffix = ''
    
    os.makedirs(output_path, exist_ok=True)
    
    ds = Dataset('z_ref_doc_scanner/data/self_collected/low-level-camera/stills/')
    im, quad_true = ds.readimage(4)
    img_name = ds.get_name(4)
    
    qf = QudrilateralFinder(args.documentModel, args.cornerModel, args.retainFactor)
    quad_pred = qf.find_quad(im)
    
    oim = Image.fromarray(draw_qudrilateral(quad_pred, np.array(im)))
    oim.save(f'{output_path}/{img_name}{img_suffix}.jpg')
    iou = IOU(quad_true, quad_pred)
    
    print(f'exected qudrilateral: {quad_true}')
    print(f'prediced qudrilateral: {quad_pred}')
    print(f'iou={iou:.02f}')
