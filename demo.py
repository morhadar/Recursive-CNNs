from PIL import Image
import os
import glob
import argparse
from evaluation import QudrilateralFinder
from utils import mesh_imgs, draw_polygon_pil

def args_processor():
    parser = argparse.ArgumentParser(description='demo for docscanner')
    parser.add_argument("-i", "--imagePath", default="z_ref_doc_scanner/data/self_collected/high-level-camera/stills/", help="Path to the document image")
    # parser.add_argument("-i", "--imagePath", default="z_ref_doc_scanner/data/self_collected/high-level-camera/stills/high_00012.jpg", help="Path to the document image")
    parser.add_argument("-o", "--outputPath", default="results/demo/", help="Path to store the result")
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement", default="trained_models/corner/v3_Feb14_08-50-21/v3_resnet.pb")
    parser.add_argument("-m", "--mesh", action='store_true', help="save images in mesh")
    return parser.parse_args()

if __name__ == "__main__":
    args = args_processor()
    os.makedirs(args.outputPath, exist_ok=True)
    imgs = [args.imagePath] if os.path.isfile(args.imagePath) else (glob.glob(f'{args.imagePath}*.jpg') + glob.glob(f'{args.imagePath}*.png'))
    imgs_pil = []
    for im_path in imgs:
        im = Image.open(im_path)
        qf = QudrilateralFinder(None, args.cornerModel)
        quad_pred = qf.find_quad_model2_only_by_top_left(im)
        draw_polygon_pil(im, quad_pred, outline='red', width=3)

        im.save(f'{args.outputPath}/{os.path.basename(im_path)}')
        print(f'prediced qudrilateral: {os.path.basename(im_path)} -- {quad_pred}')
        imgs_pil.append(im)
    mesh = mesh_imgs(imgs_pil)
    mesh.save(f'{args.outputPath}/mesh.jpg')

