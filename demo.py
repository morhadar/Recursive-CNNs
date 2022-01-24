''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import cv2
import numpy as np
import os
import argparse
from shapely.geometry import Polygon

import evaluation

def args_processor():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--imagePath", default="../058.jpg", help="Path to the document image")
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

def draw_points(points: list, im): #TODO - take from the other project!!
    """ points: [[x1, y1], [x2, y2], ....] """
    im_p = im.copy()
    for point in points:
        cv2.circle(im_p, point, 20, (255, 255, 127), 20)
    return im_p

def draw_qudrilateral(points:list, im):
    """ 
    points: [[tl], [tr], [br], [bl]]
    im: np.array
    """
    im = draw_points(points, im)
    tl, tr, br, bl = tuple(points)
    cv2.line(im, tl, tr, (255, 0, 0), 4)
    cv2.line(im, tr, br, (255, 0, 0), 4)
    cv2.line(im, br, bl, (255, 0, 0), 4)
    cv2.line(im, bl, tl, (255, 0, 0), 4)
    return im


def print_coarse_corners(corners):
    def _print_corner(corner_name, corner):
        print(f'{corner_name}: x={corner[1]},y={corner[2]}')
    print('top left corner for each cropped image!!')
    _print_corner('topleft',  corners[0])
    _print_corner('topright', corners[1])
    _print_corner('botright', corners[2])
    _print_corner('botleft',  corners[3])

def convert_coarse_corners_to_initial_gusess(corners): #TODO - find better name
    def _cvt_corner(corner):
        h,w = corner[0].shape[:2]
        x = int(corner[1] + w/2)
        y = int(corner[2] + h/2)
        return x,y
    return _cvt_corner(corners[0]), _cvt_corner(corners[1]), _cvt_corner(corners[2]), _cvt_corner(corners[3])

def find_document_quadrilateral(im):
    corners_extractor = evaluation.corner_extractor.GetCorners(args.documentModel)
    corner_refiner = evaluation.corner_refiner.corner_finder(args.cornerModel)
    
    extracted_corners = corners_extractor.get(im)
    corner_address = []
    for corner in extracted_corners:
        refined_corner = np.array(corner_refiner.get_location(corner[0], float(args.retainFactor))) #TODO - can the refined corner be negative (or bigger image dimentions)??? 
        refined_corner[0] += corner[1] # Converting from local co-ordinate to global co-ordinates of the image
        refined_corner[1] += corner[2]

        corner_address.append(tuple(refined_corner))
    return corner_address

def IOU(pol1_xy, pol2_xy):
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union

if __name__ == "__main__":
    imgs = [
    '/media/mhadar/d/data/self_collected/low-level-camera/stills/00000.png',
    ]

    img_suffix = ''
    output_path = 'results/my_results_bgr'
     
    args = args_processor()
    os.makedirs(output_path, exist_ok=True)
    for img_path in imgs:
        img_name = os.path.basename(img_path)[:-4]
        oImg = cv2.imread(img_path)
        # oImg = cv2.cvtColor(oImg, cv2.COLOR_BGR2RGB)

        corner_address = find_document_quadrilateral(oImg)
        IOU(corner_address, corner_address)
        print(corner_address)
        oImg = draw_qudrilateral(corner_address, oImg)
        # oImg = cv2.cvtColor(oImg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{output_path}/{img_name}{img_suffix}.jpg', oImg)
