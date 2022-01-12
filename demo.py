''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import cv2
import numpy as np
import os

import evaluation


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath", default="../058.jpg", help="Path to the document image")
    parser.add_argument("-o", "--outputPath", default="results", help="Path to store the result")
    parser.add_argument("-rf", "--retainFactor", help="Floating point in range (0,1) specifying retain factor",
                        default="0.85")
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                        # default="../cornerModelWell")
                        default="results/trained_models/corner/26122021_corner_smartdoc/nonamecorner_resnet.pb")
    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        # default="../documentModelWell")
                        default="results/trained_models/document/22122021_document_smartdoc/nonamedocument_resnet.pb")
    return parser.parse_args()

def draw_points(points: list, im):
    """ points: [[x1, y1], [x2, y2], ....] """
    im_p = im.copy()
    for point in points:
        cv2.circle(im_p, point, 20, (255, 255, 127), 20)
    return im_p

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

if __name__ == "__main__":
    args = args_processor()

    #choose image:
    # img = cv2.imread(args.imagePath)
    # img_path = '/media/mhadar/d/data/low-level-camera/stills/WIN_20211128_11_44_28_Pro.jpg'
    # img_path = '/media/mhadar/d/data/low-level-camera/stills/WIN_20211220_15_26_08_Pro_no_motion.jpg'
    # img_path = '/media/mhadar/d/data/low-level-camera/stills/WIN_20211220_15_27_18_Pro_no_motion.jpg'
    # img_path = '/media/mhadar/d/data/low-level-camera/stills/WIN_20211220_15_27_18_Pro_no_motion.jpg'
    # img_path = '/media/mhadar/d/data/low-level-camera/stills/WIN_20211220_15_29_23_Pro_no_motion.jpg'
    # img_path = '/media/mhadar/d/data/low-level-camera/stills/WIN_20211220_15_29_30_Pro_no_motion.jpg'
    img_path = '/media/mhadar/d/data/high-level-camera/stills/WIN_20211216_09_41_32_Pro.jpg'
    # img_path = '/media/mhadar/d/data/high-level-camera/stills/WIN_20211028_05_50_31_Pro.jpg'
    
    
    img_name = os.path.basename(img_path)[:-4]
    img_orig = cv2.imread(img_path)
    suffix = ''
    # fx, fy = 20, 11 #todo - not sure this is needed
    # img = cv2.resize(img_orig, dsize=(64,64), fx=fx, fy=fy, interpolation = cv2.INTER_AREA)
    # oImg = img
    oImg = img_orig

    
    corners_extractor = evaluation.corner_extractor.GetCorners(args.documentModel)
    corner_refiner = evaluation.corner_refiner.corner_finder(args.cornerModel)
    extracted_corners = corners_extractor.get(oImg)
    # print_coarse_corners(extracted_corners)
    # extracted_corners_for_plotting = convert_coarse_corners_to_initial_gusess(extracted_corners)
    
    # Refine the detected corners using corner refiner
    corner_address = []
    image_name = 0
    for corner in extracted_corners:
        image_name += 1
        corner_img = corner[0]
        refined_corner = np.array(corner_refiner.get_location(corner_img, 0.85)) #can it be negative also??? 

        # Converting from local co-ordinate to global co-ordinates of the image
        refined_corner[0] += corner[1]
        refined_corner[1] += corner[2]

        # Final results
        corner_address.append(refined_corner)

    for a in range(0, len(extracted_corners)):
        cv2.line(oImg, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]), (255, 0, 0), 4)
    cv2.imwrite(f'{args.outputPath}/{img_name}{suffix}.jpg', oImg)
    
    # t = np.array(corner_address)
    # # t[:,0] *= fx
    # # t[:,1] *= fy
    # tl, tr, br, bl = (t[0,0], t[0,1]), (t[1,0], t[1,1]), (t[2,0], t[2,1]), (t[3,0], t[3,1])

    # im_out = draw_points((tl, tr, br, bl), img_orig)
    # cv2.line(im_out, tl, tr, (0, 0, 255), 2)
    # cv2.line(im_out, tr, br, (0, 0, 255), 2)
    # cv2.line(im_out, br, bl, (0, 0, 255), 2)
    # cv2.line(im_out, bl, tl, (0, 0, 255), 2)
    # cv2.imwrite(f'{args.outputPath}/{img_name}{suffix}.jpg', im_out)
    # print('Done')
    # # cv2.imwrite(args.outputPath, oImg)
