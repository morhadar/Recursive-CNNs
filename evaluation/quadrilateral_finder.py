import numpy as np
from PIL import Image
from evaluation import CornersCoarseEstimation, CornerRefiner
# (ugly) Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from utils import clip_and_integer_coordinates, rotate_translate_point

class QudrilateralFinder():
    def __init__(self, document_model, corner_model, retainFactor=0.85) -> None:
        if document_model is not None:
            self.corner_extractor = CornersCoarseEstimation(document_model)
        self.corner_refiner_model = CornerRefiner(corner_model)
        self.retainFactor = retainFactor #TODO - consider moving retain factor to the init of CornerRefiner

    def find_quad(self, pil_image):
        quad_coarse = self.corner_extractor.get_prediction(pil_image)
        corners_patches = self.get_patches(pil_image, quad_coarse)
        quad_refined = [self.refine_corner(corner) for corner in corners_patches]

        # save intermediate results for visualization:
        self.quad_coarse = clip_and_integer_coordinates(quad_coarse, pil_image.size)
        
        return quad_refined
    
    def find_quad_model2_only(self, pil_image):
        corners_patches = self.slice_image_to_4_patches(pil_image)
        quad_refined = [self.refine_corner(corner) for corner in corners_patches]
        return quad_refined

    def find_quad_model2_only_by_top_left(self, pil_image):
        corners_patches = self.slice_image_to_4_patches(pil_image)
        quad_refined = [self.refine_corner2(corner, corner_type) for corner, corner_type in zip(corners_patches, ('topleft', 'topright', 'botright', 'botleft'))]
        return quad_refined

    ##########################################################################
    # region extractor:
    def get_patches(self, pil_image, quad_pred):
        coords = self.calc_coordinates_for_patches_extraction(quad_pred) #coords = tl_coord, tr_coord, br_coord, bl_coord
        coords = [clip_and_integer_coordinates(coord, pil_image.size) for coord in coords]
        patches = self.extract_patches(pil_image, coords) #patches = tl_patch, tr_patch, br_patch, bl_patch

        # save intermediate results for visualization:
        self.patches_coords = coords

        return self.reshape(patches, coords)
    
    def calc_coordinates_for_patches_extraction(self, quad):
        tl, tr, br, bl = quad
        tl_coord = self.calc_patch_box_coordinates(tl, (tr[0]-tl[0]), (bl[1]-tl[1]))
        tr_coord = self.calc_patch_box_coordinates(tr, (tr[0]-tl[0]), (br[1]-tr[1]))
        br_coord = self.calc_patch_box_coordinates(br, (br[0]-bl[0]), (br[1]-tr[1]))
        bl_coord = self.calc_patch_box_coordinates(bl, (br[0]-bl[0]), (bl[1]-tl[1]))
        return tl_coord, tr_coord, br_coord, bl_coord
    
    @staticmethod
    def calc_patch_box_coordinates(patch_middle_point, patch_w, patch_h):
        x_center, y_center = patch_middle_point
        left = x_center - patch_w//2
        top = y_center - patch_h//2
        right = left + patch_w
        bottom = top + patch_h
        return [[left, top], [right, bottom]]
    
    @staticmethod
    def extract_patches(pil_image, coords):
        return [np.array(pil_image.crop((i[0][0], i[0][1], i[1][0], i[1][1]))) for i in coords]
    
    def reshape(self, patches, coords):
        # arrange output for the next stage -> the input for the CornerRefiner: tuples of corner_patch with top-left coordinates of the patch with respect to full frame:
        tl_patch, tr_patch, br_patch, bl_patch = patches
        tl_coord, tr_coord, br_coord, bl_coord = coords
        
        top_left =      (tl_patch, tl_coord[0][0], tl_coord[0][1])
        top_right =     (tr_patch, tr_coord[0][0], tr_coord[0][1])
        bottom_right =  (br_patch, br_coord[0][0], br_coord[0][1])
        bottom_left =   (bl_patch, bl_coord[0][0], bl_coord[0][1])

        return top_left, top_right, bottom_right, bottom_left


    def slice_image_to_4_patches(self, pil_image):
        w, h = pil_image.size
        mid_w, mid_h = w//2, h//2
        coords = ([(0,       0),     (mid_w,  mid_h)], #top-left
                  [(mid_w,   0),     (w,      mid_h)], #top-right
                  [(mid_w,   mid_h), ( w,      h)], #bottom-right
                  [(0,       mid_h), ( mid_w,  h)] #bottom-left
        )
        patches = self.extract_patches(pil_image, coords)

        # save intermediate results for visualization:
        self.patches_coords = coords

        return self.reshape(patches, coords)   
    
    ##########################################################################

    def refine_corner(self, corner_patch):
        im, tl_x, tl_y = corner_patch
        corner_xy = np.array(self.corner_refiner_model.get_location(im, float(self.retainFactor))) 
        corner_xy += (tl_x, tl_y)
        return tuple(corner_xy)
    
    def refine_corner2(self, corner_patch, corner_type):
        self.angle_to_topleft= {'topleft': 0, 'topright':90, 'botright': 180, 'botleft':-90}
        im, tl_x, tl_y = corner_patch
        
        im_topleft = np.array(Image.fromarray(im).rotate(self.angle_to_topleft[corner_type], expand=True))
        corner_xy_topleft = np.array(self.corner_refiner_model.get_location(im_topleft, float(self.retainFactor)))
        
        self.angle_from_topleft= {'topleft': 0, 'topright':-90, 'botright': 180, 'botleft':90}
        h,w = im_topleft.shape[:2]
        corner_xy = np.array(rotate_translate_point(corner_xy_topleft, self.angle_from_topleft[corner_type], (w,h)))
        
        # for debugging:
        # from utils import draw_circle_pil
        # out1 = Image.fromarray(im_topleft)
        # draw_circle_pil(out1, corner_xy_topleft, outline='red')
        # out2 = Image.fromarray(im)
        # draw_circle_pil(out2, corner_xy, outline='red')
        
        corner_xy += (tl_x, tl_y)
        return tuple(corner_xy)
        