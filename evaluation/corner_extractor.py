import numpy as np
import torch
from torchvision import transforms

import model


class CornersCoarseEstimation():
    def __init__(self, checkpoint_dir):
        self.model = model.ModelFactory.get_model("resnet", 'document')
        self.model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])
              
        print(f' model parameters: {model.count_parameters(self.model)}')

    def get(self, pil_image):
        with torch.no_grad(): #TODO - why we need this???
            quad_pred = self.get_model_prediction(pil_image)
            tl_patch_coord, tr_patch_coord, br_patch_coord, bl_patch_coord = self.calculate_coordinates_for_patches_extraction(quad_pred)
            tl_patch_coord, tr_patch_coord, br_patch_coord, bl_patch_coord = [self.clip_and_integer_coordinates(i, pil_image.size) for i in (tl_patch_coord, tr_patch_coord, br_patch_coord, bl_patch_coord)]
            top_left_patch, top_right_patch, bottom_right_patch, bottom_left_patch = self.extract_patches_from_image(pil_image, coords=(tl_patch_coord, tr_patch_coord, br_patch_coord, bl_patch_coord))
            #TODO - this is temporarly just to check model2 by itself.
            # top_left_patch, top_right_patch, bottom_right_patch, bottom_left_patch = self.slice_image_to_4_patches(pil_image)

            # arrange output for the next stage -> the input for the CornerRefiner: tuples of corner_patch with top-left coordinates of the patch with respect to full frame:
            top_left =      (top_left_patch,      tl_patch_coord[0][0], tl_patch_coord[0][1])
            top_right =     (top_right_patch,     tr_patch_coord[0][0], tr_patch_coord[0][1])
            bottom_right =  (bottom_right_patch,  br_patch_coord[0][0], br_patch_coord[0][1])
            bottom_left =   (bottom_left_patch,   bl_patch_coord[0][0], bl_patch_coord[0][1])

            #save intermediate results for visualization:
            self.patches_coords = tl_patch_coord, tr_patch_coord, br_patch_coord, bl_patch_coord
            self.quad_pred = self.clip_and_integer_coordinates(quad_pred, pil_image.size) #TODO - should those two be here or in get_model_prediction or in both places??? 

            return top_left, top_right, bottom_right, bottom_left

    @staticmethod
    def extract_patches_from_image_by_coordinates(pil_image, coords):
        return [np.array(pil_image.crop((i[0][0], i[0][1], i[1][0], i[1][1]))) for i in coords]

    def slice_image_to_4_patches(self, pil_image):
        w, h = pil_image.size
        mid_w, mid_h = w//2, h//2
        coords = (([0,       0],     [mid_w,  mid_h]), #top-left
                  ([mid_w,   0],     [w,      mid_h]), #top-right
                  ([mid_w,   mid_h], [ w,      h]), #bottom-right
                  ([0,       mid_h], [ mid_w,  h]) #bottom-left
        )
        return self.extract_patches_from_image_by_coordinates(pil_image, coords)

        
    # def calculate_region_extractor_coordinates(self):
    def calculate_coordinates_for_patches_extraction(self, quad):
        tl, tr, br, bl = quad
        tl_coord = self.calc_patch_box_coordinates(tl, (tr[0]-tl[0]), (bl[1]-tl[1]))
        tr_coord = self.calc_patch_box_coordinates(tr, (tr[0]-tl[0]), (br[1]-tr[1]))
        br_coord = self.calc_patch_box_coordinates(br, (br[0]-bl[0]), (br[1]-tr[1]))
        bl_coord = self.calc_patch_box_coordinates(bl, (br[0]-bl[0]), (bl[1]-tl[1]))
        return tl_coord, tr_coord, br_coord, bl_coord

    def extract_corners_patches2(pil_image, coords):
        return [pil_image.crop(i) for i in coords]
            
    @staticmethod
    def calc_patch_box_coordinates(patch_middle_point, patch_w, patch_h):
        x_center, y_center = patch_middle_point
        # half_w, half_h = patch_w//2, patch_h//2
        left = x_center - patch_w//2
        top = y_center - patch_h//2
        right = left + patch_w
        bottom = top + patch_h
        return [[left, top], [right, bottom]]
        
    def extract_corners_patches(self, image_array, x_center, y_center):
        """ Extract four corners of the image. Read "Region Extractor" in Section III of the paper for an explanation """
        h = image_array.shape[0]
        w = image_array.shape[1]
        top_left = image_array[max(0, int(2 * y_center[0] - (y_center[3] + y_center[0]) / 2))        : int((y_center[3] + y_center[0]) / 2),
                               max(0, int(2 * x_center[0] - (x_center[1] + x_center[0]) / 2))        : int((x_center[1] + x_center[0]) / 2)]

        top_right = image_array[max(0, int(2 * y_center[1] - (y_center[1] + y_center[2]) / 2))       : int((y_center[1] + y_center[2]) / 2),
                                int((x_center[1] + x_center[0]) / 2)                                 : min(w - 1, int(x_center[1] + (x_center[1] - x_center[0]) / 2))]

        bottom_right = image_array[int((y_center[1] + y_center[2]) / 2)                              : min(h - 1, int(y_center[2] + (y_center[2] - y_center[1]) / 2)),
                                   int((x_center[2] + x_center[3]) / 2)                              : min(w - 1, int(x_center[2] + (x_center[2] - x_center[3]) / 2))]

        bottom_left = image_array[int((y_center[0] + y_center[3]) / 2)                               : min(h - 1, int(y_center[3] + (y_center[3] - y_center[0]) / 2)),
                                      max(0, int(2 * x_center[3] - (x_center[2] + x_center[3]) / 2)) : int((x_center[3] + x_center[2]) / 2)]
                          
        return top_left, top_right, bottom_right, bottom_left

    @DeprecationWarning
    def find_quadrilateral(self, pil_image):
        quad = self.get_model_prediction(pil_image)
        quad = self.clip_and_integer_coordinates(quad, pil_image.size)
        return quad
       
    def get_model_prediction(self, pil_image):
        im = self.prepare_image_for_model(pil_image)
        quad_pred = self.model(im).data.tolist()
        quad_pred = self.reshape_list(quad_pred)
        quad_pred = self.cvt_normalized_coordinates_to_pixels_coordinates(quad_pred, pil_image.size)
        return quad_pred
    
    def prepare_image_for_model(self, pil_image):
        im = self.transform(pil_image)
        im = im.unsqueeze(0)
        if torch.cuda.is_available():
            im = im.cuda()
        return im

    @staticmethod
    def reshape_list(quad):
        return [(quad[0][i], quad[0][i+1]) for i in range(0,8,2)]

    @staticmethod
    def cvt_normalized_coordinates_to_pixels_coordinates(quad, im_dimentions):
        w, h = im_dimentions
        quad = [(quad[i][0] * w, quad[i][1] * h) for i in range(4)]
        return quad

    @staticmethod
    def clip_and_integer_coordinates(quad, im_dimentions):
        """
        quad(list[list]): [[x1, y1], [x2, y2], ....]
        im_dimentions(tuple): w,h 
        """
        w, h = im_dimentions
        clip_x = lambda p: max(0, min(p, w))
        clip_y = lambda p: max(0, min(p, h))
        quad = [(int(clip_x(quad[i][0])), int(clip_y(quad[i][1]))) for i in range(len(quad))]
        return quad
        



