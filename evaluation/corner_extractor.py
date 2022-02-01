''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

from abc import abstractmethod
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import model


class CornerExtractor():
    def __init__(self, checkpoint_dir):
        self.model = model.ModelFactory.get_model("resnet", 'document')
        self.model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        print(f' model parameters: {model.count_parameters(self.model)}')

    def get(self, pil_image):
        with torch.no_grad(): 
            image_array = np.copy(pil_image)
            tl, tr, br, bl = self.get_coarse_prediction(pil_image)
            x_center, y_center = [tl[0], tr[0], br[0], bl[0]], [tl[1], tr[1], br[1], bl[1]] #TODO - refactor code to get rid of x_center y_center
            top_left, top_right, bottom_right, bottom_left = self.extract_corners_patches(image_array, x_center, y_center)
            
            #concatenate crop of image containing the corner with corner's top-left coordinates (with respect to full frame).
            top_left = (top_left, 
                        max(0, int(2 * x_center[0] - (x_center[1] + x_center[0]) / 2)),
                        max(0, int(2 * y_center[0] - (y_center[3] + y_center[0]) / 2)))
            top_right = (top_right, 
                         int((x_center[1] + x_center[0]) / 2), 
                         max(0, int(2 * y_center[1] - (y_center[1] + y_center[2]) / 2)))
            bottom_right = (bottom_right, 
                            int((x_center[2] + x_center[3]) / 2), 
                            int((y_center[1] + y_center[2]) / 2))
            bottom_left = (bottom_left, 
                           max(0, int(2 * x_center[3] - (x_center[2] + x_center[3]) / 2)),
                           int((y_center[0] + y_center[3]) / 2))

            return top_left, top_right, bottom_right, bottom_left

    # def calculate_region_extractor_coordinates(self):


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
        
    def get_coarse_prediction(self, pil_image):
        im = self.prepare_image(pil_image)
        quad_pred = self.model(im).cpu().data.numpy()[0]    
        quad_pred = self.denormalize_coordinates(pil_image, quad_pred)
        return quad_pred

    def denormalize_coordinates(self, pil_image, model_prediction):
        # model returns normalized coordinates. convert to pixels with respect to full resolution:
        w, h = pil_image.size
        x_center = model_prediction[[0, 2, 4, 6]] * w
        y_center = model_prediction[[1, 3, 5, 7]] * h
        quad = [[x_center[i], y_center[i]] for i in range(4)]
        return quad

    def prepare_image(self, pil_image):
        test_transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()]) #TODO - shouldn't it come from dataset definition?
        im = test_transform(pil_image)
        im = im.unsqueeze(0)
        if torch.cuda.is_available():
            im = im.cuda()
        return im
    
    def find_qudrilateral(self, pil_image):
        quad = self.get_coarse_prediction(pil_image)
        # model can return negative values therefore need to clip it and also integer it:
        w, h = pil_image.size
        clip_x = lambda p: max(0, min(p, w))
        clip_y = lambda p: max(0, min(p, h))
        quad = [[int(clip_x(quad[i][0])), int(clip_y(quad[i][1]))] for i in range(4)]
        return quad



