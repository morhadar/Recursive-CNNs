import numpy as np
import torch
from torchvision import transforms

import model
# Path hack. #TODO - ugly hack. get rid of it
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from utils import clip_and_integer_coordinates


class CornersCoarseEstimation():
    def __init__(self, checkpoint_dir):
        self.model = model.ModelFactory.get_model("resnet", 'document')
        self.model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])
              
        print(f' model parameters: {model.count_parameters(self.model)}')

    def get_prediction(self, pil_image):
        im = self.prepare_image_for_model(pil_image)
        quad_pred = self.model(im).data.tolist()
        quad_pred = self.reshape_list(quad_pred)
        quad_pred = self.normalized_to_pixels(quad_pred, pil_image.size)
        return quad_pred
        
    @DeprecationWarning
    def find_quadrilateral(self, pil_image):
        quad = self.get_prediction(pil_image)
        quad = clip_and_integer_coordinates(quad, pil_image.size)
        return quad
   
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
    def normalized_to_pixels(quad, im_dimentions):
        w, h = im_dimentions
        quad = [(quad[i][0] * w, quad[i][1] * h) for i in range(4)]
        return quad


        



