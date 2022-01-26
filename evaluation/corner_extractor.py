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
            # pil_image = Image.fromarray(pil_image) #TODO - why???
            test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                 transforms.ToTensor()])
            img_temp = test_transform(pil_image)

            img_temp = img_temp.unsqueeze(0)
            if torch.cuda.is_available():
                img_temp = img_temp.cuda()

            model_prediction = self.model(img_temp).cpu().data.numpy()[0]

            model_prediction = np.array(model_prediction) #model return normalized coordinates

            x_cords = model_prediction[[0, 2, 4, 6]]
            y_cords = model_prediction[[1, 3, 5, 7]]

            x_cords = x_cords * image_array.shape[1]
            y_cords = y_cords * image_array.shape[0]

            # Extract the four corners of the image. Read "Region Extractor" in Section III of the paper for an explanation.
            top_left = image_array[
                       max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2)):int((y_cords[3] + y_cords[0]) / 2),
                       max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2)):int((x_cords[1] + x_cords[0]) / 2)]

            top_right = image_array[
                        max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2)):int((y_cords[1] + y_cords[2]) / 2),
                        int((x_cords[1] + x_cords[0]) / 2):min(image_array.shape[1] - 1,
                                                               int(x_cords[1] + (x_cords[1] - x_cords[0]) / 2))]

            bottom_right = image_array[int((y_cords[1] + y_cords[2]) / 2):min(image_array.shape[0] - 1, int(
                y_cords[2] + (y_cords[2] - y_cords[1]) / 2)),
                           int((x_cords[2] + x_cords[3]) / 2):min(image_array.shape[1] - 1,
                                                                  int(x_cords[2] + (x_cords[2] - x_cords[3]) / 2))]

            bottom_left = image_array[int((y_cords[0] + y_cords[3]) / 2):min(image_array.shape[0] - 1, int(
                y_cords[3] + (y_cords[3] - y_cords[0]) / 2)),
                          max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2)):int(
                              (x_cords[3] + x_cords[2]) / 2)]
            
            #concatenate crop of image containing the corner with corner's top-left coordinates (with respect to full frame).
            top_left = (top_left, max(0, int(2 * x_cords[0] - (x_cords[1] + x_cords[0]) / 2)),
                        max(0, int(2 * y_cords[0] - (y_cords[3] + y_cords[0]) / 2)))
            top_right = (
            top_right, int((x_cords[1] + x_cords[0]) / 2), max(0, int(2 * y_cords[1] - (y_cords[1] + y_cords[2]) / 2)))
            bottom_right = (bottom_right, int((x_cords[2] + x_cords[3]) / 2), int((y_cords[1] + y_cords[2]) / 2))
            bottom_left = (bottom_left, max(0, int(2 * x_cords[3] - (x_cords[2] + x_cords[3]) / 2)),
                           int((y_cords[0] + y_cords[3]) / 2))

            return top_left, top_right, bottom_right, bottom_left
    
    @staticmethod
    def find_coarse_corner_estimation(corners):
        """ 
        each corner is a list of: [np.array of the corner, topleft_x, topleft_y]. 
        where topleft_x, topleft_y are with respect to full resolution image 
        """
        def _cvt_corner(corner):
            h,w = corner[0].shape[:2]
            x = int(corner[1] + w/2)
            y = int(corner[2] + h/2)
            return [x,y]
        return [_cvt_corner(corners[0]), _cvt_corner(corners[1]), _cvt_corner(corners[2]), _cvt_corner(corners[3])]

