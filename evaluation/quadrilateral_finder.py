import numpy as np
from evaluation import CornersCoarseEstimation, CornerRefiner

class QudrilateralFinder():
    def __init__(self, document_model, corner_model, retainFactor=0.85) -> None:
        self.corner_extractor = CornersCoarseEstimation(document_model)
        self.corner_refiner = CornerRefiner(corner_model)
        self.retainFactor = retainFactor

    def find_qudrilateral(self, im):
        extracted_corners = self.corner_extractor.get(im)
        corner_address = []
        for corner in extracted_corners:
            refined_corner = np.array(self.corner_refiner.get_location(corner[0], float(self.retainFactor)))
            refined_corner[0] += corner[1] # Converting from local co-ordinate to global co-ordinates of the image
            refined_corner[1] += corner[2]

            corner_address.append(tuple(refined_corner))
        
        # save intermediate results for visualization:
        self.quad_pred_coarse = self.corner_extractor.quad_pred
        self.patches_coords = self.corner_extractor.patches_coords

        return corner_address