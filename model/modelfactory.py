''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import model.resnet32 as resnet
import model.cornerModel as tm
import torchvision.models as models

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(model_type, dataset):
        if model_type == "resnet":
            if dataset == 'document' or dataset == 'my_document':
                return resnet.resnet20(8)
            elif dataset == 'corner' or dataset  == 'my_corner':
                return resnet.resnet20(2)
        if model_type == "resnet8":
            if dataset == 'document':
                return resnet.resnet8(8)
            elif dataset == 'corner':
                return resnet.resnet8(2)
        elif model_type == 'shallow':
            if dataset == 'document':
                return tm.MobileNet(8)
            elif dataset == 'corner':
                return tm.MobileNet(2)
        elif model_type =="squeeze":
            if dataset == 'document':
                return models.squeezenet1_1(True)
            elif dataset == 'corner':
                return models.squeezenet1_1(True)
        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)
