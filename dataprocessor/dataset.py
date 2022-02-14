import csv
import pandas as pd
import logging
import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.utils.data as td

import utils.utils as utils

# To incdude a new Dataset, inherit from Dataset and add all the Dataset specific parameters here.
# Goal : Remove any data specific parameters from the rest of the code

logger = logging.getLogger('iCARL')

def random_split(dataset, train_cutoff = 0.8):
    N = len(dataset)
    N_train = int(train_cutoff*N)
    N_test = N - N_train
    train_dataset, test_dataset = td.random_split(dataset, (N_train, N_test))
    return train_dataset, test_dataset

class DatasetBase():
    ''' Base class to represent a Dataset
    data - list of strings. paths to images. N is the number of files.
    labels - np array Nx8 (topleft_x, topleft_y, topright_x, topright_y, botleft_x, botleft_y, botright_x, botright_y). normalized coordinated.
    '''

    def __init__(self, name):
        self.name = name
        self.data = []
        self.labels = []
    
    def __repr__(self) -> None:
        logger.debug("Ground Truth Shape: %s", str(self.labels.shape))
        logger.debug("Data shape %s", str(len(self.data)))

class SmartDoc(DatasetBase):
    '''
    '''
    def __init__(self, directories=[]):
        super().__init__("smartdoc")
        data = []
        labels = []
        self.train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                    transforms.ColorJitter(1.5, 1.5, 0.9, 0.5),
                                                    transforms.ToTensor()])

        self.test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                    transforms.ToTensor()])
        logger.info("Pass train/test data paths here")
        directories = [directories] if isinstance(directories, str) else directories
        for d in directories:
            print (d, "gt.csv")
            with open(os.path.join(d, "gt.csv"), 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                import ast
                for row in spamreader:
                    data.append(os.path.join(d, row[0]))
                    test = row[1].replace("array", "")
                    labels.append(ast.literal_eval(test))
        labels = np.reshape(np.array(labels), (-1, 8))
        self.myData = [data, labels]

        self.data = data #TODO: delete this once get rid of the those redundant members
        self.labels = labels
        
        self.__repr__()

class MyDatasetDoc(td.Dataset):
    '''
    data - list of strings. path to images. N is the number of files.
    labels - np array Nx8 (topleft_x, topleft_y, topright_x, topright_y, botleft_x, botleft_y, botright_x, botright_y). normalized coordinated.
    '''
    def __init__(self, d):
        gt_colunms = ['img_name', 'topleft_x', 'topleft_y', 'topright_x', 'topright_y', 'botright_x', 'botright_y', 'botleft_x', 'botleft_y', 'w', 'h', 'ignore']
        self.train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                    transforms.ColorJitter(1.5, 1.5, 0.9, 0.5),
                                                    transforms.ToTensor()])

        self.test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                    transforms.ToTensor()])
        df = pd.read_csv(f'{d}/gt.csv', names=gt_colunms)
        df = df.drop(df[df.ignore == 1].index)
        self.data = list(d +'/'+ df.img_name)
        df[['topleft_x', 'topright_x', 'botright_x', 'botleft_x']] = df[['topleft_x', 'topright_x', 'botright_x', 'botleft_x']].div(df.w, axis=0)
        df[['topleft_y', 'topright_y', 'botright_y', 'botleft_y']] = df[['topleft_y', 'topright_y', 'botright_y', 'botleft_y']].div(df.h, axis=0)
        self.target = df.drop(['img_name', 'w', 'h', 'ignore'],axis=1).to_numpy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = Image.open(self.data[index])
        img = img.convert('RGB') if img.mode == 'L' else img
        if self.train_transform is not None:
            img = self.train_transform(img)
        target = self.target[index]
        return img, target
    
class MyDatasetCorner(td.Dataset):
    '''
    data - list of strings. path to images. N is the number of files.
    target - np array Nx2 (x, y). normalized coordinated.
    '''
    def __init__(self, d, is_rotating=False):
        """
        d - path to directory containing images and a csv gt file.
        is_rotating(bool) - if True rotate all samples to be topleft corners
        """
        gt_colunms = ['img_name', 'corner_type', 'x', 'y', 'w', 'h', 'ignore']
        self.train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                   transforms.ColorJitter(0.5, 0.5, 0.5, 0.5), #TODO - consider removing this!!! color is a good feature!
                                                   transforms.ToTensor()])

        self.test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                  transforms.ToTensor()])
        df = pd.read_csv(f'{d}/gt.csv', names=gt_colunms)
        df = df.drop(df[df.ignore == 1].index)
        # df = df.drop(df[df.corner_type != 'topleft'].index) #topleft, topright, botright, botleft #TODO - MAKE IT CONFIGURABLE FROM OUTSIDE!
        df[['x']] = df[['x']].div(df.w, axis=0)
        df[['y']] = df[['y']].div(df.h, axis=0)
        
        self.df = df
        self.data = list(d +'/'+ df.img_name)
        self.target = df.drop(['img_name', 'corner_type', 'w', 'h', 'ignore'],axis=1).to_numpy()
        
        self.is_rotating = is_rotating
        self.rotation_angles_to_be_topleft_corner= {'topleft': 0, 'topright':90, 'botright': 180, 'botleft':-90}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img = self.get_pil_image(index)
        img = img.convert('RGB') if img.mode == 'L' else img
        target = self.target[index]
        if self.is_rotating:
            angle = self.rotation_angles_to_be_topleft_corner[self.get_corner_type(index)]
            new_target_in_pixels = utils.rotate_translate_point(target * img.size, angle, img.size)
            img = img.rotate(angle, expand=True)
            target = np.array(new_target_in_pixels) / img.size
        if self.train_transform is not None:
            img = self.train_transform(img)
        return img, target
    
    def get_name(self, index):
        return self.data[index]
    
    def get_pil_image(self, index):
        return Image.open(self.get_name(index))
    
    def get_corner_type(self, index):
        return self.df.iloc[index]['corner_type']
    
class SmartDocDirectories(DatasetBase):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        for folder in os.listdir(directory):
            if (os.path.isdir(directory + "/" + folder)):
                for file in os.listdir(directory + "/" + folder):
                    images_dir = directory + "/" + folder + "/" + file
                    if (os.path.isdir(images_dir)):

                        list_gt = []
                        tree = ET.parse(images_dir + "/" + file + ".gt")
                        root = tree.getroot()
                        for a in root.iter("frame"):
                            list_gt.append(a)

                        im_no = 0
                        for image in os.listdir(images_dir):
                            if image.endswith(".jpg"):
                                # print(im_no)
                                im_no += 1

                                # Now we have opened the file and GT. Write code to create multiple files and scale gt
                                list_of_points = {}

                                # img = cv2.imread(images_dir + "/" + image)
                                self.data.append(os.path.join(images_dir, image))

                                for point in list_gt[int(float(image[0:-4])) - 1].iter("point"):
                                    myDict = point.attrib

                                    list_of_points[myDict["name"]] = (
                                        int(float(myDict['x'])), int(float(myDict['y'])))

                                ground_truth = np.asarray(
                                    (list_of_points["tl"], list_of_points["tr"], list_of_points["br"],
                                     list_of_points["bl"]))
                                ground_truth = utils.sort_gt(ground_truth)
                                self.labels.append(ground_truth)

        self.labels = np.array(self.labels)
        self.labels = np.reshape(self.labels, (-1, 8))
        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])
        
        self.__repr__()

class SelfCollectedDataset(DatasetBase):
    '''
    Class to include MNIST specific details
    '''

    def __init__(self, directory="data"):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []

        for image in os.listdir(directory):
            # print (image)
            if image.endswith("jpg") or image.endswith("JPG"):
                if os.path.isfile(os.path.join(directory, image + ".csv")):
                    with open(os.path.join(directory, image + ".csv"), 'r') as csvfile:
                        spamwriter = csv.reader(csvfile, delimiter=' ',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)

                        img_path = os.path.join(directory, image)

                        gt = []
                        for row in spamwriter:
                            gt.append(row)
                        gt = np.array(gt).astype(np.float32)
                        ground_truth = utils.sort_gt(gt)
                        self.labels.append(ground_truth)
                        self.data.append(img_path)

        self.labels = np.array(self.labels)
        self.labels = np.reshape(self.labels, (-1, 8))
        self.myData = []
        for a in range(len(self.data)):
            self.myData.append([self.data[a], self.labels[a]])

        self.__repr__()

class SmartDocCorner(DatasetBase):
    '''
    data - list of strings. path to images. N is the number of files.
    labels - np array Nx2 (x, y). normalized coordinated.
    '''

    def __init__(self, directories=[]):
        super().__init__("smartdoc")
        self.data = []
        self.labels = []
        directories = [directories] if isinstance(directories, str) else directories
        for d in directories:
            self.directory = d
            self.train_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                       transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
                                                       transforms.ToTensor()])

            self.test_transform = transforms.Compose([transforms.Resize([32, 32]),
                                                      transforms.ToTensor()])

            logger.info("Pass train/test data paths here")

            file_names = []
            with open(os.path.join(self.directory, "gt.csv"), 'r') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                import ast
                for row in spamreader:
                    file_names.append(row[0])
                    self.data.append(os.path.join(self.directory, row[0]))
                    test = row[1].replace("array", "")
                    self.labels.append((ast.literal_eval(test)))
        self.labels = np.array(self.labels)
        self.labels = np.reshape(self.labels, (-1, 2))

        self.myData = [self.data, self.labels]
        
        self.__repr__()
