import logging

from torch.autograd import Variable

logger = logging.getLogger('iCARL')
import torch.nn.functional as F
import torch
from tqdm import tqdm
import numpy as np


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self):
        pass



class Trainer(GenericTrainer):
    def __init__(self, train_iterator, model, cuda, optimizer):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * gammas[temp]
                    logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
                                 self.current_lr * gammas[temp])
                    self.current_lr *= gammas[temp]

    def train(self):
        self.model.train()
        lossAvg = []
        for img, target in tqdm(self.train_iterator):
            if self.cuda:
                img, target = img.cuda(), target.cuda() #TODO - isn't it already in cuda?
            self.optimizer.zero_grad()
            response = self.model(Variable(img))
            loss = torch.sqrt(F.mse_loss(response, Variable(target.float())))
            loss.backward()
            self.optimizer.step()
            
            lossAvg.append(loss.item())

        lossAvg = np.mean(lossAvg)
        logger.info(f'Avg Loss {lossAvg:.3f}')
        return lossAvg


class CIFARTrainer(GenericTrainer):
    def __init__(self, train_iterator, model, cuda, optimizer):
        super().__init__()
        self.cuda = cuda
        self.train_iterator = train_iterator
        self.model = model
        self.optimizer = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()

    def update_lr(self, epoch, schedule, gammas):
        for temp in range(0, len(schedule)):
            if schedule[temp] == epoch:
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                    param_group['lr'] = self.current_lr * gammas[temp]
                    logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
                                 self.current_lr * gammas[temp])
                    self.current_lr *= gammas[temp]

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, targets in tqdm(self.train_iterator):
            if self.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(Variable(inputs), pretrain=True)
            loss = self.criterion(outputs, Variable(targets))
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        logger.info("Accuracy : %s", str((correct * 100) / total))
        return correct / total
