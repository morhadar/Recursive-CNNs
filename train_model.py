import os
import argparse
from tqdm import tqdm
import torch
import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter

import dataprocessor
from dataprocessor.dataset import MyDatasetCorner, MyDatasetDoc
from experiment import Experiment
import model
import trainer
import utils

parser = argparse.ArgumentParser(description='Recursive-CNNs')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.005)')
parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20, 30],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.2, 0.2, 0.2],
                    help='LR is multiplied by gamma[k] on schedule[k], number of gammas should be equal to schedule')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--pretrain', action='store_true', default=False,
                    help='Pretrain the model on CIFAR dataset?')
parser.add_argument('--load-ram', action='store_true', default=False,
                    help='Load data in ram: TODO : Remove this')
parser.add_argument('--debug', action='store_true', default=True,
                    help='Debug messages')
parser.add_argument('--seed', type=int, default=2323,
                    help='Seeds values to be used')
# parser.add_argument('--log-interval', type=int, default=5, metavar='N',
#                     help='how many batches to wait before logging training status')
parser.add_argument('--model-type', default="resnet",
                    help='model type to be used. Example : resnet32, resnet20, densenet, test')
# parser.add_argument('--output-dir', default="results/",
#                     help='Directory to store the results; a new folder "DDMMYYYY" will be created '
#                          'in the specified directory to save the results.')
parser.add_argument('--decay', type=float, default=0.00001, help='Weight decay (L2 penalty).')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs for trianing')
# parser.add_argument('--dataset', default="document", help='Dataset to be used; example document, corner')
parser.add_argument('--loader', default="hdd", 
                    help='Loader to load data; hdd for reading from the hdd and ram for loading all data in the memory')
parser.add_argument('--name', default="noname", help='Name of the experiment')

# document:
# data_path = '/media/mhadar/d/data/RecursiveCNN_data/smartdocData_DocTrainC'; dataset_name = 'document'; batch_size=32
# data_path = '/home/mhadar/projects/doc_scanner/data/data_generator/v1'; dataset_name = 'my_document'; batch_size=32
# parser.add_argument('--dataset', default = dataset_name, help='Dataset to be used; example document, corner')
# parser.add_argument("-i", "--data-dirs", nargs='+', default = data_path, help="input Directory of train data")
# parser.add_argument("-v", "--validation-dirs", nargs='+', default = data_path, help="input Directory of val data")

# corner:
# data_path = "/media/mhadar/d/data/RecursiveCNN_data/cornerTrain64"; dataset_name = 'corner'; batch_size=32
# data_path = "/home/mhadar/projects/doc_scanner/data/data_generator/v1_corners"; dataset_name = 'my_corner'; batch_size=32
data_path = "/home/mhadar/projects/doc_scanner/data/data_generator/v2_corners"; dataset_name = 'my_corner'; batch_size=32
parser.add_argument('--dataset', default = dataset_name, help='Dataset to be used; example document, corner')
parser.add_argument("-i", "--data-dirs", nargs='+', default = data_path, help="input Directory of train data")
parser.add_argument("-v", "--validation-dirs", nargs='+', default = data_path, help="input Directory of val data")

args = parser.parse_args()

args.output_dir = 'results' #TODO - can I unify directory results and runs? will it make troubles to tensorboard?
args.output_tensorboard = 'runs'
args.train_cutoff = 0.8


is_rotating = True
args.name = 'v3'

########## for debug #############
# args.output_dir = 'results/debug'
# args.output_tensorboard = 'results/debug'
# args.train_cutoff = 0.01
# args.epochs = 1
# torch.manual_seed(args.seed)
# args.no_cuda = True #for reproducibility
#################################

e = Experiment(args.name, args.output_dir, args.output_tensorboard)
writer = SummaryWriter(e.tensorboard_log_dir)
logger = utils.utils.setup_logger(e.log_file_name) #TODO - fix log so I can read it
print(e)

# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0" if use_cuda else "cpu")
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 3, 'pin_memory': True} if args.cuda else {}

if args.dataset in ['document', 'corner']: #ugly hack to support old code. #TODO - get rid of it
    training_data = dataprocessor.DatasetFactory.get_dataset(args.data_dirs, args.dataset)
    testing_data  = dataprocessor.DatasetFactory.get_dataset(args.validation_dirs, args.dataset)
    train_dataset = dataprocessor.LoaderFactory.get_loader(args.loader, training_data.myData, transform=training_data.train_transform, cuda=args.cuda)
    test_dataset  = dataprocessor.LoaderFactory.get_loader(args.loader, testing_data.myData,  transform=training_data.test_transform,  cuda=args.cuda)
    
else:
    if args.dataset == 'my_document':
        dataset = MyDatasetDoc(args.data_dirs)
    if args.dataset == 'my_corner':
        dataset = MyDatasetCorner.from_directory(args.data_dirs, is_rotating=is_rotating)
    train_dataset, test_dataset = dataset.random_split(args.train_cutoff)

train_dataloader = td.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
test_dataloader  = td.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)

myModel = model.ModelFactory.get_model(args.model_type, args.dataset)
if args.cuda:
    myModel.cuda()

# Should I pretrain the model on CIFAR?
if args.pretrain:
    trainset = dataprocessor.DatasetFactory.get_dataset(None, "CIFAR")
    train_iterator_cifar = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    # Define the optimizer used in the experiment
    cifar_optimizer = torch.optim.SGD(myModel.parameters(), args.lr, momentum=args.momentum,
                                      weight_decay=args.decay, nesterov=True)

    # Trainer object used for training
    cifar_trainer = trainer.CIFARTrainer(train_iterator_cifar, myModel, args.cuda, cifar_optimizer)

    for epoch in tqdm(range(0, 70)):
        logger.info("Epoch : %d", epoch)
        cifar_trainer.update_lr(epoch, [30, 45, 60], args.gammas)
        cifar_trainer.train(epoch)

    # Freeze the model
    counter = 0
    for name, param in myModel.named_parameters():
        # Getting the length of total layers so I can freeze x% of layers
        gen_len = sum(1 for _ in myModel.parameters())
        if counter < int(gen_len * 0.5):
            param.requires_grad = False
            logger.warning(name)
        else:
            logger.info(name)
        counter += 1

optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, myModel.parameters()), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.decay, nesterov=True)

my_trainer = trainer.Trainer(train_dataloader, myModel, args.cuda, optimizer)
my_eval = trainer.EvaluatorFactory.get_evaluator("rmse", args.cuda)

for epoch in range(args.epochs):
    logger.info("Epoch : %d", epoch)
    my_trainer.update_lr(epoch, args.schedule, args.gammas)
    lossAvg = my_trainer.train()
    writer.add_scalar('loss/train', lossAvg, epoch)
    lossAvg_test = my_eval.evaluate(my_trainer.model, test_dataloader)
    writer.add_scalar('loss/test', lossAvg_test, epoch)

# next line is for debugging (refactoring, etc). lossAvg belongs for training 'v2_corners' / 'my_document v1
# assert lossAvg == 0.519621719121933 or lossAvg == 1.5293890237808228
torch.save(myModel.state_dict(), os.path.join(e.out_path, args.name + "_" + args.model_type+ ".pb"))
e.store_json()
