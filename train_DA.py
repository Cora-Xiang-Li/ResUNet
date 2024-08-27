'''
    Description: This is modified training process with DA
    Date: 06/08/2024
    Author: Cora Li
'''
import datetime
import os
import time
import shutil
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import math

from DomainAdversarialLoss import DomainAdversarialLoss

import torch
from config import ArgParser, Constants
from model import ResUNet, ResNeXt
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from utils import AverageMeter, Model_Logger

from data import Counting_dataset

logger = Model_Logger('train')
logger.enable_exception_hook()
writer = SummaryWriter()

# Define a Convolutional Domain Discriminator
class ConvDomainDiscriminator(nn.Module):
    def __init__(self, in_channels=8, feature_dim=4, hidden_size=256):
        super(ConvDomainDiscriminator, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, hidden_size, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_size)
        self.conv2 = nn.Conv3d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(hidden_size)
        # Assuming the feature_dim is reduced due to strides, adjust accordingly
        reduced_dim = feature_dim // 4
        self.fc = nn.Linear(hidden_size * reduced_dim**3, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.sigmoid(x)
        return x
    
def train(args):
    # Initialize device
    logger.info("Training date time: {}".format(datetime.datetime.now()))
    logger.info("=========================")
    logger.info("Hyper arguments:")
    for arg_name, arg_value in vars(args).items():
        logger.info("{}: {}".format(arg_name, arg_value))
    logger.info("=========================")
    start_time = time.time()
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.device))
        torch.backends.cudnn.benchmark = True
        logger.info("Using device: CUDA_{}".format(args.device))
    else:
        logger.warn("Using device: CPU")

    cudnn.benchmark = True
    # Data loading and splitting
    logger.info("Start loading data.")
    dataset1 = Counting_dataset(os.path.join(Constants.DATA_FOLDER, Constants.DATASET[args.dataset1]),
                                       input_type=args.dataset_type,
                                       crop_size=args.patch_size,
                                       training_factor=args.training_scale,
                                       resize=args.image_resize)
    dataset2 = Counting_dataset(os.path.join(Constants.DATA_FOLDER, Constants.DATASET[args.dataset2]),
                                       input_type=args.dataset_type,
                                       crop_size=args.patch_size,
                                       training_factor=args.training_scale,
                                       resize=args.image_resize)
    dataset3 = Counting_dataset(os.path.join(Constants.DATA_FOLDER, Constants.DATASET[args.dataset3]),
                                       input_type=args.dataset_type,
                                       crop_size=args.patch_size,
                                       training_factor=args.training_scale,
                                       resize=args.image_resize)
    dataset4 = Counting_dataset(os.path.join(Constants.DATA_FOLDER, Constants.DATASET[args.dataset4]),
                                       input_type=args.dataset_type,
                                       crop_size=args.patch_size,
                                       training_factor=args.training_scale,
                                       resize=args.image_resize)
    source_set = dataset1
    train_source_dataset, val_set = random_split(source_set, [0.7, 0.3])
    train_target_dataset, test_set = random_split(dataset4, [0.7, 0.3])
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # ForeverDataIterator: make training epoch number not controlled by batchsize*n
    # train_source_iter = ForeverDataIterator(train_source_loader)
    # train_target_iter = ForeverDataIterator(train_target_loader)
    logger.info("Loading data completed. Elapsed time: {:.2f}sec.".format(time.time() - start_time))

    logger.info("Start initailizing model")
    model = ResUNet().to(device)
    domain_discri = ConvDomainDiscriminator().to(device)

    # define optimizer and lr scheduler
    optimizer = AdamW(list(model.parameters()) + list(domain_discri.parameters()),
                      lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.learning_rate * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    da_loss = DomainAdversarialLoss(domain_discri).to(device)
    criterion = torch.nn.MSELoss(reduction='mean')
    count_mse = torch.nn.MSELoss(reduction='mean') # Distinguish voxel-wise loss and count loss
    count_mae = torch.nn.L1Loss(reduction='mean')
    
    logger.info("Initialization Completed. Elapsed time: {:.2f}sec".format(time.time() - start_time))
    
    # start training
    best_loss1 = 1000.
    train_time = time.time()
    for epoch in range(args.epoch):
        epoch_time = time.time()
        valid_loss = AverageMeter()
        valid_mse = AverageMeter()
        valid_mae = AverageMeter()
        test_loss = AverageMeter()
        test_mse = AverageMeter()
        test_mae = AverageMeter()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        domain_accs = AverageMeter()
        print("lr:", lr_scheduler.get_last_lr()[0])

        model.train()
        da_loss.train()

        end = time.time()
        optimizer.zero_grad()
        for index, ((img_s, dot_s), (img_t, _)) in enumerate(zip(train_source_loader, train_target_loader)):
            torch.cuda.empty_cache() 
            x_s = img_s.to(device)
            labels_s = dot_s.to(device)
            x_t = img_t.to(device)

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            x = torch.cat((x_s, x_t), dim=0)
            # f, _, _, _, _ = backbone(x)
            # y = model(x)
            y, f = model(x)
            y_s, y_t = y.chunk(2, dim=0)
            f_s, f_t = f.chunk(2, dim=0)

            cls_loss = F.mse_loss(y_s, labels_s)
            transfer_loss = da_loss(f_s, f_t)
            domain_acc = da_loss.domain_discriminator_accuracy
            loss = cls_loss + transfer_loss * args.trade_off

            losses.update(loss.item(), x_s.size(0))
            domain_accs.update(domain_acc.item(), x_s.size(0))

            loss.backward()
            # compute gradient and do SGD step
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                optimizer.step()
                lr_scheduler.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if index % 10 == 0:
                writer.add_scalar(tag='Running loss of {}'.format(Constants.LOG_NAME),
                                  scalar_value=loss.item(), global_step=epoch)

        # Validation
        model.eval()
        for _, (img, dot) in enumerate(val_loader):
            ground_truth = dot.sum([1, 2, 3])
            img = img.to(device)
            dot = dot.to(device)
            dot = dot * args.training_scale
            adjust_factor = 10 ** (math.log10(10) - math.log10(torch.max(dot)))
            dot =  dot * adjust_factor
            outputs, _ = model(img)
            softmax_tensor = torch.nn.functional.softmax(outputs, dim=-1)

            # min_val = torch.min(softmax_tensor)
            # max_val = torch.max(softmax_tensor)
            # outputs = (softmax_tensor - min_val) / (max_val - min_val)
            outputs = softmax_tensor
            adjust_factor = 10 ** (math.log10(10) - math.log10(torch.max(outputs)))
            outputs =  outputs * adjust_factor

            loss = criterion(outputs, dot)
            counts = outputs.sum([1, 2, 3]).detach().cpu() / args.training_scale

            valid_loss.update(loss.item())
            valid_mse.update(count_mse(counts, ground_truth))
            valid_mae.update(count_mae(counts, ground_truth))

        # Tensorboard writer and logging out per epoch
        logger.info("Epoch:{} Cost: {:.1f} sec\n\
                    \nValid\
                    Loss: {:.2f}, \
                    MSE: {:.2f}, \
                    MAE: {:.2f}, \
                    ".format(epoch + 1,time.time() - epoch_time,
                             valid_loss.get('avg'),
                             valid_mse.get('avg'),
                             valid_mae.get('avg')
                             ))
        # evaluate on validation set
        loss1 = valid_loss.get('avg')
        # remember best acc@1 and save checkpoint
        torch.save(model.state_dict(), '/home/xiangli/ResUNet/dann/val_latest.pth')
        if loss1 < best_loss1:
            shutil.copy('/home/xiangli/ResUNet/dann/val_latest.pth', '/home/xiangli/ResUNet/dann/val_best.pth')
        best_loss1 = min(loss1, best_loss1)
        torch.cuda.empty_cache()
    print("best_loss1 = {:3.1f}".format(best_loss1))
    logger.info("Training completed ({:.2f} sec), starting testing...".format(time.time() - train_time))
    # Test
    model.load_state_dict(torch.load('/home/xiangli/ResUNet/dann/val_latest.pth'))
    model.eval()
    for _, (img, dot) in enumerate(test_loader):
        ground_truth = dot.sum([1, 2, 3])
        img = img.to(device)
        dot = dot.to(device)
        dot = dot * args.training_scale
        adjust_factor = 10 ** (math.log10(10) - math.log10(torch.max(dot)))
        dot =  dot * adjust_factor
        outputs, _ = model(img)
        softmax_tensor = torch.nn.functional.softmax(outputs, dim=-1)
        min_val = torch.min(softmax_tensor)
        max_val = torch.max(softmax_tensor)
        outputs = (softmax_tensor - min_val) / (max_val - min_val)
        adjust_factor = 10 ** (math.log10(10) - math.log10(torch.max(outputs)))
        outputs =  outputs * adjust_factor

        loss = criterion(outputs, dot)
        counts = outputs.sum([1, 2, 3]).detach().cpu() / args.training_scale

        test_loss.update(loss.item())
        test_mse.update(count_mse(counts, ground_truth))
        test_mae.update(count_mae(counts, ground_truth))
    logger.info("Test   Cost: {:.1f} sec\
                \nLoss: {:.2f}, \
                MSE: {:.2f}, \
                MAE: {:.2f}, \
                ".format(time.time() - start_time,
                         test_loss.get('avg'),
                         test_mse.get('avg'),
                         test_mae.get('avg'),
                                            ))

    # Cleaning up after the training
    writer.close()
    model_stats = summary(model, (args.batch_size, img.shape[1], img.shape[2], img.shape[3]), verbose=0)
    logger.info("Model Summary:\n{}".format(str(model_stats)))
    logger.info("Finished.")

if __name__ == '__main__':
    parser = ArgParser()
    parser.load_arguments()
    args = parser.parse_args()
    train(args)
