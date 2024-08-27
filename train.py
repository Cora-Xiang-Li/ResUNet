'''
    This training is designed for dot map processing
'''
import datetime
import os
import time

import torch
from config import ArgParser, Constants
from model import ResUNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from utils import AverageMeter, Model_Logger
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data import Counting_dataset

from ray import tune
from ray.tune.schedulers import ASHAScheduler

logger = Model_Logger('train')
logger.enable_exception_hook()
writer = SummaryWriter()

def gaussian_blur(dot):
    # Apply Gaussian Blur to each image in the batch
    dot_np = dot.detach().numpy()
    dot_blurred = np.zeros_like(dot_np)
    for i in range(dot_np.shape[0]):  # Loop over batch size
        for j in range(dot_np.shape[1]):  # Loop over channels
            dot_blurred[i, j] = cv2.GaussianBlur(dot_np[i, j], (15, 15), 4, 4)
            # Convert back to a PyTorch tensor
    dot_blurred_tensor = torch.from_numpy(dot_blurred)
    return dot_blurred_tensor
                        
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
        # device = torch.device('cpu')

    # Data loading and splitting
    logger.info("Start loading data.")
    dataset = Counting_dataset(os.path.join(Constants.DATA_FOLDER,
                                        Constants.DATASET[args.source_dataset]),
                           input_type=args.dataset_type,
                           crop_size=args.patch_size,
                           training_factor=args.training_scale,
                           resize=args.image_resize
                           )
    train_set, valid_set, test_set = random_split(dataset, [0.2, 0.7, 0.1])
    # train_set.dataset.train()
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    logger.info("Loading data completed. Elapsed time: {:.2f}sec.".format(time.time() - start_time))

    logger.info("Start initailizing model")
    model = ResUNet(momentum=args.BN_momentum)
    
    model = model.to(device)

    # Initialize the optimizer with weight decay and learning rate
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.restart_step, T_mult=2)
    # scheduler = StepLR(optimizer, step_size=50, gamma=args.lr_decay)

    criterion = torch.nn.MSELoss(reduction='mean')
    count_mse = torch.nn.MSELoss(reduction='mean') # Distinguish voxel-wise loss and count loss
    count_mae = torch.nn.L1Loss(reduction='mean')

    logger.info("Initialization Completed. Elapsed time: {:.2f}sec".format(time.time() - start_time))

    #TODO Checkpoint loading

    train_time = time.time()
    for epoch in range(args.epoch):
        epoch_time = time.time()
        epoch_loss = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_mae = AverageMeter()
        valid_loss = AverageMeter()
        valid_mse = AverageMeter()
        valid_mae = AverageMeter()
        test_loss = AverageMeter()
        test_mse = AverageMeter()
        test_mae = AverageMeter()

        model.train()
        iters = len(train_dataloader)
        for index, (img, dot) in enumerate(train_dataloader):
            ground_truth = dot.sum([1, 2, 3])
            # dot = dot * args.training_scale
            dot = gaussian_blur(dot)
            img = img.to(device)
            dot = dot.to(device)
            model.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(img)
                if torch.sum(dot) == 0:
                    logger.info("Step {} of epoch {} has the zero cells annotation.".format(index, epoch))
                    continue
                loss = criterion(outputs, dot)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            counts = outputs.sum([1, 2, 3]).detach().cpu() / args.training_scale

            epoch_loss.update(loss.item())
            epoch_mse.update(count_mse(counts, ground_truth))
            epoch_mae.update(count_mae(counts, ground_truth))

            if epoch % 2 == 0:
                plt.imshow(outputs[0][0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                plt.imshow(dot[0][0].detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
                plt.title(f"Epoch {epoch}")
                # Save the plot to a file
                plt.savefig('grayscale_image.png', bbox_inches='tight')
                plt.close()  # Close the figure to free up memory
            # plt.imshow(dot[0][0][0].cpu(), dot[0][0][1].cpu())
            # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

            if index % 100 == 0:
                writer.add_scalar(tag='Running loss of {}'.format(Constants.LOG_NAME),
                                  scalar_value=loss.item(), global_step=epoch)
            # Step the scheduler
            scheduler.step(epoch + index / iters)

        # Validation
        model.eval()
        for _, (img, dot) in enumerate(valid_dataloader):
            ground_truth = dot.sum([1, 2, 3])
            out_indices = torch.nonzero(dot > 0, as_tuple=False)
            in_dot = out_indices[:, [3, 2]]
            dot = gaussian_blur(dot)
            img = img.to(device)
            dot = dot.to(device)

            outputs = model(img)
            loss = criterion(outputs, dot)

            counts = outputs.sum([1, 2, 3]).detach().cpu() / args.training_scale

            valid_loss.update(loss.item())
            valid_mse.update(count_mse(counts, ground_truth))
            valid_mae.update(count_mae(counts, ground_truth))

        # Tensorboard writer and logging out per epoch
        writer.add_scalars(main_tag='Epoch loss of {}'.format(Constants.LOG_NAME), tag_scalar_dict={
            'Train pixel loss': epoch_loss.get('avg'),
            'Train MAE': epoch_mae.get('avg'),
            'Valid pixel loss': valid_loss.get('avg'),
            'Valid MAE': valid_mae.get('avg')
        }, global_step=epoch)
        logger.info("Epoch:{} Cost: {:.1f} sec\nTrain\
                    Loss: {:.2f}, \
                    MSE: {:.2f}, \
                    MAE: {:.2f}, \
                    \nValid\
                    Loss: {:.2f}, \
                    MSE: {:.2f}, \
                    MAE: {:.2f}, \
                    ".format(epoch + 1,time.time() - epoch_time,
                             epoch_loss.get('avg'),
                             epoch_mse.get('avg'),
                             epoch_mae.get('avg'),
                             valid_loss.get('avg'),
                             valid_mse.get('avg'),
                             valid_mae.get('avg')
                             ))

    logger.info("Training completed ({:.2f} sec), starting testing...".format(time.time() - train_time))
    model.eval()
    for _, (img, dot) in enumerate(test_dataloader):
            ground_truth = dot.sum([1, 2, 3])
            out_indices = torch.nonzero(dot > 0, as_tuple=False)
            in_dot = out_indices[:, [3, 2]]
            dot = gaussian_blur(dot)
            img = img.to(device)
            dot = dot.to(device)

            outputs = model(img)
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
    
    ''' Collect data for test
    
    # Define the data rows using variables
    test_data = [
        ("...", args.learning_rate, args.weight_decay, args.BN_momentum, args.batch_size, args.epoch, args.training_scale, test_loss.get('avg'), test_mse.get('avg'), test_mae.get('avg')),
    ]
    row_str = ''
    for row in test_data:
        # Format the row using string formatting
        row_str = "| {}  |    {}   |   {}    |    {}    |    {}    |  {} |      {}     |{:.2f}| {:.2f}|{:.2f}|".format(*row)
        
    with open('test.txt', 'a') as file:
        file.write("\n" + row_str)
    '''

if __name__ == '__main__':
    parser = ArgParser()
    parser.load_arguments()
    args = parser.parse_args()
    train(args)
