import datetime
import os
import time

import torch
from config import ArgParser, Constants
from model import ResUNet
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from utils import AverageMeter, Model_Logger
import h5py
import numpy as np
from torch.cuda.amp import GradScaler
from data import Counting_dataset, Read_pseudolabel
import math
import matplotlib.pyplot as plt

logger = Model_Logger('train')
logger.enable_exception_hook()
writer = SummaryWriter()
    
def generate_pseudo_labels(model, unlabeled_dataloader, device, threshold=0.5):
    model.eval()
    pseudo_labels = "pseudo_labels"
    save_path = os.path.join(Constants.DATA_FOLDER, pseudo_labels,'pseudo_labels.hdf5')

    img_dataset = []
    dot_dataset = []

    for index, (img, dot) in enumerate(unlabeled_dataloader):
        img = img.to(device)
        outputs = model(img)
        
        for i in img:
            img_dataset.append(i.cpu().numpy())
        for o in outputs:
            dot_dataset.append(o.cpu().detach().numpy())

    with h5py.File(save_path, 'w') as f:
        f.create_dataset('imgs', data=np.array(np.squeeze(img_dataset, axis=1)), dtype='int8', compression='gzip', compression_opts=9)
        f.create_dataset('counts', data=np.array(np.squeeze(dot_dataset, axis=1)), compression='gzip', compression_opts=9)
        print(f"Saved pseudo labels and images to {save_path}")

    del img_dataset
    del dot_dataset
    torch.cuda.empty_cache()  # Clear cache

    return pseudo_labels


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
        device = torch.device('cpu')

    # Data loading and splitting
    logger.info("Start loading data.")
    # labeled_dataset = Counting_dataset(os.path.join(Constants.DATA_FOLDER, Constants.DATASET[args.source_dataset]),
    #                                    input_type=args.dataset_type,
    #                                    crop_size=args.patch_size,
    #                                    training_factor=args.training_scale,
    #                                    resize=args.image_resize)

    # unlabeled_dataset = Counting_dataset(os.path.join(Constants.DATA_FOLDER, Constants.DATASET[args.dataset]),
    #                                      input_type=args.source_dataset_type,
    #                                      crop_size=args.patch_size,
    #                                      training_factor=args.training_scale,
    #                                      resize=args.image_resize)
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
    # dataset = ConcatDataset([dataset2, dataset4])
    # dataset = dataset2

    # valid_ratio = 1 - args.training_ratio - unlabeled_ratio
    # train_set, valid_set, test_set = random_split(labeled_dataset, [args.training_ratio, valid_ratio, 0.1])
    train_set = dataset3
    valid_set, unlabeled_set, test_set = random_split(dataset4, [0.2, 0.7, 0.1])
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    unlabeled_dataloader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    logger.info("Loading data completed. Elapsed time: {:.2f}sec.".format(time.time() - start_time))

    logger.info("Start initializing model")
    model = ResUNet(momentum=args.BN_momentum)
    model = model.to(device)

    # Initialize the optimizer with weight decay and learning rate
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.restart_step, T_mult=2)

    # criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
    criterion = torch.nn.MSELoss(reduction='mean')
    count_mse = torch.nn.MSELoss(reduction='mean')
    count_mae = torch.nn.L1Loss(reduction='mean')

    logger.info("Initialization Completed. Elapsed time: {:.2f}sec".format(time.time() - start_time))

    self_training_iter = 5
    for iteration in range(self_training_iter):
        logger.info(f"Self-training iteration {iteration + 1}/{self_training_iter}")
        
        if iteration > 0:
            # Generate pseudo-labels
            # pseudo_labels = generate_pseudo_labels(model, unlabeled_dataloader, device)
            model.eval()
            pseudo_labels = "pseudo_labels"
            save_path = os.path.join(Constants.DATA_FOLDER, pseudo_labels,'pseudo_labels.hdf5')
            # Delete the old file if it exists
            if os.path.exists(save_path):
                os.remove(save_path)
            img_dataset = []
            dot_dataset = []

            for _, (img, _) in enumerate(unlabeled_dataloader):
                img = img.to(device)
                outputs = model(img)
                softmax_tensor = torch.nn.functional.softmax(outputs, dim=-1)

                min_val = torch.min(softmax_tensor)
                max_val = torch.max(softmax_tensor)
                outputs = (softmax_tensor - min_val) / (max_val - min_val)
                # outputs = torch.clamp(outputs, min=1e-7, max=1.0 - 1e-7)
                adjust_factor = 10 ** (math.log10(100) - math.log10(torch.max(outputs)))
                outputs =  outputs * adjust_factor
                for i in img:
                    img_dataset.append(i.cpu().numpy())
                for o in outputs:
                    dot_dataset.append(o.cpu().detach().numpy())

                density_map3 = outputs[0][0].cpu().detach().numpy()  # Ensure it's a numpy array
                    # Plot and save the second density map
                plt.imshow(density_map3, cmap='hot', interpolation='nearest')
                plt.title('3 Density Map')
                plt.colorbar()
                plt.savefig('density_map3.png')  # Save the image
                plt.close()  # Close the figure to free up memory

            with h5py.File(save_path, 'w') as f:
                f.create_dataset('imgs', data=np.array(np.squeeze(img_dataset, axis=1)), compression='gzip', compression_opts=9)
                f.create_dataset('counts', data=np.array(np.squeeze(dot_dataset, axis=1)), compression='gzip', compression_opts=9)
                print(f"Saved pseudo labels and images to {save_path}")

            del img_dataset
            del dot_dataset
            # torch.cuda.empty_cache()
            print('start to load concated datasets...')
            pseudo_dataset = Read_pseudolabel(os.path.join(Constants.DATA_FOLDER, pseudo_labels),
                                              input_type=args.dataset_type)
            combined_dataset = ConcatDataset([train_set, pseudo_dataset])
            train_dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, shuffle=True)

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
                img = img.to(device)
                dot = dot.to(device)
                # Scale the dot tensor values
                # min_val = torch.min(dot)
                # max_val = torch.max(dot)
                # dot = (dot - min_val) / (max_val - min_val)
                # dot = torch.clamp(dot, min=0.0, max=1.0 - 1e-7)
                dot = dot * args.training_scale
                adjust_factor = 10 ** (math.log10(100) - math.log10(torch.max(dot)))
                dot =  dot * adjust_factor
                optimizer.zero_grad()  # Reset the gradients
                with torch.set_grad_enabled(True):
                    outputs = model(img)
                    # outputs = torch.clamp(outputs, min=0.0, max=1.0 - 1e-7)
                    # Apply softmax to the tensor across a specified dimension (e.g., across the last dimension)
                    softmax_tensor = torch.nn.functional.softmax(outputs, dim=-1)

                    min_val = torch.min(softmax_tensor)
                    max_val = torch.max(softmax_tensor)
                    outputs = (softmax_tensor - min_val) / (max_val - min_val)

                    # Calculate the threshold based on the 90th percentile
                    # threshold = torch.quantile(outputs.flatten(), 0.1)
                    # outputs = torch.where(outputs < threshold, torch.tensor(0.0), outputs)
                    # outputs[outputs < 0.1] = 0
                    outputs = outputs * args.training_scale
                    adjust_factor = 10 ** (math.log10(100) - math.log10(torch.max(outputs)))
                    outputs =  outputs * adjust_factor

                    # Access the first and second density maps
                    first_density_map = dot[0][0].cpu().numpy()  # Ensure it's a numpy array

                    # Plot and save the first density map
                    plt.imshow(first_density_map, cmap='hot', interpolation='nearest')
                    plt.title('First Density Map')
                    plt.colorbar()
                    plt.savefig('first_density_map.png')  # Save the image
                    plt.close()  # Close the figure to free up memory

                    second_density_map = outputs[0][0].cpu().detach().numpy()  # Ensure it's a numpy array
                    # Plot and save the second density map
                    plt.imshow(second_density_map, cmap='hot', interpolation='nearest')
                    plt.title('Second Density Map')
                    plt.colorbar()
                    plt.savefig('second_density_map.png')  # Save the image
                    plt.close()  # Close the figure to free up memory

                    if torch.sum(dot) == 0:
                        # logger.info("Step {} of epoch {} has invalid cells annotation.".format(index, epoch))
                        continue
                    loss = criterion(outputs, dot)
                    loss.backward()
                    optimizer.step()

                counts = outputs.sum([1, 2, 3]).detach().cpu() / args.training_scale
                adjust_factor = 10 ** (math.log10(ground_truth[0]) - math.log10(counts[0]))
                counts =  counts * adjust_factor

                epoch_loss.update(loss.item())
                epoch_mse.update(count_mse(counts, ground_truth))
                epoch_mae.update(count_mae(counts, ground_truth))

                if index % 100 == 0:
                    writer.add_scalar(tag='Running loss of {}'.format(Constants.LOG_NAME),
                                      scalar_value=loss.item(), global_step=epoch)

                # Step the scheduler
                scheduler.step(epoch + index / iters)

            # Validation
            model.eval()
            for _, (img, dot) in enumerate(valid_dataloader):
                ground_truth = dot.sum([1, 2, 3])
                # Clamp Values to Avoid Log of Zero
                # dot = torch.clamp(dot, min=1e-7, max=1.0 - 1e-7)
                # outputs = torch.clamp(outputs, min=1e-7, max=1.0 - 1e-7)
                img = img.to(device)
                dot = dot.to(device)
                # dot = torch.clamp(dot, min=1e-7, max=1.0 - 1e-7)
                dot = dot * args.training_scale
                adjust_factor = 10 ** (math.log10(100) - math.log10(torch.max(dot)))
                dot =  dot * adjust_factor
                with torch.no_grad():
                    outputs = model(img)
                    softmax_tensor = torch.nn.functional.softmax(outputs, dim=-1)

                    min_val = torch.min(softmax_tensor)
                    max_val = torch.max(softmax_tensor)
                    outputs = (softmax_tensor - min_val) / (max_val - min_val)

                    # Calculate the threshold based on the 90th percentile
                    # threshold = torch.quantile(outputs.flatten(), 0.1)
                    # outputs = torch.where(outputs < threshold, torch.tensor(0.0), outputs)
                    # outputs[outputs < 0.1] = 0
                    outputs = outputs * args.training_scale
                    adjust_factor = 10 ** (math.log10(100) - math.log10(torch.max(outputs)))
                    outputs =  outputs * adjust_factor
                    loss = criterion(outputs, dot)

                counts = outputs.sum([1, 2, 3]).detach().cpu() / args.training_scale
                adjust_factor = 10 ** (math.log10(ground_truth[0]) - math.log10(counts[0]))
                counts =  counts * adjust_factor

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
                        ".format(epoch + 1, time.time() - epoch_time,
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
        img = img.to(device)
        dot = dot.to(device)
        dot = dot * args.training_scale
        adjust_factor = 10 ** (math.log10(100) - math.log10(torch.max(dot)))
        dot =  dot * adjust_factor

        outputs = model(img)

        loss = criterion(outputs, dot)
        counts = outputs.sum([1, 2, 3]).detach().cpu() / args.training_scale
        adjust_factor = 10 ** (math.log10(ground_truth[0]) - math.log10(counts[0]))
        counts =  counts * adjust_factor
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
    torch.cuda.empty_cache()
    parser = ArgParser()
    parser.load_arguments()
    args = parser.parse_args()
    train(args)
