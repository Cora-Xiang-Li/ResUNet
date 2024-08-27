import argparse
import datetime
import os



class ArgParser(argparse.ArgumentParser):

    def load_arguments(self):
        self.add_argument('-T', '--tag', type=str, default='Normal exp P/N with Unc',)
        self.add_argument('-L', '--learning-rate', type=float, default=1e-2)
        self.add_argument('-E', '--epoch', type=int, default=20)
        self.add_argument('-N', '--device', type=str, default='0')
        self.add_argument('-LD','--lr-decay', type=float, default=0.75)
        self.add_argument('-TO', '--trade-off', type=int, default=1)
        # self.add_argument('-M', '--momentum', type=float, default=0.9,
        #                   help='Momentum value for SGD optimizer.')
        self.add_argument('-WD', '--weight-decay', type=float, default=1e-4,
                          help="Decay weight for optimizer.")
        self.add_argument('-RT', '--restart-step', type=int, default=10,
                          help='Restart step for cosine annealing warm restarts.')
        self.add_argument('-BM', '--BN-momentum', type=float, default=0.9,
                          help="Momentum for batch normalize.")
        self.add_argument('--lr-gamma', default=0.1, type=float, help='parameter for lr scheduler')
        self.add_argument('-DP', '--dropout', type=float, default=0.1)
        self.add_argument('-O', '--output', type=str, default='outputs')
        self.add_argument('-DS', '--source-dataset', type=str, default='adi')   # current use
        self.add_argument('-DT', '--dataset', type=str, default='mbm')  # unlabeled dataset
        self.add_argument('-D1', '--dataset1', type=str, default='adi')
        self.add_argument('-D2', '--dataset2', type=str, default='dcc')
        self.add_argument('-D3', '--dataset3', type=str, default='mbm')
        self.add_argument('-D4', '--dataset4', type=str, default='vgg')
        self.add_argument('-TS', '--source-dataset-type', type=str, default='h5py')
        self.add_argument('-TT', '--dataset-type', type=str, default='h5py')
        self.add_argument('-TR', '--training-ratio', type=float, default=0.2,
                           help="Training data ratio, test ratio set as 0.1")   # train 0.2 | val 0.1 | unlabeled 0.7
        self.add_argument('-B', '--batch-size', type=int, default=4)
        self.add_argument('-TSS', '--training-scale-s', type=int, default=500,
                          help="mbm: 1000, dcc: 500, bcd: 100,\
                          adi: 100, vgg: 100, 'mbc: 1000")
        self.add_argument('-TST', '--training-scale', type=int, default=1000,
                          help="mbm: 100, dcc: 100, bcd: 100,\
                          adi: 100, vgg: 100, 'mbc: 100")
        self.add_argument('-RS', '--image-resize', type=int, default=512)
        self.add_argument('-P', '--patch-size', type=tuple, default=128,
                          help="Cropping size of image.")
        self.add_argument('-WS', '--warm-start', type=int, default=0,
                          help="Epochs only train regressor on source domain.")
        self.add_argument('-MS', '--memory-saving', type=bool, default=True)

class Constants:
    ROOT_PATH = '/home/xiangli/ResUNet' # Baseline directory path
    
    MODEL_NAME = 'UDA' # Model directory path

    DATA_FOLDER = os.path.join(ROOT_PATH, 'data')
    OUTPUT_FOLDER = os.path.join(ROOT_PATH, 'output')
    LOG_FOLDER = os.path.join(ROOT_PATH, 'log')
    LOG_NAME = datetime.datetime.now()
    TARGET_TRAIN_FILELIST = ''
    TARGET_VALID_FILELIST = ''

    # MBC: 3D
    DATASET = {'vgg': 'VGG', 'mbm':'MBM', 'adi':'ADI', 'bcd':'BCD',
               'mnist_m':'MNIST_moving', 'dcc':'DCC', 'gcc':'GCC', 'ucf':'UCF'}

    CFG = [[32, 'R', 'M', 64, 'R', 'M', 128, 'R', 'M', 512, 'R'], [128, 'R', 'U', 64, 'R', 'U', '32', 'R', 'U', 1, 'R']]


