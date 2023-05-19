# ENPM673 Final Project

### Data loader and Preprocessing
"""
This project uses KITTI Visual Odometry Dataset, which can be found at http://www.cvlibs.net/datasets/kitti/eval_odometry.php
"""

# !pip install einops
# !pip install timm
# !pip install wandb

"""### Download Custom Depth dataset, NYUv2, inria
Run below 2 cells only for the first time in your terminal.
"""

# !gdown 1wbDWkn4uyYUe3hGPYoXWQN_MMvGCda-R

# !unzip "/content/depth_dataset.zip"


# Importing libraries
import glob
import os, errno
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import timm
import wandb
from numpy.core.numeric import Inf
import glob as glob_module
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from torchvision import transforms, utils
import random
from glob import glob
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import json
from torch.utils.data import ConcatDataset
from sklearn import linear_model
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter

class KITTI:
    def __init__(self,
                 data_path=r"/dataset/sequences",
                 pose_path=r"dataset/poses",
                 cam_id="0",
                 sequence="00",

                 ):
        """
        Dataloader for KITTI Visual Odometry Dataset
            http://www.cvlibs.net/datasets/kitti/eval_odometry.php

        Arguments:
            data_path {str}: path to data sequences
            pose_path {str}: path to poses
            sequence {str}: sequence to be tested (default: "00")
        """
        self.data_path = data_path
        self.sequence = sequence
        self.cam_id = cam_id
        self.frame_id = 0

        # Read ground truth poses
        with open(os.path.join(pose_path, sequence+".txt")) as f:
            self.poses = f.readlines()

        # Get frames list
        frames_dir = os.path.join(data_path, sequence, "image_{}".format(cam_id), "*.png")
        print(frames_dir)
        self.frames = sorted(glob_module.glob(frames_dir))

        # Camera Parameters
        self.cam_params = {}
        frame = cv2.imread(self.frames[self.frame_id], 0)
        self.cam_params["width"] = frame.shape[0]
        self.cam_params["height"] = frame.shape[1]
        self.read_intrinsics_parameters()

    def __len__(self):
        return len(self.frames)

    def get_the_next_data(self):
        """
        Returns:
            frame {ndarray}: image frame at index self.frame_id
            pose {list}: list containing the ground truth pose [x, y, z]
            frame_id {int}: integer representing the frame index
        """
        # Read frame as grayscale
        frame = cv2.imread(self.frames[self.frame_id], 0)
        self.cam_params["width"] = frame.shape[0]
        self.cam_params["height"] = frame.shape[0]

        # Read poses
        pose = self.poses[self.frame_id]
        pose = pose.strip().split()
        pose = [float(pose[3]), float(pose[7]), float(pose[11])]  # coordinates for the left camera
        frame_id = self.frame_id
        self.frame_id = self.frame_id + 1
        return frame, pose, frame_id


    def read_intrinsics_parameters(self):
        """
        Reads camera intrinsics parameters

        Returns:
            cam_params {dict}: dictionary with focal lenght and principal point
        """
        calib_file = os.path.join(self.data_path, self.sequence, "calib.txt")
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            line = lines[int(self.cam_id)].strip().split()
            [fx, cx, fy, cy] = [float(line[1]), float(line[3]), float(line[6]), float(line[7])]

            # focal length of camera
            self.cam_params["fx"] = fx
            self.cam_params["fy"] = fy
            # principal point (optical center)
            self.cam_params["cx"] = cx
            self.cam_params["cy"] = cy

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to('cpu').float())
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

class Depth_Dataset(Dataset):
    """
        Dataset class for the Depth Task. Requires for each image, its depth ground-truth and
        segmentation mask
        Args:
            :- config -: json config file
            :- dataset_name -: str
            :- split -: split ['train', 'val', 'test']
    """
    def __init__(self, config, dataset_name, split=None):
        self.split = split
        self.config = config

        path_images = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_images'])
        path_depths = os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_depths'])

        self.paths_images = get_total_paths(path_images, config['Dataset']['extensions']['ext_images'])
        self.paths_depths = get_total_paths(path_depths, config['Dataset']['extensions']['ext_depths'])
        
        assert (self.split in ['train', 'test', 'val']), "Invalid split!"
        assert (len(self.paths_images) == len(self.paths_depths)), "Different number of instances between the input and the depth maps"
        assert (config['Dataset']['splits']['split_train'] + config['Dataset']['splits']['split_test'] + config['Dataset']['splits']['split_val'] == 1), "Invalid splits (sum must be equal to 1)"

        # utility func for splitting
        self.paths_images, self.paths_depths = get_splitted_dataset(config, self.split, dataset_name, self.paths_images, self.paths_depths)

        # Get the transforms
        self.transform_image, self.transform_depth = GetTransforms(config)

        # get p_flip from config
        self.p_flip = config['Dataset']['transforms']['p_flip'] if split=='train' else 0
        self.p_crop = config['Dataset']['transforms']['p_crop'] if split=='train' else 0
        self.p_rot = config['Dataset']['transforms']['p_rot'] if split=='train' else 0
        self.resize = config['Dataset']['transforms']['resize']

    def __len__(self):
        """
            Function to get the number of images using the given list of images
        """
        return len(self.paths_images)

    def __getitem__(self, idx):
        """
            Getter function in order to get the triplet of images / depth maps and segmentation masks
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.transform_image(Image.open(self.paths_images[idx]))
        depth = self.transform_depth(Image.open(self.paths_depths[idx]))
        imgorig = image.clone()

        if random.random() < self.p_flip:
            image = TF.hflip(image)
            depth = TF.hflip(depth)

        if random.random() < self.p_crop:
            random_size = random.randint(256, self.resize-1)
            max_size = self.resize - random_size
            left = int(random.random()*max_size)
            top = int(random.random()*max_size)
            image = TF.crop(image, top, left, random_size, random_size)
            depth = TF.crop(depth, top, left, random_size, random_size)
            image = transforms.Resize((self.resize, self.resize))(image)
            depth = transforms.Resize((self.resize, self.resize))(depth)

        if random.random() < self.p_rot:
            #rotate
            random_angle = random.random()*20 - 10 #[-10 ; 10]
            mask = torch.ones((1,self.resize,self.resize)) #useful for the resize at the end
            mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            image = TF.rotate(image, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            depth = TF.rotate(depth, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
            #crop to remove black borders due to the rotation
            left = torch.argmax(mask[:,0,:]).item()
            top = torch.argmax(mask[:,:,0]).item()
            coin = min(left,top)
            size = self.resize - 2*coin
            image = TF.crop(image, coin, coin, size, size)
            depth = TF.crop(depth, coin, coin, size, size)
            #Resize
            image = transforms.Resize((self.resize, self.resize))(image)
            depth = transforms.Resize((self.resize, self.resize))(depth)
        # exit(0)
        return image, depth

"""### Camera Model setup"""

class CamModel(object):
    """
    Class that represents a pin-hole camera model (or projective camera model).
    In the pin-hole camera model, light goes through the camera center (cx, cy) before its projection
    onto the image plane.
    """
    def __init__(self, params):
        """
        Creates a camera model

        Arguments:
            params {dict} -- Camera parameters
        """

        # Image resolution
        self.width = params['width']
        self.height = params['height']
        # Focal length of camera
        self.fx = params['fx']
        self.fy = params['fy']
        # Optical center (principal point)
        self.cx = params['cx']
        self.cy = params['cy']
        # Distortion coefficients.
        # k1, k2, and k3 are the radial coefficients.
        # p1 and p2 are the tangential distortion coefficients.
        #self.distortion_coeff = [params['k1'], params['k2'], params['p1'], params['p2'], params['k3']]
        self.mat = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]])

"""### Dense Estimation model using visual transformers

#### Depth Estimation Transformer Model Architecture
"""

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners)
        return x

class HeadDepth(nn.Module):
    def __init__(self, features):
        super(HeadDepth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            # nn.ReLU()
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.head(x)
        # x = (x - x.min())/(x.max()-x.min() + 1e-15)
        return x

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class Fusion(nn.Module):
    def __init__(self, resample_dim):
        super(Fusion, self).__init__()
        self.res_conv1 = ResidualConvUnit(resample_dim)
        self.res_conv2 = ResidualConvUnit(resample_dim)
        #self.resample = nn.ConvTranspose2d(resample_dim, resample_dim, kernel_size=2, stride=2, padding=0, bias=True, dilation=1, groups=1)

    def forward(self, x, previous_stage=None):
        if previous_stage == None:
            previous_stage = torch.zeros_like(x)
        output_stage1 = self.res_conv1(x)
        output_stage1 += previous_stage
        output_stage2 = self.res_conv2(output_stage1)
        output_stage2 = nn.functional.interpolate(output_stage2, scale_factor=2, mode="bilinear", align_corners=True)
        return output_stage2

class Read_ignore(nn.Module):
    def __init__(self, start_index=1):
        super(Read_ignore, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index:]


class Read_add(nn.Module):
    def __init__(self, start_index=1):
        super(Read_add, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class Read_projection(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(Read_projection, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)
        return self.project(features)

class MyConvTranspose2d(nn.Module):
    def __init__(self, conv, output_size):
        super(MyConvTranspose2d, self).__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x

class Resample(nn.Module):
    def __init__(self, p, s, h, emb_dim, resample_dim):
        super(Resample, self).__init__()
        assert (s in [4, 8, 16, 32]), "s must be in [0.5, 4, 8, 16, 32]"
        self.conv1 = nn.Conv2d(emb_dim, resample_dim, kernel_size=1, stride=1, padding=0)
        if s == 4:
            self.conv2 = nn.ConvTranspose2d(resample_dim,
                                resample_dim,
                                kernel_size=4,
                                stride=4,
                                padding=0,
                                bias=True,
                                dilation=1,
                                groups=1)
        elif s == 8:
            self.conv2 = nn.ConvTranspose2d(resample_dim,
                                resample_dim,
                                kernel_size=2,
                                stride=2,
                                padding=0,
                                bias=True,
                                dilation=1,
                                groups=1)
        elif s == 16:
            self.conv2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(resample_dim, resample_dim, kernel_size=2,stride=2, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Reassemble(nn.Module):
    def __init__(self, image_size, read, p, s, emb_dim, resample_dim):
        """
        p = patch size
        s = coefficient resample
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super(Reassemble, self).__init__()
        channels, image_height, image_width = image_size

        #Read
        self.read = Read_ignore()
        if read == 'add':
            self.read = Read_add()
        elif read == 'projection':
            self.read = Read_projection(emb_dim)

        #Concat after read
        self.concat = Rearrange('b (h w) c -> b c h w',
                                c=emb_dim,
                                h=(image_height // p),
                                w=(image_width // p))

        #Projection + Resample
        self.resample = Resample(p, s, image_height, emb_dim, resample_dim)

    def forward(self, x):
        x = self.read(x)
        x = self.concat(x)
        x = self.resample(x)
        return x

torch.manual_seed(0)

class DepthTransformerModel(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 1024,
                 resample_dim       = 256,
                 read               = 'projection',
                 num_layers_encoder = 24,
                 hooks              = [5, 11, 17, 23],
                 reassemble_s       = [4, 8, 16, 32],
                 transformer_dropout= 0,
                 nclasses           = 2,
                 type               = "depth",
                 model_timm         = "vit_large_patch16_384"):
        """
        Depth Transformer Model
        type : {"depth"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)
        self.type_ = type

        #Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        #Reassembles Fusion
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        #Head
        if type == "depth":
            self.head_depth = HeadDepth(resample_dim)

    def forward(self, img):

        t = self.transformer_encoders(img)
        previous_stage = None
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result
        out_depth = None
        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        return out_depth

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))

def get_total_paths(path, ext):
    return glob(os.path.join(path, '*'+ext))

def get_splitted_dataset(config, split, dataset_name, path_images, path_depths):
    list_files = [os.path.basename(im) for im in path_images]
    np.random.seed(config['General']['seed'])
    np.random.shuffle(list_files)
    if split == 'train':
        selected_files = list_files[:int(len(list_files)*config['Dataset']['splits']['split_train'])]
    elif split == 'val':
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train']):int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val'])]
    else:
        selected_files = list_files[int(len(list_files)*config['Dataset']['splits']['split_train'])+int(len(list_files)*config['Dataset']['splits']['split_val']):]

    path_images = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_images'], im[:-4]+config['Dataset']['extensions']['ext_images']) for im in selected_files]
    path_depths = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_depths'], im[:-4]+config['Dataset']['extensions']['ext_depths']) for im in selected_files]
    # path_segmentation = [os.path.join(config['Dataset']['paths']['path_dataset'], dataset_name, config['Dataset']['paths']['path_segmentations'], im[:-4]+config['Dataset']['extensions']['ext_segmentations']) for im in selected_files]
    return path_images, path_depths

def GetTransforms(config):
    im_size = config['Dataset']['transforms']['resize']
    transform_image = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_depth = transforms.Compose([
        transforms.Resize((im_size, im_size)),
        transforms.Grayscale(num_output_channels=1) ,
        transforms.ToTensor()
    ])
    return transform_image, transform_depth

def get_losses(config):
    def NoneFunction(a, b):
        return 0
    loss_depth = NoneFunction
    type = config['General']['type']
    if type=="depth":
        if config['General']['loss_depth'] == 'mse':
            loss_depth = nn.MSELoss()
        elif config['General']['loss_depth'] == 'ssi':
            loss_depth = Scale_And_Shift_Invariant_Loss()
    return loss_depth

def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def get_optimizer(config, net):
    names = set([name.split('.')[0] for name, _ in net.named_modules()]) - set(['', 'transformer_encoders'])
    params_backbone = net.transformer_encoders.parameters()
    params_scratch = list()
    for name in names:
        params_scratch += list(eval("net."+name).parameters())

    if config['General']['optim'] == 'adam':
        optimizer_backbone = optim.Adam(params_backbone, lr=config['General']['lr_backbone'])
        optimizer_scratch = optim.Adam(params_scratch, lr=config['General']['lr_scratch'])
    elif config['General']['optim'] == 'sgd':
        optimizer_backbone = optim.SGD(params_backbone, lr=config['General']['lr_backbone'], momentum=config['General']['momentum'])
        optimizer_scratch = optim.SGD(params_scratch, lr=config['General']['lr_scratch'], momentum=config['General']['momentum'])
    return optimizer_backbone, optimizer_scratch

def get_schedulers(optimizers):
    return [ReduceLROnPlateau(optimizer) for optimizer in optimizers]

def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class Gradient_Losses(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class Scale_And_Shift_Invariant_Loss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = Gradient_Losses(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target):
        #preprocessing
        mask = target > 0

        #calcul
        scale, shift = compute_scale_and_shift(prediction, target, mask)
        # print(scale, shift)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)

class Predictor(object):
    def __init__(self, config, input_images, frame):
        self.input_images = input_images
        self.config = config
        self.type = self.config['General']['type']
        self.frame = frame

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        self.model = DepthTransformerModel(
                    image_size  =   (3,resize,resize),
                    emb_dim     =   config['General']['emb_dim'],
                    resample_dim=   config['General']['resample_dim'],
                    read        =   config['General']['read'],
                    nclasses    =   len(config['Dataset']['classes']) + 1,
                    hooks       =   config['General']['hooks'],
                    model_timm  =   config['General']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['General']['patch_size'],
        )
        # path_model = os.path.join(config['General']['path_model'], 'DepthTransformerModel_{}.p'.format(config['General']['model_timm']))
        path_model = os.path.join('models/FocusOnDepth_vit_base_patch16_384.p')
        self.model.load_state_dict(
            torch.load(path_model, map_location=self.device)['model_state_dict'], strict=False
        )
        self.model.eval()
        self.transform_image = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.output_dir = self.config['General']['path_predicted_images']
        create_dir(self.output_dir)

    def run(self):
        with torch.no_grad():
            for images in self.input_images:
                pil_im = Image.open(images).convert("RGB")
                original_size = pil_im.size
                tensor_im = self.transform_image(pil_im).unsqueeze(0)
                output_depth = self.model(tensor_im)
                output_depth = 1-output_depth
                output_depth = transforms.ToPILImage()(output_depth.squeeze(0).float()).resize(original_size, resample=Image.BICUBIC)
                path_dir_depths = os.path.join(self.output_dir, 'depths')
                create_dir(path_dir_depths)
                output_depth.save(os.path.join(path_dir_depths, os.path.basename(images)))

    def get_depth(self):
        with torch.no_grad():
            # Convert the NumPy array to a PIL Image
            # Convert the single-channel image to a 3-channel image by duplicating the channels
            three_channel_frame = np.stack((self.frame,) * 3, axis=-1)
            pil_frame = Image.fromarray(three_channel_frame)

            
            # Get the original size of the frame
            original_size = pil_frame.size
            
            # Transform the frame and get the depth and segmentation output
            tensor_im = self.transform_image(pil_frame).unsqueeze(0)
            output_depth = self.model(tensor_im)
            output_depth = 1 - output_depth
            
            # Convert the output depth tensor back to a PIL Image and resize it to the original size
            output_depth = transforms.ToPILImage()(output_depth.squeeze(0).float()).resize(original_size, resample=Image.BICUBIC)
            return output_depth

"""#### Train the Depth Estimation Transformer Model"""

class Trainer(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = self.config['General']['type']

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)
        resize = config['Dataset']['transforms']['resize']
        self.model = DepthTransformerModel(
                    image_size  =   (3,resize,resize),
                    emb_dim     =   config['General']['emb_dim'],
                    resample_dim=   config['General']['resample_dim'],
                    read        =   config['General']['read'],
                    nclasses    =   len(config['Dataset']['classes']) + 1,
                    hooks       =   config['General']['hooks'],
                    model_timm  =   config['General']['model_timm'],
                    type        =   self.type,
                    patch_size  =   config['General']['patch_size'],
        )

        self.model.to(self.device)
        # print(self.model)
        # exit(0)

        self.loss_depth = get_losses(config)
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(config, self.model)
        self.schedulers = get_schedulers([self.optimizer_backbone, self.optimizer_scratch])

    def train(self, train_dataloader, val_dataloader):
        epochs = self.config['General']['epochs']
        val_loss = Inf
        for epoch in range(epochs):  # loop over the dataset multiple times
            print("Epoch ", epoch+1)
            running_loss = 0.0
            self.model.train()
            pbar = tqdm(train_dataloader)
            pbar.set_description("Training")
            for i, (X, Y_depths) in enumerate(pbar):
                # get the inputs; data is a list of [inputs, labels]
                X, Y_depths = X.to(self.device), Y_depths.to(self.device)
                # zero the parameter gradients
                self.optimizer_backbone.zero_grad()
                self.optimizer_scratch.zero_grad()
                # forward + backward + optimizer
                output_depths = self.model(X)
                output_depths = output_depths.squeeze(1) if output_depths != None else None

                Y_depths = Y_depths.squeeze(1) #1xHxW -> HxW
                # Y_segmentations = Y_segmentations.squeeze(1) #1xHxW -> HxW
                # get loss
                loss = self.loss_depth(output_depths, Y_depths)
                loss.backward()
                # step optimizer
                self.optimizer_scratch.step()
                self.optimizer_backbone.step()

                running_loss += loss.item()
                if np.isnan(running_loss):
                    print('\n',
                        X.min().item(), X.max().item(),'\n',
                        Y_depths.min().item(), Y_depths.max().item(),'\n',
                        output_depths.min().item(), output_depths.max().item(),'\n',
                        loss.item(),
                    )
                    exit(0)

                pbar.set_postfix({'training_loss': running_loss/(i+1)})

            new_val_loss = self.Run_evaluation(val_dataloader)

            if new_val_loss < val_loss:
                self.SaveModel()
                val_loss = new_val_loss

            self.schedulers[0].step(new_val_loss)
            self.schedulers[1].step(new_val_loss)

        print('Finished Training')

    def Run_evaluation(self, val_dataloader):
        """
            Evaluate the model on the validation set and visualize some results
            on wandb
            :- val_dataloader -: torch dataloader
        """
        val_loss = 0.
        self.model.eval()
        X_1 = None
        Y_depths_1 = None
        Y_segmentations_1 = None
        output_depths_1 = None
        output_segmentations_1 = None
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")
            for i, (X, Y_depths) in enumerate(pbar):
                X, Y_depths= X.to(self.device), Y_depths.to(self.device)
                output_depths = self.model(X)
                output_depths = output_depths.squeeze(1) if output_depths != None else None
                Y_depths = Y_depths.squeeze(1)
                if i==0:
                    X_1 = X
                    Y_depths_1 = Y_depths
                    output_depths_1 = output_depths
                # get loss
                loss = self.loss_depth(output_depths, Y_depths)
                val_loss += loss.item()
                pbar.set_postfix({'validation_loss': val_loss/(i+1)})

        return val_loss/(i+1)

    def SaveModel(self):
        path_model = os.path.join(self.config['General']['path_model'], self.model.__class__.__name__)
        create_dir(path_model)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_backbone_state_dict': self.optimizer_backbone.state_dict(),
                    'optimizer_scratch_state_dict': self.optimizer_scratch.state_dict()
                    }, path_model+'.p')
        print('Model saved at : {}'.format(path_model))

"""#### Run Train"""

with open('config.json', 'r') as f:
    config = json.load(f)
np.random.seed(config['General']['seed'])

list_data = config['Dataset']['paths']['list_datasets']

## train set
depth_datasets_train = []
for dataset_name in list_data:
    depth_datasets_train.append(Depth_Dataset(config, dataset_name, 'train'))

train_data = ConcatDataset(depth_datasets_train)
train_dataloader = DataLoader(train_data, batch_size=config['General']['batch_size'], shuffle=True)
## validation set
depth_datasets_val = []
for dataset_name in list_data:
    depth_datasets_val.append(Depth_Dataset(config, dataset_name, 'val'))
val_data = ConcatDataset(depth_datasets_val)
val_dataloader = DataLoader(val_data, batch_size=config['General']['batch_size'], shuffle=True)

trainer = Trainer(config)
trainer.train(train_dataloader, val_dataloader)

training_loss = []
validation_loss = []
epochs = range(1, len(training_loss) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_loss, 'r', label='Training loss')
plt.plot(epochs, validation_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

"""#### Run Inferance on the Depth Estimation Model"""

with open('config.json', 'r') as f:
    config = json.load(f)

input_images = glob('input/*.jpg') + glob('input/*.png')
predictor = Predictor(config, input_images, frame=None)
predictor.run()

"""#### Plot Predicted Dense Map"""

def plot_depth(depth_image_path):
    # Check if the file exists
    if not os.path.exists(depth_image_path):
        print(f"File not found: {depth_image_path}")
        return

    # Open the depth image using PIL
    depth_image = Image.open(depth_image_path)

    # Plot the depth image using matplotlib
    plt.imshow(depth_image, cmap='viridis')
    plt.colorbar()
    plt.title('Depth Image')
    plt.show()

# Define the directory path
directory_path = "output/depths"

# Get all .png files in the directory
image_paths = glob(os.path.join(directory_path, "*.png"))

# Loop through all image paths and plot each image
for image_path in image_paths:
    plot_depth(image_path)