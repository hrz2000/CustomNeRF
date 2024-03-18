import os
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from nerf.data_utils import *
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from einops import rearrange
import imageio
import numpy as np
import glob

# LLFFDataset
from kornia import create_meshgrid
def get_ray_directions(H,W,focal):
    grid = create_meshgrid(H,W,normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    directions = torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) 
    return directions # (H, W, 3)


def get_rays(directions, c2w):
    rays_d = directions @ c2w[:, :3].T
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:, 3].expand(rays_d.shape) # H, W, 3
    rays_d = rays_d.view(-1, 3) # H*W, 3
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d

def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[...,0] / rays_o[...,2]
    oy_oz = rays_o[...,1] / rays_o[...,2]
    
    # Projection
    o0 = -1./(W/(2.*focal)) * ox_oz
    o1 = -1./(H/(2.*focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - ox_oz)
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - oy_oz)
    d2 = 1 - o2
    
    rays_o = torch.stack([o0, o1, o2], -1) # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1) # (B, 3)
    
    return rays_o, rays_d

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, np.linalg.inv(pose_avg_homo)


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


def read_depth_image(path, img_wh):
    """Read image and output RGB image (0-1).

    Args:
        path (str): path to file

    Returns:
        array: RGB image (0-1)
    """
    # print(path)
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')
    return img # H W C

# read color images or depth images
def read_image(img_path, img_wh, blend_a=True):
    img = imageio.imread(img_path).astype(np.float32)/255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])

    if img.shape[2] == 4: # blend A to RGB
        if blend_a:
            img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
        else:
            img = img[..., :3]*img[..., -1:]

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    return img

class LLFFDataset(Dataset):
    def __init__(self, device, root_dir, split='train', resolution_level=1, spheric_poses=False, val_num=1, opt=None):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.opt = opt
        self.root_dir = root_dir
        self.split = split
        print(os.path.join(root_dir, 'images', '*'))
        w,h = Image.open(glob.glob(os.path.join(root_dir, 'images', '*'))[0]).size
        w,h = w//resolution_level, h//resolution_level
        self.img_wh = (int(w),int(h))
        self.spheric_poses = spheric_poses
        self.val_num = 4
        self.define_transforms()
        self.device = device

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy')) # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*[0-9].[Jjp]*')))
        if self.opt.keyword is not None:
            self.mask_paths = [t.replace('JPG','png').replace('jpg','png') for t in self.image_paths] #.replace('.png',f'_{self.opt.keyword}_mask.png')
        else:
            self.mask_paths = [t.replace('JPG','png').replace('.png','_mask.png') for t in self.image_paths]

        self.mask_paths = [t.replace('images',self.opt.keyword) for t in self.mask_paths]
        
        if self.split in ['train', 'val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images
        # assert H*self.img_wh[0] == W*self.img_wh[1], \
        #     f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'
        
        self.focal *= self.img_wh[0]/W

        self.scale = self.img_wh[0]/W

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center) # choose val image as the closest to
                                                   # center image

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal) # (H, W, 3)
            
        self.n_frames = len(self.poses)

        if self.split == 'test':
            if self.opt.inter_pose:
                self.n_images = 4
                idxs = torch.linspace(0, len(self.poses)-1, self.n_images).to(torch.long)
                self.poses = self.poses[idxs]

                poses_new = []
                for i in range(3):
                    c2w1 = np.eye(4)
                    c2w1[:3,:4] = self.poses[i]
                    c2w2 = np.eye(4)
                    c2w2[:3,:4] = self.poses[i+1]
                    c2w1 = torch.tensor(c2w1)
                    c2w2 = torch.tensor(c2w2)
                    tmp = inter_pose_num(c2w1, c2w2, 25)[...,:3,:4].numpy()#.to(device)
                    if i==0:
                        poses_new.extend(tmp[0:])
                    else:
                        poses_new.extend(tmp[1:])
                self.n_images = len(poses_new)
                self.poses_test = poses_new[::-1]
            else:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                    # given in the original repo. Mathematically if near=1
                                    # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)

            self.n_frames = len(self.poses_test)

        self.rays_o = []
        self.rays_d = []
        self.c2w = []
        poses = self.poses_test if self.split == 'test' else self.poses
        for idx in tqdm(range(len(poses))):
            c2w = torch.FloatTensor(poses[idx])
            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.opt.is360Scene:
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0], self.focal, 1.0, rays_o, rays_d)
            self.rays_o.append(rays_o)
            self.rays_d.append(rays_d)
            self.c2w.append(c2w)
        self.rays_o = torch.stack(self.rays_o).to(self.device)
        self.rays_d = torch.stack(self.rays_d).to(self.device)
        self.c2w = torch.stack(self.c2w).to(self.device)

        self.imgs = []
        self.masks = []
        for idx, img_path in enumerate(self.image_paths):
            # img_path = img_path.replace('.jpg','.png').replace('.png',f'_{self.opt.unique}.png')
            img = Image.open(img_path).convert('RGB').resize(self.img_wh)
            try:
                mask = Image.open(self.mask_paths[idx]).convert('L').resize(self.img_wh)
                mask = self.transform(mask).permute(1,2,0)[...,-1]

            except:
                mask = Image.open(self.mask_paths[0]).convert('L').resize(self.img_wh)
                mask = self.transform(mask).permute(1,2,0)[...,-1]
                mask[:] = 0

            img = self.transform(img).permute(1,2,0) # (3, h, w)
            self.imgs.append(img)
            self.masks.append(mask)
        self.imgs = torch.stack(self.imgs).to(self.device)
        self.masks = torch.stack(self.masks).to(self.device)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'test':
            return self.n_frames
        if self.split == 'train':
            return 100
        else:
            return 6

    def __getitem__(self, idx):
        # idx = idx % self.n_frames
        if self.split == 'train':
            idx = np.random.randint(0, self.n_frames)

        rays_o = self.rays_o[idx]
        rays_d = self.rays_d[idx]
        c2w = self.c2w[idx]

        if self.split != 'test':
            rgbs = self.imgs[idx]
            mask = self.masks[idx]
        else:
            rgbs = self.imgs[0]
            mask = self.masks[0]

        W = self.img_wh[0]
        H = self.img_wh[1]
        return rgbs, mask, rays_o, rays_d, H, W, idx

    def get_test_(self,):
        if not self.spheric_poses:
            focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                # given in the original repo. Mathematically if near=1
                                # and far=infinity, then this number will converge to 4
            radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
            self.poses_test = create_spiral_poses(radii, focus_depth)
        else:
            radius = 1.1 * self.bounds.min()
            self.poses_test = create_spheric_poses(radius)  
    
    
    
    
    
    
    
    







