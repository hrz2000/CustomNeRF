import os
import math

import trimesh
import numpy as np

import torch
import torch.nn as nn

import plyfile
import skimage
import skimage.measure
import time
import torch.nn.functional as F

import raymarching

from .provider_utils import custom_meshgrid, safe_normalize


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def cdf_Phi_s(x, s):  # \VarPhi_s(t)
    # den = 1 + torch.exp(-s*x)
    # y = 1./den
    # return y
    return torch.sigmoid(x * s)


def sdf_to_alpha(sdf: torch.Tensor, s):
    # [(B), N_rays, N_pts]
    cdf = cdf_Phi_s(sdf, s)
    # [(B), N_rays, N_pts-1]
    # TODO: check sanity.
    opacity_alpha = (cdf[..., :-1] - cdf[..., 1:]) / (cdf[..., :-1] + 1e-10)
    opacity_alpha = torch.clamp_min(opacity_alpha, 0)
    return cdf, opacity_alpha


def sdf_to_w(sdf: torch.Tensor, s):
    device = sdf.device
    # [(B), N_rays, N_pts-1]
    cdf, opacity_alpha = sdf_to_alpha(sdf, s)

    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ],
        dim=-1,
    )

    # [(B), N_rays, N_pts-1]
    visibility_weights = (
            opacity_alpha * torch.cumprod(shifted_transparency, dim=-1)[..., :-1]
    )

    return cdf, opacity_alpha, visibility_weights


def alpha_to_w(alpha: torch.Tensor):
    device = alpha.device
    # [(B), N_rays, N_pts]
    shifted_transparency = torch.cat(
        [
            torch.ones([*alpha.shape[:-1], 1], device=device),
            1.0 - alpha + 1e-10,
        ],
        dim=-1,
    )

    # [(B), N_rays, N_pts-1]
    visibility_weights = alpha * torch.cumprod(shifted_transparency, dim=-1)[..., :-1]

    return visibility_weights


def plot_pointcloud(pc, color=None):
    # pc: [N, 3]
    # color: [N, 3/4]
    print('[visualize points]', pc.shape, pc.dtype, pc.min(0), pc.max(0))
    pc = trimesh.PointCloud(pc, color)
    # axis
    axes = trimesh.creation.axis(axis_length=4)
    # sphere
    # sphere = trimesh.creation.icosphere(radius=1)
    mesh = trimesh.load('./mesh_sdf/doll_sdf_g_9k.ply')

    trimesh.Scene([pc, axes, mesh]).show()


def convert_sigma_samples_to_ply(
        input_3d_sigma_array: np.ndarray,
        voxel_grid_origin,
        volume_size,
        ply_filename_out,
        level=5.0,
        offset=None,
        scale=None, ):
    """
    Convert sdf samples to .ply

    :param input_3d_sdf_array: a float array of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :volume_size: a list of three floats
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = skimage.measure.marching_cubes(
        input_3d_sigma_array, level=level, spacing=volume_size
    )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
    # mesh_points = np.matmul(mesh_points, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % str(ply_filename_out))
    ply_data.write(ply_filename_out)

    print(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )

    return verts, faces, normals


class NeRFRenderer(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.bound = opt.bound

        self.cascade = 1 + math.ceil(math.log2(opt.bound))

        print(self.cascade)

        self.grid_size = 128
        self.cuda_ray = opt.cuda_ray
        self.min_near = opt.min_near
        self.density_thresh = opt.density_thresh


        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        # NOTE: aabb (can be rectangular) is only used to generate points, we still rely on bound (always cubic) to calculate density grid and hashing.

        aabb_train = torch.FloatTensor([-opt.bound, -opt.bound, -opt.bound, opt.bound, opt.bound, opt.bound])
        aabb_infer = aabb_train.clone()
        aabb_train_bg = 2 * aabb_train.clone()
        aabb_infer_bg = 2 * aabb_train.clone()

        self.register_buffer('aabb_train', aabb_train)
        self.register_buffer('aabb_infer', aabb_infer)
        self.register_buffer('aabb_train_bg', aabb_train_bg)
        self.register_buffer('aabb_infer_bg', aabb_infer_bg)

        # extra state for cuda raymarching
        if self.cuda_ray:
            # density grid
            density_grid = torch.zeros([self.cascade, self.grid_size ** 3])  # [CAS, H * H * H]
            density_bitfield = torch.zeros(self.cascade * self.grid_size ** 3 // 8,
                                           dtype=torch.uint8)  # [CAS * H * H * H // 8]
            self.register_buffer('density_grid', density_grid)
            self.register_buffer('density_bitfield', density_bitfield)
            self.mean_density = 0
            self.iter_density = 0
            # step counter
            step_counter = torch.zeros(16, 2, dtype=torch.int32)  # 16 is hardcoded for averaging...
            self.register_buffer('step_counter', step_counter)
            self.mean_count = 0
            self.local_step = 0

    def forward(self, x, d):
        raise NotImplementedError()

    def density(self, x):
        raise NotImplementedError()

    def density_mesh(self, x):
        raise NotImplementedError()

    def density_with_nablas(self, x):
        raise NotImplementedError()

    def forward_radiance(self, x, d):
        raise NotImplementedError()

    def forward_s(self):
        raise NotImplementedError()

    def color(self, x, d, mask=None, **kwargs):
        raise NotImplementedError()

    def reset_extra_state(self):
        if not self.cuda_ray:
            return
            # density grid
        self.density_grid.zero_()
        self.mean_density = 0
        self.iter_density = 0
        # step counter
        self.step_counter.zero_()
        self.mean_count = 0
        self.local_step = 0

    def run(self, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None, ambient_ratio=1.0, shading='albedo',
            bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)##todo

        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        # random sample light_d if not provided
        if light_d is None:
            # gaussian noise around the ray origin, so the light always face the view dir (avoid dark face)
            light_d = (rays_o[0] + torch.randn(3, device=device, dtype=torch.float))
            light_d = safe_normalize(light_d)

        # print(f'nears = {nears.min().item()} ~ {nears.max().item()}, fars = {fars.min().item()} ~ {fars.max().item()}')

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.
        # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = self.density(xyzs.reshape(-1, 3))

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1))  # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1])  # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps,
                                        det=not self.training).detach()  # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
                    -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])  # a manual clip.

            # only forward new points to save computation
            new_density_outputs = self.density(new_xyzs.reshape(-1, 3))
            # new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)  # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1)  # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1, index=z_index.unsqueeze(-1).expand_as(tmp_output))

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)

        sigmas, rgbs, normals = self(xyzs.reshape(-1, 3), dirs.reshape(-1, 3))
        if rgbs.shape[-1] > 3:
            n_dim = rgbs.shape[-1]-3
            rgbs, masks = rgbs.split([3,n_dim], dim=-1)
            masks = masks.view(N, -1, n_dim)  # [N, T+t, 3]
        else:
            masks = None
        sigmas = sigmas.view(N, -1, 1)
        rgbs = rgbs.view(N, -1, 3)  # [N, T+t, 3]
        if normals is not None:
            normals = normals.view(N, -1, 3)

        if self.opt.train_conf:
            results = self.weights_sum_i(sample_dist, sigmas, normals, dirs, weights, z_vals, nears, fars, rgbs, prefix, masks=masks, is_all=True)

            if self.opt.soft_mask:
                edit_mask = torch.sigmoid((masks - self.opt.conf_thr) * 100)
                sigmas_fg = sigmas.clone() * edit_mask
                sigmas_bg = sigmas.clone() * (1 - edit_mask)
            else:
                edit_mask = masks>0.5
                sigmas_bg = sigmas.clone()
                sigmas_bg[edit_mask] = 0
                sigmas_fg = sigmas.clone()
                sigmas_fg[~edit_mask] = 0

            # 什么weights是一样的吗
            results['sigma'] = sigmas
            results['rgbs'] = rgbs
            results['edit_mask'] = edit_mask
            results['fg']=self.weights_sum_i(sample_dist, sigmas_fg, normals, dirs, weights, z_vals, nears, fars, rgbs, prefix, masks=masks, if_fg=True)
            results['bg']=self.weights_sum_i(sample_dist, sigmas_bg, normals, dirs, weights, z_vals, nears, fars, rgbs, prefix, masks=masks)
            # dict_keys(['image', 'render_mask', 'depth', 'weights_sum', 'weights', 'mask', 'fg', 'bg'])

        return results

    def weights_sum_i(self, sample_dist, sigmas, normals, dirs, weights, z_vals, nears, fars, rgbs, prefix, masks=None, bg_color=None, if_fg=False, is_all=False):

        if is_all and self.opt.detach_bg:#detach的方式 todo
            edit_points = masks.mean(-1,keepdims=True) >= 0.5
            new_sig = torch.zeros_like(sigmas)
            new_rgb = torch.zeros_like(rgbs)
            new_sig[edit_points] = sigmas[edit_points]
            new_sig[~edit_points] = sigmas[~edit_points].detach()
            new_rgb[edit_points[...,-1]] = rgbs[edit_points[...,-1]]
            new_rgb[~edit_points[...,-1]] = rgbs[~edit_points[...,-1]].detach()
            sigmas = new_sig
            rgbs = new_rgb

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * sigmas.squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]

        results = {}
        if normals is not None:
            loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
            results['loss_orient'] = loss_orient.mean()

        # calculate weight_sum (mask)
        weights_sum = weights.sum(dim=-1)  # [N]
        # print(weights_sum)
        # calculate depth
        ori_z_vals = ((z_vals - nears) / (fars - nears)).clamp(0, 1)
        depth = torch.sum(weights * ori_z_vals, dim=-1)

        # calculate color
        image = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]

        # print(image)
        # sys.exit()
        if normals is not None:
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, dim=-2)


        image = image.view(*prefix, 3)
        depth = depth.view(*prefix)

        
        if if_fg and bg_color is not None:
            results['black_image'] = image
            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        
        if self.opt.train_conf:
            if self.opt.detach_mask_from_field:
                render_mask = torch.sum(weights.unsqueeze(-1).detach() * masks, dim=-2)  # [N, 3], in [0, 1]
            else:
                render_mask = torch.sum(weights.unsqueeze(-1) * masks, dim=-2)  # [N, 3], in [0, 1]
            render_mask = render_mask.view(*prefix, -1)
            results['render_mask'] = render_mask

        results['depth'] = depth
        if normals is not None:
            results['normal'] = normals
            results['normal_map'] = normal_map
        results['weights_sum'] = weights_sum
        results['weights'] = weights
        results['mask'] = mask
        return results 

    def run_cuda2(self, rays_o, rays_d, dt_gamma=0, light_d=None, ambient_ratio=1.0, shading='albedo',
                 bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.cuda()
        rays_d = rays_d.cuda()
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                     self.aabb_train if self.training else self.aabb_infer)

        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()  # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield,
                                                                    self.cascade, self.grid_size, nears, fars, counter,
                                                                    self.mean_count, perturb, 128,
                                                                    force_all_rays, dt_gamma, max_steps)

            # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

            sigmas, rgbs, normals = self(xyzs, dirs)
            rgbs = rgbs[...,:3]

            # print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs.float(), deltas, rays, T_thresh)

            # normals related regularizations
            if normals is not None and self.opt.if_smooth:
                # orientation loss
                weights = 1 - torch.exp(-sigmas)
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()

                # surface normal smoothness
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                loss_smooth = (normals - normals_perturb).abs()
                results['loss_smooth'] = loss_smooth.mean()

        else:

            # allocate outputs
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 3, dtype=dtype, device=device)
            # confi = torch.zeros(N, 1, dtype=dtype, device=device)
            normal_map = torch.zeros(N, 3, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)  # [N]
            rays_t = nears.clone()  # [N]

            step = 0

            while step < max_steps:  # hard coded max step
                # max_steps 1024

                # count alive rays
                n_alive = rays_alive.shape[0]
                # exit loop
                if n_alive <= 0:
                    break
                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1)
                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d,
                                                            self.bound, self.density_bitfield, self.cascade,
                                                            self.grid_size, nears, fars, 128,
                                                            perturb if step == 0 else False, dt_gamma, max_steps)
                # torch.Size([65664, 3]) torch.Size([65664, 2])
                sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                rgbs = rgbs[...,:3]

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)
                rays_alive = rays_alive[rays_alive >= 0]
                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step

        # image = image[...,:3]

        if self.opt.bg_color:
            color = np.array(self.opt.bg_color)
            bg_color = torch.tensor([color]).float().cuda()

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        # image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)

        # depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)

        weights_sum = weights_sum.reshape(*prefix)

        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        if normals is not None:
            results['normal'] = normals
            results['normal_map'] = normal_map
        results['weights_sum'] = weights_sum
        results['mask'] = mask

        return results

    def run_cuda(self, rays_o, rays_d, dt_gamma=0, light_d=None, ambient_ratio=1.0, shading='albedo',
                 bg_color=None, perturb=False, force_all_rays=False, max_steps=1024, T_thresh=1e-4, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: image: [B, N, 3], depth: [B, N]

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.cuda()
        rays_d = rays_d.cuda()
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        # pre-calculate near far
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d,
                                                     self.aabb_train if self.training else self.aabb_infer)

        results = {}

        if self.training:
            # setup counter
            counter = self.step_counter[self.local_step % 16]
            counter.zero_()  # set to 0
            self.local_step += 1

            xyzs, dirs, deltas, rays = raymarching.march_rays_train(rays_o, rays_d, self.bound, self.density_bitfield,
                                                                    self.cascade, self.grid_size, nears, fars, counter,
                                                                    self.mean_count, perturb, 128,
                                                                    force_all_rays, dt_gamma, max_steps)

            # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

            sigmas, rgbs, normals = self(xyzs, dirs)
            # rgbs = rgbs[...,:3]

            # print(f'valid RGB query ratio: {mask.sum().item() / mask.shape[0]} (total = {mask.sum().item()})')

            weights_sum, depth, image = raymarching.composite_rays_train(sigmas, rgbs.float(), deltas, rays, T_thresh)
            # assert image.shape[-1] == 4
            

            # normals related regularizations
            if normals is not None and self.opt.if_smooth:
                # orientation loss
                weights = 1 - torch.exp(-sigmas)
                loss_orient = weights.detach() * (normals * dirs).sum(-1).clamp(min=0) ** 2
                results['loss_orient'] = loss_orient.mean()

                # surface normal smoothness
                normals_perturb = self.normal(xyzs + torch.randn_like(xyzs) * 1e-2)
                loss_smooth = (normals - normals_perturb).abs()
                results['loss_smooth'] = loss_smooth.mean()

        else:

            # allocate outputs
            dtype = torch.float32

            weights_sum = torch.zeros(N, dtype=dtype, device=device)
            depth = torch.zeros(N, dtype=dtype, device=device)
            image = torch.zeros(N, 4, dtype=dtype, device=device)
            normal_map = torch.zeros(N, 3, dtype=dtype, device=device)

            n_alive = N
            rays_alive = torch.arange(n_alive, dtype=torch.int32, device=device)  # [N]
            rays_t = nears.clone()  # [N]

            step = 0

            while step < max_steps:  # hard coded max step
                # max_steps 1024

                # count alive rays
                n_alive = rays_alive.shape[0]
                # exit loop
                if n_alive <= 0:
                    break
                # decide compact_steps
                n_step = max(min(N // n_alive, 8), 1) #一次一步
                xyzs, dirs, deltas = raymarching.march_rays(n_alive, n_step, rays_alive, rays_t, rays_o, rays_d,
                                                            self.bound, self.density_bitfield, self.cascade,
                                                            self.grid_size, nears, fars, 128,
                                                            perturb if step == 0 else False, dt_gamma, max_steps)
                sigmas, rgbs, normals = self(xyzs, dirs, light_d, ratio=ambient_ratio, shading=shading)
                # rgbs = rgbs[...,:3]

                raymarching.composite_rays(n_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas, weights_sum, depth, image, T_thresh)
                rays_alive = rays_alive[rays_alive >= 0]
                # print(f'step = {step}, n_step = {n_step}, n_alive = {n_alive}, xyzs: {xyzs.shape}')

                step += n_step
                
        n_dim = image.shape[-1]
        if n_dim > 3:
            image, confi = image.split([3,n_dim-3], dim=-1)

        if self.opt.bg_color:
            color = np.array(self.opt.bg_color)
            bg_color = torch.tensor([color]).float().cuda()

            image = image + (1 - weights_sum).unsqueeze(-1) * bg_color

        # image = image + (1 - weights_sum).unsqueeze(-1) * bg_color
        image = image.view(*prefix, 3)

        # depth = torch.clamp(depth - nears, min=0) / (fars - nears)
        depth = depth.view(*prefix)

        weights_sum = weights_sum.reshape(*prefix)

        mask = (nears < fars).reshape(*prefix)

        results['image'] = image
        results['depth'] = depth
        if normals is not None:
            results['normal'] = normals
            results['normal_map'] = normal_map
        results['weights_sum'] = weights_sum
        results['mask'] = mask

        return results

    def sample_pts_linnear(self, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None, N_upsample_iters=1,
                           perturb=True, **kwargs):
        num_steps *= 2
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        return xyzs

    def sample_pts_sdf(self, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None, N_upsample_iters=1,
                       perturb=True, **kwargs):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                # query SDF and RGB
                density_outputs = self.density(xyzs.reshape(-1, 3))

                # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                for k, v in density_outputs.items():
                    density_outputs[k] = v.view(N, num_steps, -1)

                _sdf = density_outputs['sigma'].squeeze(-1)

                _d = z_vals
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = (
                        _sdf[..., :-1],
                        _sdf[..., 1:],
                    )  # (...,N_samples-1)
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (
                            next_z_vals - prev_z_vals + 1e-5
                    )
                    prev_dot_val = torch.cat(
                        [
                            torch.zeros_like(dot_val[..., :1], device=device),
                            dot_val[..., :-1],
                        ],
                        dim=-1,
                    )  # jianfei: prev_slope, right shifted,  (...,N_samples-1)
                    dot_val = torch.stack(
                        [prev_dot_val, dot_val], dim=-1
                    )  # jianfei: concat prev_slope with slope ,  (...,N_samples-1,2)
                    dot_val, _ = torch.min(
                        dot_val, dim=-1, keepdim=False
                    )  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope),  (...,N_samples-1)
                    dot_val = dot_val.clamp(-10.0, 0.0)

                    dist = next_z_vals - prev_z_vals
                    prev_esti_sdf = (
                            mid_sdf - dot_val * dist * 0.5
                    )
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5

                    phi_s_base = 64
                    # phi_s_base = 256
                    prev_cdf = cdf_Phi_s(
                        prev_esti_sdf, phi_s_base * (2 ** i)
                    )  # \VarPhi_s(x) in paper
                    next_cdf = cdf_Phi_s(next_esti_sdf, phi_s_base * (2 ** i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

                    _w = alpha_to_w(alpha)
                    d_fine = sample_pdf(
                        _d, _w, upsample_steps // N_upsample_iters, det=not perturb
                    )
                    _d = torch.cat([_d, d_fine], dim=-1)

                    pts_fine = rays_o.unsqueeze(-2) + d_fine.unsqueeze(
                        -1
                    ) * rays_d.unsqueeze(-2)

                    density_fine = self.density(pts_fine.reshape(-1, 3))
                    # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                    for k, v in density_fine.items():
                        density_fine[k] = v.view(N, upsample_steps // N_upsample_iters, -1)

                    sdf_fine = density_fine['sigma'].squeeze(-1)
                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, 1, d_sort_indices)

                new_z_vals = _d

        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]

        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)

        return xyzs.reshape(-1, 3), dirs.reshape(-1, 3), new_z_vals

    def sample_pts_sdf_composite(self, rays_o, rays_d, num_steps=64, upsample_steps=64, N_upsample_iters=4,
                                 perturb=False):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
        nears[nears > 1000] = nears.min()
        fars[fars > 1000] = 1.0

        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                # query SDF and RGB
                density_outputs = self.density(xyzs.reshape(-1, 3))

                # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                for k, v in density_outputs.items():
                    density_outputs[k] = v.view(N, num_steps, -1)

                _sdf = density_outputs['sigma'].squeeze(-1)

                _d = z_vals
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = (
                        _sdf[..., :-1],
                        _sdf[..., 1:],
                    )  # (...,N_samples-1)
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (
                            next_z_vals - prev_z_vals + 1e-5
                    )
                    prev_dot_val = torch.cat(
                        [
                            torch.zeros_like(dot_val[..., :1], device=device),
                            dot_val[..., :-1],
                        ],
                        dim=-1,
                    )  # jianfei: prev_slope, right shifted,  (...,N_samples-1)
                    dot_val = torch.stack(
                        [prev_dot_val, dot_val], dim=-1
                    )  # jianfei: concat prev_slope with slope ,  (...,N_samples-1,2)
                    dot_val, _ = torch.min(
                        dot_val, dim=-1, keepdim=False
                    )  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope),  (...,N_samples-1)
                    dot_val = dot_val.clamp(-10.0, 0.0)

                    dist = next_z_vals - prev_z_vals
                    prev_esti_sdf = (
                            mid_sdf - dot_val * dist * 0.5
                    )
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5

                    phi_s_base = 64
                    # phi_s_base = 256
                    prev_cdf = cdf_Phi_s(
                        prev_esti_sdf, phi_s_base * (2 ** i)
                    )  # \VarPhi_s(x) in paper
                    next_cdf = cdf_Phi_s(next_esti_sdf, phi_s_base * (2 ** i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

                    _w = alpha_to_w(alpha)
                    d_fine = sample_pdf(
                        _d, _w, upsample_steps // N_upsample_iters, det=not perturb
                    )
                    _d = torch.cat([_d, d_fine], dim=-1)

                    pts_fine = rays_o.unsqueeze(-2) + d_fine.unsqueeze(
                        -1
                    ) * rays_d.unsqueeze(-2)

                    density_fine = self.density(pts_fine.reshape(-1, 3))
                    # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                    for k, v in density_fine.items():
                        density_fine[k] = v.view(N, upsample_steps // N_upsample_iters, -1)

                    sdf_fine = density_fine['sigma'].squeeze(-1)
                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, 1, d_sort_indices)

                new_z_vals = _d

        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]

        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)

        return xyzs.reshape(-1, 3), dirs.reshape(-1, 3), new_z_vals

    def sample_pts_sdf_multi(self, rays_o, rays_d, scence_id, num_steps=128, upsample_steps=128, light_d=None,
                             N_upsample_iters=4,
                             perturb=True, **kwargs):
        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        aabb = self.aabb_train if self.training else self.aabb_infer
        aabbt = aabb.clone()
        if scence_id in [2, 3]:
            aabbt /= 5

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabbt, self.min_near)
        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                # query SDF and RGB
                density_outputs = self.density(xyzs.reshape(-1, 3), scence_id)

                # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                for k, v in density_outputs.items():
                    density_outputs[k] = v.view(N, num_steps, -1)

                _sdf = density_outputs['sigma'].squeeze(-1)

                _d = z_vals
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = (
                        _sdf[..., :-1],
                        _sdf[..., 1:],
                    )  # (...,N_samples-1)
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (
                            next_z_vals - prev_z_vals + 1e-5
                    )
                    prev_dot_val = torch.cat(
                        [
                            torch.zeros_like(dot_val[..., :1], device=device),
                            dot_val[..., :-1],
                        ],
                        dim=-1,
                    )  # jianfei: prev_slope, right shifted,  (...,N_samples-1)
                    dot_val = torch.stack(
                        [prev_dot_val, dot_val], dim=-1
                    )  # jianfei: concat prev_slope with slope ,  (...,N_samples-1,2)
                    dot_val, _ = torch.min(
                        dot_val, dim=-1, keepdim=False
                    )  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope),  (...,N_samples-1)
                    dot_val = dot_val.clamp(-10.0, 0.0)

                    dist = next_z_vals - prev_z_vals
                    prev_esti_sdf = (
                            mid_sdf - dot_val * dist * 0.5
                    )
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5

                    phi_s_base = 64
                    # phi_s_base = 256
                    prev_cdf = cdf_Phi_s(
                        prev_esti_sdf, phi_s_base * (2 ** i)
                    )  # \VarPhi_s(x) in paper
                    next_cdf = cdf_Phi_s(next_esti_sdf, phi_s_base * (2 ** i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)

                    _w = alpha_to_w(alpha)
                    d_fine = sample_pdf(
                        _d, _w, upsample_steps // N_upsample_iters, det=not perturb
                    )
                    _d = torch.cat([_d, d_fine], dim=-1)

                    pts_fine = rays_o.unsqueeze(-2) + d_fine.unsqueeze(
                        -1
                    ) * rays_d.unsqueeze(-2)

                    density_fine = self.density(pts_fine.reshape(-1, 3), scence_id)
                    # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                    for k, v in density_fine.items():
                        density_fine[k] = v.view(N, upsample_steps // N_upsample_iters, -1)

                    sdf_fine = density_fine['sigma'].squeeze(-1)
                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, 1, d_sort_indices)

                new_z_vals = _d

        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]

        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        return xyzs.reshape(-1, 3)


    def run_sdf(self, rays_o, rays_d, num_steps=128, upsample_steps=128, light_d=None, N_upsample_iters=4,
                ambient_ratio=1.0, shading='albedo',
                bg_color_test=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        ori_rays_o = rays_o.clone()
        ori_rays_d = rays_d.clone()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)

        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        # random sample light_d if not provided

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # upsample z_vals (nerf-like)

        if upsample_steps > 0:
            with torch.no_grad():
                # query SDF and RGB
                density_outputs = self.density(xyzs.reshape(-1, 3))
                # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                for k, v in density_outputs.items():
                    density_outputs[k] = v.view(N, num_steps, -1)
                _sdf = density_outputs['sigma'].squeeze(-1)

                _d = z_vals
                new_z_vals = torch.tensor([]).cuda().view(N, 0)
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = (
                        _sdf[..., :-1],
                        _sdf[..., 1:],
                    )  # (...,N_samples-1)
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (
                            next_z_vals - prev_z_vals + 1e-5
                    )
                    prev_dot_val = torch.cat(
                        [
                            torch.zeros_like(dot_val[..., :1], device=device),
                            dot_val[..., :-1],
                        ],
                        dim=-1,
                    )  # jianfei: prev_slope, right shifted,  (...,N_samples-1)
                    dot_val = torch.stack(
                        [prev_dot_val, dot_val], dim=-1
                    )  # jianfei: concat prev_slope with slope ,  (...,N_samples-1,2)
                    dot_val, _ = torch.min(
                        dot_val, dim=-1, keepdim=False
                    )  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope),  (...,N_samples-1)
                    dot_val = dot_val.clamp(-10.0, 0.0)

                    dist = next_z_vals - prev_z_vals
                    prev_esti_sdf = (
                            mid_sdf - dot_val * dist * 0.5
                    )
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5

                    phi_s_base = 64
                    prev_cdf = cdf_Phi_s(
                        prev_esti_sdf, phi_s_base * (2 ** i)
                    )  # \VarPhi_s(x) in paper
                    next_cdf = cdf_Phi_s(next_esti_sdf, phi_s_base * (2 ** i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                    _w = alpha_to_w(alpha)
                    d_fine = sample_pdf(
                        _d, _w, upsample_steps // N_upsample_iters, det=not perturb
                    )

                    new_z_vals = torch.cat([new_z_vals, d_fine], dim=-1)
                    _d = torch.cat([_d, d_fine], dim=-1)

                    pts_fine = rays_o.unsqueeze(-2) + d_fine.unsqueeze(-1) * rays_d.unsqueeze(-2)


                    density_fine = self.density(pts_fine.reshape(-1, 3))
                    # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                    for k, v in density_fine.items():
                        density_fine[k] = v.view(N, upsample_steps // N_upsample_iters, -1)
                    sdf_fine = density_fine['sigma'].squeeze(-1)

                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, 1, d_sort_indices)

                    new_z_vals, new_z_vals_sort_indices = torch.sort(new_z_vals, dim=-1)

                    new_z_vals = _d

        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]

        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.
        d_mid = 0.5 * (new_z_vals[..., 1:] + new_z_vals[..., :-1])
        xyz_mid = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * d_mid.unsqueeze(
            -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
        xyz_mid = torch.min(torch.max(xyz_mid, aabb[:3]), aabb[3:])  # a manual clip.

        out_fine = self.density_with_nablas(xyzs.reshape(-1, 3))  # , if_detach=False)
        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in out_fine.items():
            if k == 'ind_occu':
                continue
            out_fine[k] = v.view(N, upsample_steps + num_steps, -1)
        sdf = out_fine['sigma'].squeeze(-1)

        nablas = out_fine['nabla']

        cdf, opacity_alpha = sdf_to_alpha(
            sdf, self.forward_s()
        )

        dirs_mid = rays_d.view(-1, 1, 3).expand_as(xyz_mid)
        out = self.forward_radiance(xyz_mid.reshape(-1, 3), dirs_mid.reshape(-1, 3))
        if len(out) == 2:
            radiances, ind_occu_mid = out
        else:
            radiances = out

        d_final = d_mid
        visibility_weights = alpha_to_w(opacity_alpha)

        radiances = radiances.reshape(-1, upsample_steps + num_steps - 1, 3)

        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)

        depth_map = torch.sum(
            visibility_weights
            / (visibility_weights.sum(-1, keepdim=True) + 1e-10)
            * d_final,
            -1,
        )
        acc_map = torch.sum(visibility_weights, -1)

        if bg_color_test:
            rgb_map = rgb_map + (1. - acc_map[..., None]) * bg_color_test

        if self.opt.bg_color:
            color = np.array(self.opt.bg_color)
            bg_color = torch.tensor([color]).float().cuda()
            rgb_map = rgb_map + (1. - acc_map[..., None]) * bg_color

        normals_map = F.normalize(nablas, dim=-1)
        N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
        normals_map = (normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]).sum(dim=-2)

        if self.opt.if_mask and 'ind_occu' in out_fine.keys():
            ind_occu = out_fine['ind_occu']
            pred_norm = nablas.reshape(-1, 3)[ind_occu]
        else:
            pred_norm = nablas.reshape(-1, 3)

        pred_norm = torch.norm(pred_norm, dim=-1).reshape(-1)
        pred_norm = pred_norm[pred_norm > 0]
        loss_eikonal = F.mse_loss(pred_norm, pred_norm.new_ones(pred_norm.shape), reduction='mean')
        results['loss_eikonal'] = loss_eikonal

        if nablas is not None and self.opt.if_smooth:
            # orientation loss

            # loss_orient = acc_map[..., None].detach() * (nablas[:, :-1, :] * dirs).sum(-1).clamp(min=0) ** 2
            # results['loss_orient'] = loss_orient.mean()

            # surface normal smoothness

            _, normals_perturb, _, _ = self.density_with_nablas(xyzs.reshape(-1, 3) +
                                                                torch.randn_like(xyzs.reshape(-1, 3)) * 1e-2,
                                                                return_dict=False)

            if self.opt.if_mask:
                ind_occu = out_fine['ind_occu']
                loss_smooth = (nablas.reshape(-1, 3)[ind_occu] - normals_perturb.reshape(-1, 3)[ind_occu]).abs()
            else:
                loss_smooth = (nablas.reshape(-1, 3) - normals_perturb.reshape(-1, 3)).abs()
            results['loss_smooth'] = loss_smooth.mean()

        mask = (nears < fars).reshape(*prefix)

        results['xyzs'] = xyzs
        results['xyz_mid'] = xyz_mid
        results['dirs_mid'] = dirs_mid

        results['rad_sdf'] = sdf.reshape(-1, num_steps + upsample_steps)
        results['radiances'] = radiances.reshape(-1, num_steps + upsample_steps - 1, 3)

        if self.opt.backbone == 'mesh_neus' and 'ind_occu' in out_fine.keys():
            results['ind_occu'] = out_fine['ind_occu']
            results['ind_occu_mid'] = ind_occu_mid

        results['image'] = rgb_map
        results['depth'] = depth_map
        results['normal_map'] = normals_map
        results['normal'] = nablas
        results['weights_sum'] = acc_map
        results['mask'] = mask

        return results

    def run_composite(self, rays_o, rays_d, bg_model, num_steps=128, upsample_steps=128, light_d=None,
                      N_upsample_iters=4, ambient_ratio=1.0, shading='albedo',
                      bg_color=None, perturb=False, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # bg_color: [BN, 3] in range [0, 1]
        # return: image: [B, N, 3], depth: [B, N]

        ori_rays_o = rays_o.clone()
        ori_rays_d = rays_d.clone()

        prefix = rays_o.shape[:-1]
        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact
        device = rays_o.device

        results = {}

        # choose aabb
        aabb = self.aabb_train if self.training else self.aabb_infer

        # sample steps

        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)

        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist
            # z_vals = z_vals.clamp(nears, fars) # avoid out of bounds xyzs.

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():
                # query SDF and RGB
                density_outputs = self.density(xyzs.reshape(-1, 3))

                # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                for k, v in density_outputs.items():
                    density_outputs[k] = v.view(N, num_steps, -1)

                _sdf = density_outputs['sigma'].squeeze(-1)
                # print(_sdf[0])
                _d = z_vals
                for i in range(N_upsample_iters):
                    prev_sdf, next_sdf = (
                        _sdf[..., :-1],
                        _sdf[..., 1:],
                    )  # (...,N_samples-1)
                    prev_z_vals, next_z_vals = _d[..., :-1], _d[..., 1:]
                    mid_sdf = (prev_sdf + next_sdf) * 0.5
                    dot_val = (next_sdf - prev_sdf) / (
                            next_z_vals - prev_z_vals + 1e-5
                    )
                    prev_dot_val = torch.cat(
                        [
                            torch.zeros_like(dot_val[..., :1], device=device),
                            dot_val[..., :-1],
                        ],
                        dim=-1,
                    )  # jianfei: prev_slope, right shifted,  (...,N_samples-1)
                    dot_val = torch.stack(
                        [prev_dot_val, dot_val], dim=-1
                    )  # jianfei: concat prev_slope with slope ,  (...,N_samples-1,2)
                    dot_val, _ = torch.min(
                        dot_val, dim=-1, keepdim=False
                    )  # jianfei: find the minimum of prev_slope and current slope. (forward diff vs. backward diff., or the prev segment's slope vs. this segment's slope),  (...,N_samples-1)
                    dot_val = dot_val.clamp(-10.0, 0.0)

                    dist = next_z_vals - prev_z_vals
                    prev_esti_sdf = (
                            mid_sdf - dot_val * dist * 0.5
                    )
                    next_esti_sdf = mid_sdf + dot_val * dist * 0.5

                    phi_s_base = 64
                    # phi_s_base = 256
                    prev_cdf = cdf_Phi_s(
                        prev_esti_sdf, phi_s_base * (2 ** i)
                    )  # \VarPhi_s(x) in paper
                    next_cdf = cdf_Phi_s(next_esti_sdf, phi_s_base * (2 ** i))
                    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
                    # print(alpha[0])
                    _w = alpha_to_w(alpha)
                    # print(_w[0])
                    # print(_w.size())
                    # sys.exit()
                    d_fine = sample_pdf(
                        _d, _w, upsample_steps // N_upsample_iters, det=not perturb
                    )
                    _d = torch.cat([_d, d_fine], dim=-1)

                    pts_fine = rays_o.unsqueeze(-2) + d_fine.unsqueeze(
                        -1
                    ) * rays_d.unsqueeze(-2)

                    density_fine = self.density(pts_fine.reshape(-1, 3))
                    # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
                    for k, v in density_fine.items():
                        density_fine[k] = v.view(N, upsample_steps // N_upsample_iters, -1)

                    sdf_fine = density_fine['sigma'].squeeze(-1)
                    _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
                    _d, d_sort_indices = torch.sort(_d, dim=-1)
                    _sdf = torch.gather(_sdf, 1, d_sort_indices)

                new_z_vals = _d

        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]

        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.
        d_mid = 0.5 * (new_z_vals[..., 1:] + new_z_vals[..., :-1])
        xyz_mid = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * d_mid.unsqueeze(
            -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
        xyz_mid = torch.min(torch.max(xyz_mid, aabb[:3]), aabb[3:])  # a manual clip.

        out_fine = self.density_with_nablas(xyzs.reshape(-1, 3))  # , if_detach=False)
        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in out_fine.items():
            if k == 'ind_occu':
                continue
            out_fine[k] = v.view(N, upsample_steps + num_steps, -1)
        sdf = out_fine['sigma'].squeeze(-1)

        nablas = out_fine['nabla']

        cdf, opacity_alpha = sdf_to_alpha(
            sdf, self.forward_s()
        )

        dirs_mid = rays_d.view(-1, 1, 3).expand_as(xyz_mid)
        out = self.forward_radiance(xyz_mid.reshape(-1, 3), dirs_mid.reshape(-1, 3))
        if len(out) == 2:
            radiances, ind_occu_mid = out
        else:
            radiances = out

        d_final = d_mid
        visibility_weights = alpha_to_w(opacity_alpha)

        radiances = radiances.reshape(-1, upsample_steps + num_steps - 1, 3)

        rgb_map = torch.sum(visibility_weights[..., None] * radiances, -2)

        depth_map = torch.sum(
            visibility_weights
            / (visibility_weights.sum(-1, keepdim=True) + 1e-10)
            * d_final,
            -1,
        )
        acc_map = torch.sum(visibility_weights, -1)

        normals_map = F.normalize(nablas, dim=-1)
        N_pts = min(visibility_weights.shape[-1], normals_map.shape[-2])
        normals_map = (normals_map[..., :N_pts, :] * visibility_weights[..., :N_pts, None]).sum(dim=-2)

        if self.opt.if_mask and 'ind_occu' in out_fine.keys():
            ind_occu = out_fine['ind_occu']
            pred_norm = nablas.reshape(-1, 3)[ind_occu]
        else:
            pred_norm = nablas.reshape(-1, 3)

        pred_norm = torch.norm(pred_norm, dim=-1).reshape(-1)
        pred_norm = pred_norm[pred_norm > 0]
        loss_eikonal = F.mse_loss(pred_norm, pred_norm.new_ones(pred_norm.shape), reduction='mean')
        results['loss_eikonal'] = loss_eikonal

        # with torch.no_grad():
        # 进行背景运算
        rays_o = ori_rays_o
        rays_d = ori_rays_d

        rays_o = rays_o.contiguous().view(-1, 3)
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_o.shape[0]  # N = B * N, in fact

        # choose aabb
        aabb = self.aabb_train_bg if self.training else self.aabb_infer_bg

        # sample steps
        # nears = fars
        nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)

        nears.unsqueeze_(-1)
        fars.unsqueeze_(-1)

        num_steps = int(num_steps * 2)
        upsample_steps = int(upsample_steps * 2)

        z_vals = torch.linspace(0.0, 1.0, num_steps, device=device).unsqueeze(0)  # [1, T]
        z_vals = z_vals.expand((N, num_steps))  # [N, T]
        z_vals = nears + (fars - nears) * z_vals  # [N, T], in [nears, fars]

        # perturb z_vals
        sample_dist = (fars - nears) / num_steps
        if perturb:
            z_vals = z_vals + (torch.rand(z_vals.shape, device=device) - 0.5) * sample_dist

        # generate xyzs
        xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(
            -1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]
        xyzs = torch.min(torch.max(xyzs, aabb[:3]), aabb[3:])  # a manual clip.

        # plot_pointcloud(xyzs.reshape(-1, 3).detach().cpu().numpy())

        # query SDF and RGB
        density_outputs = bg_model.density(xyzs.reshape(-1, 3))

        # sigmas = density_outputs['sigma'].view(N, num_steps) # [N, T]
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(N, num_steps, -1)

        # upsample z_vals (nerf-like)
        if upsample_steps > 0:
            with torch.no_grad():

                deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
                deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)

                alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1))  # [N, T]
                alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15],
                                           dim=-1)  # [N, T+1]
                weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T]

                # sample new z_vals
                z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1])  # [N, T-1]
                new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1], upsample_steps,
                                        det=not self.training).detach()  # [N, t]

                new_xyzs = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * new_z_vals.unsqueeze(
                    -1)  # [N, 1, 3] * [N, t, 1] -> [N, t, 3]
                new_xyzs = torch.min(torch.max(new_xyzs, aabb[:3]), aabb[3:])  # a manual clip.

            # only forward new points to save computation
            new_density_outputs = bg_model.density(new_xyzs.reshape(-1, 3))
            # new_sigmas = new_density_outputs['sigma'].view(N, upsample_steps) # [N, t]
            for k, v in new_density_outputs.items():
                new_density_outputs[k] = v.view(N, upsample_steps, -1)

            # re-order
            z_vals = torch.cat([z_vals, new_z_vals], dim=1)  # [N, T+t]
            z_vals, z_index = torch.sort(z_vals, dim=1)

            xyzs = torch.cat([xyzs, new_xyzs], dim=1)  # [N, T+t, 3]
            xyzs = torch.gather(xyzs, dim=1, index=z_index.unsqueeze(-1).expand_as(xyzs))

            for k in density_outputs:
                tmp_output = torch.cat([density_outputs[k], new_density_outputs[k]], dim=1)
                density_outputs[k] = torch.gather(tmp_output, dim=1,
                                                  index=z_index.unsqueeze(-1).expand_as(tmp_output))

        if self.opt.object_bound:
            aabb = np.array(self.opt.object_bound)
            aabb = torch.tensor([aabb]).float().cuda()
            # choose aabb
            # aabb = torch.FloatTensor([-0.07, -0.15, -0.18, 0.08, 0.3, 0.15]).cuda()
            nears, fars = raymarching.near_far_from_aabb(rays_o, rays_d, aabb, self.min_near)
            nears = nears.unsqueeze(1)
            fars = fars.unsqueeze(1)

            mask1 = z_vals > nears  # .repeat(1, z_vals.size()[1])
            mask2 = z_vals < fars
            mask = mask1 * mask2
            density_outputs['sigma'][mask] = 0

        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T+t-1]
        deltas = torch.cat([deltas, sample_dist * torch.ones_like(deltas[..., :1])], dim=-1)
        alphas = 1 - torch.exp(-deltas * density_outputs['sigma'].squeeze(-1))  # [N, T+t]
        alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
        weights = alphas * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]

        dirs = rays_d.view(-1, 1, 3).expand_as(xyzs)
        for k, v in density_outputs.items():
            density_outputs[k] = v.view(-1, v.shape[-1])

        sigmas, rgbs, _ = bg_model(xyzs.reshape(-1, 3), dirs.reshape(-1, 3))

        rgbs = rgbs.view(N, -1, 3)  # [N, T+t, 3]

        # calculate color
        bg_color = torch.sum(weights.unsqueeze(-1) * rgbs, dim=-2)  # [N, 3], in [0, 1]

        if self.opt.bg_color:
            weights_sum = weights.sum(dim=-1)
            color = np.array(self.opt.bg_color)
            bg_color1 = torch.tensor([color]).float().cuda()
            bg_color = bg_color + (1 - weights_sum).unsqueeze(-1) * bg_color1

        rgb_map = rgb_map + (1. - acc_map[..., None]) * bg_color

        mask = (nears < fars).reshape(*prefix)

        num_steps = int(num_steps * 0.5)
        upsample_steps = int(upsample_steps * 0.5)

        results['xyzs'] = xyzs
        results['xyz_mid'] = xyz_mid
        results['dirs_mid'] = dirs_mid

        results['rad_sdf'] = sdf.reshape(-1, num_steps + upsample_steps)
        results['radiances'] = radiances.reshape(-1, num_steps + upsample_steps - 1, 3)
        if self.opt.backbone == 'mesh_sdf' and 'ind_occu' in out_fine.keys():
            results['ind_occu'] = out_fine['ind_occu']
            results['ind_occu_mid'] = ind_occu_mid

        results['image'] = rgb_map
        results['depth'] = depth_map
        results['normal_map'] = normals_map
        results['normal'] = nablas
        results['weights_sum'] = acc_map
        results['mask'] = mask

        return results

    @torch.no_grad()
    def update_extra_state(self, decay=0.95, S=128):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return

            ### update density grid
        tmp_grid = - torch.ones_like(self.density_grid)

        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        for xs in X:
            for ys in Y:
                for zs in Z:

                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                       dim=-1)  # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long()  # [N]
                    xyzs = 2 * coords.float() / (self.grid_size - 1) - 1  # [N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        # bound = self.bound
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_xyzs = xyzs * (bound - half_grid_size)

                        # add noise in [-hgs, hgs]
                        cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size

                        # query density
                        sigmas = self.density(cas_xyzs)['sigma'].reshape(-1).detach()

                        # assign
                        tmp_grid[cas, indices] = sigmas.float()

        # ema update
        valid_mask = self.density_grid >= 0
        self.density_grid[valid_mask] = torch.maximum(self.density_grid[valid_mask] * decay, tmp_grid[valid_mask])
        self.mean_density = torch.mean(self.density_grid[valid_mask]).item()
        self.iter_density += 1

        # convert to bitfield
        # print(self.mean_density, self.density_thresh)
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

        # print(f'[density grid] min={self.density_grid.min().item():.4f}, max={self.density_grid.max().item():.4f}, mean={self.mean_density:.4f}, occ_rate={(self.density_grid > density_thresh).sum() / (128**3 * self.cascade):.3f} | [step counter] mean={self.mean_count}')

    def render(self, rays_o, rays_d, staged=False, max_ray_batch=2048, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        if self.cuda_ray and not 'neus' in self.opt.backbone:
            _run = self.run_cuda
        elif 'neus' in self.opt.backbone:
            _run = self.run_sdf
        else:
            _run = self.run

        results = _run(rays_o, rays_d, **kwargs)
        del rays_o, rays_d

        return results

    def render_dyback_O2(self, rays_o, rays_d, staged=False, max_ray_batch=2048, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        _run = self.run

        # _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # results = _run(rays_o, rays_d, **kwargs)

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            image = torch.empty((B, N, 3), device=device)
            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b + 1, head:tail], rays_d[b:b + 1, head:tail], **kwargs)
                    image[b:b + 1, head:tail] = results_['image']
                    head += max_ray_batch

            results = {}
            results['image'] = image

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results

    def render_cuda(self, rays_o, rays_d, staged=False, max_ray_batch=4096, **kwargs):
        # rays_o, rays_d: [B, N, 3], assumes B == 1
        # return: pred_rgb: [B, N, 3]

        _run = self.run_cuda

        # _run = self.run

        B, N = rays_o.shape[:2]
        device = rays_o.device

        # results = _run(rays_o, rays_d, **kwargs)

        # never stage when cuda_ray
        if staged and not self.cuda_ray:
            depth = torch.empty((B, N), device=device)
            rad_sdf = torch.empty((B, N, self.opt.num_steps + self.opt.upsample_steps), device=device)
            radiances = torch.empty((B, N, self.opt.num_steps + self.opt.upsample_steps - 1, 3), device=device)
            image = torch.empty((B, N, 3), device=device)
            normal = torch.empty((B, N, self.opt.num_steps + self.opt.upsample_steps, 3), device=device)
            normal_map = torch.empty((B, N, 3), device=device)
            weights_sum = torch.empty((B, N), device=device)

            for b in range(B):
                head = 0
                while head < N:
                    tail = min(head + max_ray_batch, N)
                    results_ = _run(rays_o[b:b + 1, head:tail], rays_d[b:b + 1, head:tail], **kwargs)
                    depth[b:b + 1, head:tail] = results_['depth']
                    if 'sdf' in self.opt.backbone and 'tcnn' not in self.opt.backbone:
                        rad_sdf[b:b + 1, head:tail] = results_['rad_sdf']
                        radiances[b:b + 1, head:tail] = results_['radiances']
                    weights_sum[b:b + 1, head:tail] = results_['weights_sum']
                    image[b:b + 1, head:tail] = results_['image']
                    if 'normal' in results_.keys():
                        normal[b:b + 1, head:tail] = results_['normal']
                        normal_map[b:b + 1, head:tail] = results_['normal_map']
                    head += max_ray_batch

            results = {}
            if 'sdf' in self.opt.backbone and 'tcnn' not in self.opt.backbone:
                results['rad_sdf'] = rad_sdf
                results['radiances'] = radiances
            results['depth'] = depth
            if 'normal' in results_.keys():
                results['normal'] = normal
                results['normal_map'] = normal_map
            results['image'] = image
            results['weights_sum'] = weights_sum

        else:
            results = _run(rays_o, rays_d, **kwargs)

        return results