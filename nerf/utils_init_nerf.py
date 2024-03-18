import os
import glob
import time
import cv2
import tqdm
import imageio
import copy
import tensorboardX
import clip
import torchvision.transforms as T
import torch
import numpy as np

import torch

import torch.nn as nn
from nerf.sd import StableDiffusion
from nerf.clip import CLIP
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from rich.console import Console


class Trainer_Nerf(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=5,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        self.ema = None
        if self.opt.no_scalar:
            self.scaler = None
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(self.opt)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.opt.editing_from:
                self.log(f"[INFO] Loading {self.opt.editing_from} ...")
                self.load_checkpoint(self.opt.editing_from, model_only=True)

            if self.opt.use_ckpt == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.opt.use_ckpt == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.opt.use_ckpt == "latest_model":##
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            else: 
                self.log(f"[INFO] Loading {self.opt.use_ckpt} ...")
                self.load_checkpoint(self.opt.use_ckpt)

        self.model_pretrained = self.model

        if self.opt.pretrained:
            self.model_pretrained = copy.deepcopy(self.model)
            if self.opt.editing_from is None:
                self.opt.editing_from = self.opt.use_ckpt
            checkpoint_dict = torch.load(self.opt.editing_from, map_location=self.device)
            missing_keys, unexpected_keys = self.model_pretrained.load_state_dict(
                checkpoint_dict['model'], strict=False)
            print("[INFO] loaded pretrained model.")

            if not self.opt.test:
                self.guidance = StableDiffusion(self.device, opt.sd_version, opt=opt)
                self.guidance.set_system(self)
                self.clip_guidance = CLIP(self.device)

                for p in self.guidance.parameters():
                    p.requires_grad = False
                for p in self.clip_guidance.parameters():
                    p.requires_grad = False
                self.prepare_text_embeddings()

        self.metrics_list = []
        self.edit_box_dict = {}
        self.pt_dict = {}
        self.attn_dict = {}
        self.feat_dict = {}


    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    def train_step_pretrain(self, data):
        torch.cuda.empty_cache()

        rgbs, mask, rays_o, rays_d, H, W, img_path = data
        rgbs, mask, rays_o, rays_d = rgbs.to(torch.float), mask.to(torch.float), rays_o.to(torch.float), rays_d.to(torch.float)

        rgbs = rgbs.reshape(1,-1,3)
        mask = mask.reshape(1,-1,1)
        rays_o = rays_o.reshape(1,-1,3)
        rays_d = rays_d.reshape(1,-1,3)

        rgbs = rgbs.to(self.device)
        mask = mask.to(self.device)
        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)

        if self.opt.batch_rays:
            select_inds = np.random.choice(rays_o.shape[1], size=[self.opt.batch_rays], replace=False) # 1024
            rgbs = rgbs[:, select_inds, :]
            mask = mask[:, select_inds, :]
            rays_o = rays_o[:, select_inds, :]
            rays_d = rays_d[:, select_inds, :]

        B, N = rays_o.shape[:2]

        outputs = self.model.render(rays_o, rays_d, staged=False, perturb=True, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image']
        mask_volume = outputs['weights_sum'].reshape(B, N)
        mask_volume = torch.clamp(mask_volume, 1e-5, 1 - 1e-5)

        loss_c = self.opt.train_rgb * F.mse_loss(pred_rgb.reshape(-1, 3), rgbs.reshape(-1, 3), reduction='mean')
        loss = loss_c

        loss_dict = {
            'loss_c': loss_c.item(),
        }

        if self.opt.train_conf:
            pred_mask = outputs['render_mask']
            loss_c = self.opt.train_conf * F.mse_loss(pred_mask.reshape(-1), mask.reshape(-1), reduction='mean')
            loss += loss_c

            loss_dict.update({
                'loss_m': loss_c.item(),
            })

        del data, rgbs, mask, rays_o, rays_d
        return pred_rgb, mask_volume, loss, loss_dict

    def get_pt(self, rays_o, rays_d, img_path, bg_color, B, H, W,):
        if img_path not in self.pt_dict:
            outputs_pt = self.model_pretrained.render(
                rays_o, rays_d, staged=False, perturb=True, bg_color=bg_color, force_all_rays=True, **vars(self.opt))  
            pt_mask = outputs_pt['render_mask'].reshape(B, H, W, -1).contiguous()
            pt_rgb = outputs_pt['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            pt_rgb_bg = outputs_pt['bg']['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            pt_rgb_fg = outputs_pt['fg']['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            pt_depth_fg = outputs_pt['fg']['depth'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()

            match_probs = None
            if self.opt.clip_view:
                with torch.no_grad():
                    images = self.clip_guidance.transformCLIP(pt_rgb)
                    logits_per_image, logits_per_text = self.clip_guidance.model(images, self.clip_match_text)
                    match_probs = logits_per_image.softmax(dim=1)

            self.pt_dict[img_path] = (pt_rgb_bg.cpu().detach(), pt_rgb_fg.cpu().detach(), pt_mask.cpu().detach(), pt_depth_fg.cpu().detach(), match_probs)
        else:
            pt_rgb_bg, pt_rgb_fg, pt_mask, pt_depth_fg = [t.detach().cuda() for t in self.pt_dict[img_path][:-1]]
            match_probs = self.pt_dict[img_path][-1]
            
        return pt_rgb_fg, pt_rgb_bg, pt_mask, pt_depth_fg, match_probs

    def get_textz(self, B, match_probs=None):
        if self.opt.clip_view:
            select = match_probs.max(-1).indices.tolist()
            text_z = torch.cat([self.text_z[t].clone() for t in select])
            text_z_fg = torch.cat([self.text_z_fg[t].clone() for t in select])
            text_z_norm = torch.cat([self.text_z_norm[t].clone() for t in select])
            text_z_norm_fg = torch.cat([self.text_z_norm_fg[t].clone() for t in select])
        else:
            text_z = self.text_z
            text_z_fg = self.text_z_fg
            text_z_norm = self.text_z_norm
            text_z_norm_fg = self.text_z_norm_fg

        return text_z, text_z_fg, text_z_norm, text_z_norm_fg, None
    
    def train_step_sd(self, pred_rgb, outputs, B,H,W,img_path, match_probs, pose=None, tuning_cls=False):
        text_z, text_z_fg, text_z_norm, text_z_norm_fg, text_z_fg2 = self.get_textz(B, match_probs)
        pred_rgb_fg = outputs['fg']['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]

        t_ratio = 1
        if self.opt.g_only:
            text_emb = text_z
            img_rgb = pred_rgb
        elif self.opt.l_only:
            text_emb = text_z_fg
            img_rgb = pred_rgb_fg
            t_ratio = self.opt.local_t_ratio
        else:
            if np.random.random() < self.opt.global_ratio:
                text_emb = text_z
                img_rgb = pred_rgb
            else:
                text_emb = text_z_fg
                img_rgb = pred_rgb_fg
                t_ratio = self.opt.local_t_ratio

        pred_rgb_512 = F.interpolate(img_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.guidance.encode_imgs(pred_rgb_512)

        loss_dict = {}
        loss_sd, loss_dict = self.guidance.train_step(latents, text_emb, mask=None, gt_rgb=None, img_path=img_path, system=self, camera=pose, tuning_cls=tuning_cls, t_ratio=t_ratio)
        return loss_sd, loss_dict

    # calculate the text embs.
    def prepare_text_embeddings(self):
        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        def get_sd_text_emb(input_text):
            if self.opt.clip_view:
                text_z = []
                for d in ['front', 'side', 'back']:#, 'side', 'overhead', 'bottom']:
                    # construct dir-encoded text
                    text = f"{input_text}, {d} view"
                    text_z_i = self.guidance.get_text_embeds([text], [self.opt.negative])
                    text_z.append(text_z_i)
            elif not self.opt.dir_text:
                text_z = self.guidance.get_text_embeds(
                    [input_text], [self.opt.negative])
            else:
                assert False
            return text_z
        
        self.text_z = get_sd_text_emb(self.opt.text)
        self.text_z_fg = get_sd_text_emb(self.opt.text_fg)
        self.text_z_norm = get_sd_text_emb(self.opt.text_norm)
        self.text_z_norm_fg = get_sd_text_emb(self.opt.text_fg_norm)
        self.text_z_bg = self.guidance.get_text_embeds([self.opt.text_bg], [self.opt.negative])

        # self.text_bg_z = self.clip_guidance.get_text_embeds(self.opt.text_bg)
        # self.text_fg_z = self.clip_guidance.get_text_embeds(self.opt.text_fg_norm)

        if self.opt.clip_view:
            # prompt = ["a bear statue's front view in a forest", "a bear statue's back view in a forest"]
            # prompt = ["front view of an object", "side view of an object", 'back view of an object']
            prompt = ["front face of an object", "side face of an object", 'back face of an object']
            self.clip_match_text = clip.tokenize(prompt).to(self.device)#torch.Size([3, 77])
            self.clip_trans = T.Compose(
                [
                    T.Resize((224,224)),
                    T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
                ]
            )

    def train_step_editing(self, data):
        rgbs, mask, rays_o, rays_d, H, W, img_path = data
        B, N = rays_o.shape[:2]

        if self.opt.random_bg_c:
            bg_color = torch.rand((1,3), device=rays_o.device).repeat(B*N,1) 
        elif self.opt.black_bg_c:
            bg_color = torch.zeros((1,3), device=rays_o.device).repeat(B*N,1) 
        elif self.opt.white_bg_c:
            bg_color = torch.ones((1,3), device=rays_o.device).repeat(B*N,1) 
        else:
            bg_color = None
        outputs = self.model.render(rays_o, rays_d, staged=False, perturb=True, force_all_rays=True, bg_color=bg_color, **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
        pred_depth = outputs['depth'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()  # [1, 1, H, W]
        pred_ws = outputs['weights_sum'].reshape(B, H, W)
        pred_ws_fg = outputs['fg']['weights_sum'].reshape(B, H, W)
        pred_mask = outputs['render_mask'].reshape(B, H, W, -1)
        pred_rgb_fg = outputs['fg']['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
        pred_depth_fg = outputs['fg']['depth'].reshape(B, H, W, 1).permute(0, 3, 1, 2).contiguous()  # [1, 1, H, W]
        pred_rgb_bg = outputs['bg']['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]

        pt_rgb_fg, pt_rgb_bg, pt_mask, pt_depth_fg, match_probs = self.get_pt(rays_o, rays_d, img_path, bg_color, B, H, W)

        if self.opt.ori_bg:
            non_edit = (pt_mask + pred_mask) < 0.5
            pt_rgb_bg = rgbs.reshape(B, H, W, 3).permute(0, 3, 1, 2) * non_edit + (~non_edit) * pt_rgb_bg

        loss_dict = {}
        if self.opt.lambda_sd:
            loss_sd, loss_dict_sd = self.train_step_sd(pred_rgb, outputs, B,H,W,img_path, match_probs=match_probs, pose = None, tuning_cls=False)
            loss = loss_sd
            loss_dict.update(loss_dict_sd)

        if self.opt.keep_bg:
            loss_bg = self.opt.keep_bg * F.l1_loss(pt_rgb_bg, pred_rgb_bg)
            loss += loss_bg
            loss_dict.update(dict(loss_bg=loss_bg.item()))

        del data, outputs
        return pred_rgb, pred_ws, loss, loss_dict

    def eval_step(self, data):
        rgbs, mask, rays_o, rays_d, H, W, img_path = data

        rgbs = rgbs.to(self.device)
        mask = mask.to(self.device)
        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)

        B, N = rays_o.shape[:2]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        kwargs = vars(self.opt)

        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False, light_d=light_d,
                                    ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                    **kwargs)

        gt_rgb = rgbs.reshape(B,H,W,3)
        all_pred_rgb = outputs['image'].reshape(B, H, W, 3)
        all_pred_depth = outputs['depth'].reshape(B, H, W, 1).repeat(1,1,1,3)
        im_list = [gt_rgb,all_pred_rgb,all_pred_depth]

        if self.opt.train_conf:
            gt_mask = mask.reshape(B, H, W, 1).repeat(1,1,1,3)
            all_pred_mask = outputs['render_mask'].reshape(B, H, W, -1).mean(-1,keepdims=True).repeat(1,1,1,3)
            all_pred_rgb_fg = outputs['fg']['image'].reshape(B, H, W, 3)
            all_pred_rgb_bg = outputs['bg']['image'].reshape(B, H, W, 3)
            im_list.extend([gt_mask, all_pred_mask, all_pred_rgb_fg, all_pred_rgb_bg])
            del gt_mask, all_pred_mask, all_pred_rgb_fg, all_pred_rgb_bg

        all_pred_rgb = torch.cat(im_list,dim=2)

        loss = torch.tensor([0])
        del data, rgbs, mask, rays_o, rays_d, outputs, gt_rgb, all_pred_depth, im_list
        return all_pred_rgb, None, None, loss

    def test_step(self, data, bg_color=None, perturb=False, if_gui=False):
        dir = torch.tensor(0)
        if if_gui:
            rays_o = data['rays_o']  # [B, N, 3]
            rays_d = data['rays_d']  # [B, N, 3]
            B, N = rays_o.shape[:2]
            H, W = data['H'], data['W']
            dir = data['dir']
        else:
            rgbs, mask, rays_o, rays_d, H, W, img_path = data
            B, N = rays_o.shape[:2]

            if not self.opt.if_data_cuda:
                rgbs = rgbs.to(self.device)
                mask = mask.to(self.device)
                rays_o = rays_o.to(self.device)
                rays_d = rays_d.to(self.device)

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None

        rays_o = rays_o.to(self.device)
        rays_d = rays_d.to(self.device)

        kwargs = vars(self.opt)
        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                    ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                    **kwargs)

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)

        if self.opt.pretrained:
            outputs_pt = self.model_pretrained.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                        ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                        **vars(self.opt))
            pred_rgb_pretrained = outputs_pt['image'].reshape(B, H, W, 3)

        if self.opt.train_conf and self.opt.render_all:
            im_list = []
            gt_mask = mask.reshape(B, H, W, 1).repeat(1,1,1,3)
            all_pred_mask = outputs['render_mask'].reshape(B, H, W, 1).repeat(1,1,1,3)
            all_pred_rgb_fg = outputs['fg']['image'].reshape(B, H, W, 3)
            all_pred_rgb_bg = outputs['bg']['image'].reshape(B, H, W, 3)
            im_list.extend([pred_rgb, all_pred_mask, all_pred_rgb_fg, all_pred_rgb_bg])
            pred_rgb = torch.cat(im_list,dim=2)

        return pred_rgb, pred_depth, dir

    def train(self, train_loader, valid_loader, max_epochs):


        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        self.save_checkpoint(full=True, best=False)


        for epoch in range(self.epoch + 1, max_epochs + 1):

            self.epoch = epoch

            self.train_one_epoch(train_loader)

            if self.epoch % self.eval_interval == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=True, best=False)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True, if_gui=False, split=None, is_train=False):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'
        if split is not None:
            name = name + '_' + split

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, name),exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, dir = self.test_step(data, if_gui=if_gui)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)

                    cv2.imwrite(os.path.join(save_path, name, f'{i:03d}.png'),
                            cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=30)#, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")
        return

    def train_step(self, data):
        if self.opt.pretrained:
            return self.train_step_editing(data)
        else:
            return self.train_step_pretrain(data)

    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if 'neus' in self.opt.backbone:
                        self.model.update_extra_state_sdf()
                    else:
                        self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_ws, loss, loss_dict = self.train_step(data)
                del pred_rgbs, pred_ws, data

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            del loss


            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = sum(loss_dict.values())
            total_loss += loss_val

            if self.local_rank == 0:

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    print_str = f""
                    for k, v in loss_dict.items():
                        print_str += "{} : {:.4f}, ".format(k, v)

                    pbar.set_description(
                        print_str+f" ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

            del loss_dict

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}. average_loss {average_loss}")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()
        # self.model.train()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            imgs = []
            for data in loader:
                self.local_step += 1
                if not self.opt.val_all_images:
                    if self.local_step == 5:
                        break
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_normal, loss = self.eval_step(data)
                    del data

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_normal_list = [torch.zeros_like(preds_normal).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_normal_list, preds_normal)
                    preds_normal = torch.cat(preds_normal_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in
                                        range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    if not self.opt.val_all_images:
                        imgs.append(preds)
                    else:
                        save_path = os.path.join(self.workspace, 'validation_all', f'{self.local_step}.png')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        pred = preds[0].detach().cpu().numpy()
                        pred = (pred * 255).astype(np.uint8)
                        cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        if not self.opt.val_all_images:
            save_path = os.path.join(self.workspace, 'validation', f'{name}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            preds = torch.cat(imgs,dim=1)
            pred = preds[0].detach().cpu().numpy()
            pred = (pred * 255).astype(np.uint8)

            cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
        del preds, pred, imgs
        torch.cuda.empty_cache()

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished. loss: ({total_loss / self.local_step:.4f})")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            if self.scaler is not None:
                state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt) and not old_ckpt.endswith('0.pth'):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")