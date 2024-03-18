from transformers import logging
from torch.optim.adam import Adam
logging.set_verbosity_error()
import numpy as np
from diffusers import DiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from torch.cuda.amp import custom_bwd, custom_fwd

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', opt=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.opt = opt

        print(f'[INFO] loading stable diffusion...')
        
        if self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            model_key = self.sd_version
            print(f"load model: {model_key}")

        self.pipe = DiffusionPipeline.from_pretrained(model_key).to(self.device)

        if self.opt.use_cd is not None and not self.opt.test:
            model_id = self.opt.use_cd
            self.pipe.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")
            self.pipe.load_textual_inversion(model_id, weight_name="<new1>.bin")

        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * self.opt.max_ratio)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        self.null_dict = dict()

        print(f'[INFO] loaded stable diffusion!')
        self.system = None

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings
    

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def get_params(self, lr):
        params = [
        ]
        return params

    def set_system(self, system):
        self.system = system

    def train_step(self, latents, text_embeddings, mask=None, img_path=None, tuning=False, gt_rgb=None, t_val=None, system=None, is_all=False, camera=None, tuning_cls=False, t_ratio=1):#n,12
        # guidance_scale = self.opt.guidance_scale
        guidance_scale = self.opt.cfg


        if self.opt.stage_time:
            cur_iters = system.global_step
            total_iters = self.opt.iters
            max_step = self.max_step
            min_step = self.min_step
            if cur_iters > total_iters/2:
                max_step = int(max_step * 0.5)
            # t = max(int((1 - cur_iters / total_iters) * max_step), min_step)
            # t = torch.tensor([t], dtype=torch.long, device=self.device)
            t = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        t = (t * t_ratio).to(torch.long)

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latent_model_input = torch.cat([latents_noisy] * 2)
            bs = len(latent_model_input)
            class_labels = None
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, class_labels=class_labels).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise) * self.opt.lambda_sd ############### 0.01

        grad = torch.nan_to_num(grad)

        target = (latents - grad).detach()#ok
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, target, reduction="sum")
        loss_dict = dict(loss_sds=loss.item())

        return loss, loss_dict