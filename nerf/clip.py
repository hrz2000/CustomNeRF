import torch
import torch.nn as nn
import torchvision.transforms as T
import clip

class CLIP(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        # clip.load("ViT-B/32", device=device)
        self.model, self.preprocess = clip.load("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/herunze/.cache/clip/ViT-B-32.pt", device=device)
        self.model.cuda().eval()
        self.transformCLIP = T.Compose([
            T.Resize(size=224, interpolation= T.InterpolationMode.BICUBIC, max_size=None, antialias=None),
            T.CenterCrop(size=(224,224)),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def get_text_embeds(self, text):
        text_tokens = clip.tokenize(text).cuda()
        with torch.no_grad():
            text_feature = self.model.encode_text(text_tokens).float()
        return text_feature
    
    def encode_img(self, img):
        img = self.transformCLIP(img)
        image_feature = self.model.encode_image(img).float()
        return image_feature
        
        
        


