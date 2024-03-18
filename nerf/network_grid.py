import torch


from .renderer import NeRFRenderer

from .encoding import get_encoder

from .provider_utils import trunc_exp
from .base import get_embedder
import tinycudann as tcnn
from torch import nn

class RGB_network(nn.Module):
    def __init__(self, input_ch_views, opt=None) -> None:
        super().__init__()
        self.opt = opt
        if self.opt.mask_no_dir:
            self.rgb_network = tcnn.Network(input_ch_views+64, 3,
                {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 1
                }
            )
            ndim = 1
            self.conf_network = tcnn.Network(64, ndim,
                {"otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 1
            })
        else:
            self.rgb_network = tcnn.Network(input_ch_views+64, 3,
                {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 1
                }
            )
            ndim = 1 if self.opt.keyword2 is None else 2
            self.conf_network = tcnn.Network(input_ch_views+64, ndim,
                {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 1
                }
            )
    def forward(self, x):
        if self.opt.mask_no_dir:
            dir_dim = x.shape[-1] - 64
            dir_fea, fea = x.split([dir_dim, 64], dim=-1)
            rgb = self.rgb_network(x)
            if self.opt.mask_no_dir_nodetach:
                rgb2 = self.conf_network(fea)
            else:
                rgb2 = self.conf_network(fea.detach())
        else:
            rgb = self.rgb_network(x)
            rgb2 = self.conf_network(x.detach())
        rgb = torch.cat([rgb,rgb2],dim=-1)
        return rgb

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 device=torch.device('cuda'),
                 num_layers=8,
                 hidden_dim=128,
                 skips=[4],
                 W_geo_feat=-1,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 variance_init=0.05,
                 speed_factor=10.0,
                 ):

        super().__init__(opt)

        # self.num_layers = num_layers
        # self.hidden_dim = hidden_dim
        
        log2_hashmap_size=21
        desired_resolution=8192

        # log2_hashmap_size=17
        # desired_resolution=4096/4

        self.pos_en, self.pos_en_dim = get_encoder('tiledgrid', input_dim=3, log2_hashmap_size=log2_hashmap_size,
                                                   desired_resolution=desired_resolution)

        self.network = tcnn.Network(self.pos_en_dim, 64,
            {"otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2
        })

        self.density_network = tcnn.Network(64, 1,
            {"otype": "FullyFusedMLP",
             "activation": "ReLU",
             "output_activation": "None",
             "n_neurons": 64,
             "n_hidden_layers": 1
         })

        self.dir_en, input_ch_views = get_embedder(4)

        if self.opt.train_conf:
            if self.opt.detach_mask_from_field or self.opt.mask_no_dir:
                self.rgb_network = RGB_network(input_ch_views, opt=opt)
            else:
                ndim = 1
                self.rgb_network = tcnn.Network(input_ch_views+64, 3+ndim,
                    {
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "Sigmoid",
                        "n_neurons": 64,
                        "n_hidden_layers": 1
                    }
                )
        else:
            self.rgb_network = tcnn.Network(input_ch_views+64, 3,
                {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 1
                }
            )

        self.bg_net = None

    def background(self, d):

        rgbs = torch.zeros(d.size(), dtype=d.dtype).to(d.device)

        return rgbs

    # add a density blob to the scene center
    def gaussian(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g


    def forward(self, x, d, l=None, ratio=1, shading='albedo'):

        x_en = self.pos_en(x, bound=self.opt.bound)

        fea = self.network(x_en)

        sigma = self.density_network(fea)
        sigma = trunc_exp(sigma.squeeze() + self.gaussian(x))

        view_en = self.dir_en(d)

        rgb_input = torch.concat([view_en, fea], dim=-1)

        radiances = self.rgb_network(rgb_input)

        del x, d, x_en, fea, view_en, rgb_input
        torch.cuda.empty_cache()

        return sigma, radiances, None


    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x_en = self.pos_en(x, bound=self.opt.bound)

        fea = self.network(x_en)

        sigma = self.density_network(fea)

        sigma = trunc_exp(sigma.squeeze() + self.gaussian(x))

        return {
            'sigma': sigma,
        }


    def get_params(self, lr):

        params = [
            {'params': self.pos_en.parameters(), 'lr': lr * 10},
            {'params': self.network.parameters(), 'lr': lr},
            {'params': self.density_network.parameters(), 'lr': lr},
            {'params': self.rgb_network.parameters(), 'lr': lr},
            # {'params': self.conf_network.parameters(), 'lr': lr},
        ]

        return params
