import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pprint import pprint

class PPG_GAN_1D(nn.Module):
    para = {
        # Generator 
        "z_dim": 1200,
        "gen_input_channels": 1,
        "gen_stride": 1,
        "gen_activation": nn.ReLU(inplace=True),
        "gen_filters": {
            "short":  (80, 5, 2),    
            "middle": (40, 21, 10),
            "long":   (20, 61, 30),
        },
        "gen_total_filter": {"total": (50, 15, 7)},
        "gen_final_convs": [
            # (out_channels, kernel_size, padding, activation_fn)
            (50, 4, 1, nn.Tanh),
            (1, 2, 1, nn.Tanh),
        ],
        
        # Discriminator 
        "dis_input_length": 1200,
        "dis_activation": nn.LeakyReLU(0.2, inplace=True),
        "dis_channels": [1, 64, 128, 256, 512, 1], #[in, h1, h2, h3, h4, out]
        "dis_kernels": [4, 4, 4, 4, 8],
        "dis_strides": [2, 2, 2, 2, 1],
        "dis_paddings": [1, 1, 1, 1, 0],
        "dis_output_classes": 1
    }

    # visualizer
    vis_gen = []
    vis_dis = []




    def __init__(self, config):
        super(PPG_GAN_1D, self).__init__()

        self.config = {}
        for key in self.para:
            self.config[key] = config.get(key, self.para[key])

        # save dim
        self.z_dim = self.config["z_dim"]
        self.dis_input_length = self.config["dis_input_length"]
        
        # model
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

    # build Generator
    

    def _build_generator(self):
        cfg = self.config
        stride = cfg["gen_stride"]
        generator_module = nn.Module()
        
        # s_cfg (80, 5, 2)
        s_cfg = cfg["gen_filters"]["short"] 
        # index: [0]=channels, [1]=kernel, [2]=padding
        generator_module.sFilter = self._gen_conv_block(1, s_cfg[0], s_cfg[1], stride, s_cfg[2])
        
        # middle
        m_cfg = cfg["gen_filters"]["middle"]
        generator_module.mFilter = self._gen_conv_block(1, m_cfg[0], m_cfg[1], stride, m_cfg[2])

        # long
        l_cfg = cfg["gen_filters"]["long"]
        generator_module.lFilter = self._gen_conv_block(1, l_cfg[0], l_cfg[1], stride, l_cfg[2])

        # total
        total_in_channels = s_cfg[0] + m_cfg[0] + l_cfg[0]
        t_cfg = cfg["gen_total_filter"]["total"] # t_cfg là (50, 15, 7)
        generator_module.tFilter = self._gen_conv_block(total_in_channels, t_cfg[0], t_cfg[1], stride, t_cfg[2])

        final_layers = []
        current_channels = t_cfg[0] 
        
        for (out_c, nk, padding, act_fn) in cfg["gen_final_convs"]:
            final_layers.append(nn.Conv1d(current_channels, out_c, nk, padding=padding, bias=False))
            if act_fn:
                final_layers.append(act_fn())
            current_channels = out_c 
            
        generator_module.final_convs = nn.Sequential(*final_layers)
        
        return generator_module

    def _gen_conv_block(self, in_c, out_c, nk, stride, padding, activation=nn.ReLU(inplace=True)):
      
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, nk, stride, padding=padding, bias=False),
            nn.BatchNorm1d(out_c),
            activation
        )



    # Discriminator 
    def _build_discriminator(self):
        cfg = self.config
        dis_module = nn.Module()
        channels, kernels, strides, paddings, act = cfg["dis_channels"], cfg["dis_kernels"], cfg["dis_strides"], cfg["dis_paddings"], cfg["dis_activation"]
        
        conv_layers = []
        
        for i in range(len(channels) - 1):
            conv_layers.append(
                nn.Conv1d(
                    channels[i], 
                    channels[i+1], 
                    kernel_size=kernels[i], 
                    stride=strides[i], 
                    padding=paddings[i], 
                    bias=False
                )
            )
            
            if i < len(channels) - 2: 
                conv_layers.append(nn.BatchNorm1d(channels[i+1]))
                conv_layers.append(act)
            else:
                conv_layers.append(act) # (512 -> 1) LeakyReLU
                
        dis_module.model = nn.Sequential(*conv_layers)
        
        # FC head
        dis_module.pool = nn.AdaptiveAvgPool1d(1)
        fc_in_features = channels[-1] # Kênh đầu ra của lớp conv cuối
        
        dis_module.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in_features, cfg["dis_output_classes"]),
       
        )
        
        return dis_module



    # Forward 
    def forward_generator(self, z):
        self.vis_gen.clear() 
        b_size = z.size(0)
        
        if z.dim() == 2:
            z = z.view(b_size, self.config["gen_input_channels"], -1) 
        if z.size(2) != self.z_dim:
            z = F.interpolate(z, size=self.z_dim, mode='linear', align_corners=False)
        
        x = z 
        self.vis_gen.append(x)
        
        xs = self.generator.sFilter(x)
        xm = self.generator.mFilter(x)
        xl = self.generator.lFilter(x)
        self.vis_gen.append(xs); self.vis_gen.append(xm); self.vis_gen.append(xl)

        x = torch.cat((xs, xm, xl), 1)
        self.vis_gen.append(x)
        
        x = self.generator.tFilter(x)
        self.vis_gen.append(x)
        
        x = self.generator.final_convs(x)
        self.vis_gen.append(x)

        # Min-Max
        x_flat = x.view(b_size, -1)
        min_v = torch.min(x_flat, dim=1, keepdim=True)[0].view(b_size, 1, 1)
        max_v = torch.max(x_flat, dim=1, keepdim=True)[0].view(b_size, 1, 1)
        range_v = max_v - min_v
        range_v = torch.where(range_v == 0, torch.tensor(1e-6, device=x.device), range_v)
        
        normalised_x = (x - min_v) / range_v
        self.vis_gen.append(normalised_x)
        
        return normalised_x
    


    def forward_discriminator(self, x):
        self.vis_dis.clear() 
        
        if x.size(2) != self.dis_input_length:
            x = F.interpolate(x, size=self.dis_input_length, mode='linear', align_corners=False)
        
        self.vis_dis.append(x)
        
        x = self.discriminator.model(x)
        self.vis_dis.append(x)
        
        x = self.discriminator.pool(x)
        self.vis_dis.append(x)
        
        x = self.discriminator.fc(x)
        self.vis_dis.append(x)
        
        return x
    
    
    # helper
    def visualizer(self, part='all'):
        if part in ['all', 'gen']:
            print("\n Generator Feature Map Shapes")
            for item in self.vis_gen:
                print(item.shape)
        if part in ['all', 'dis']:
            print("\n Discriminator Feature Map Shapes")
            for item in self.vis_dis:
                print(item.shape)





    # print output           
    def print_structure(self, part='all'):
      
        def clean_config(config_dict):
            cleaned = {}
            for k, v in config_dict.items():
                if isinstance(v, nn.Module):
                    cleaned[k] = v.__class__.__name__
                elif isinstance(v, type) and issubclass(v, nn.Module):
                    cleaned[k] = v.__name__
                elif isinstance(v, list):
                    cleaned[k] = [
                        tuple(item.__class__.__name__ if isinstance(item, nn.Module) else (item.__name__ if (isinstance(item, type) and issubclass(item, nn.Module)) else item) for item in sub_tuple)
                        if isinstance(sub_tuple, tuple) else sub_tuple
                        for sub_tuple in v
                    ]
                elif isinstance(v, dict):
                     cleaned[k] = clean_config(v)
                else:
                    cleaned[k] = v
            return cleaned

        if part in ['all', 'gen']:
            print("\n--- Generator Config ---")
            gen_config = {
                "z_dim": self.config["z_dim"],
                "gen_input_channels": self.config["gen_input_channels"],
                "gen_activation": self.config["gen_activation"],
                "gen_filters": self.config["gen_filters"],
                "gen_total_filter": self.config["gen_total_filter"],
                "gen_final_convs": self.config["gen_final_convs"]
            }

            pprint(clean_config(gen_config), width=120, sort_dicts=False)
            
        if part in ['all', 'dis']:
            print("\n--Discriminator Config ---")
            dis_config = {
                "dis_input_length": self.config["dis_input_length"],
                "dis_activation": self.config["dis_activation"],
                "dis_channels": self.config["dis_channels"],
                "dis_kernels": self.config["dis_kernels"],
                "dis_strides": self.config["dis_strides"],
                "dis_paddings": self.config["dis_paddings"],
                "dis_output_classes": self.config["dis_output_classes"]
            }
       
            pprint(clean_config(dis_config), width=120, sort_dicts=False)




# Test
if __name__ == "__main__":

    para = {
         "z_dim": 1200,
         "dis_input_length": 1200,
    }

    batch_size = 4
    z_dim = para.get("z_dim", 1200) 
    input_z = torch.randn((batch_size, z_dim))

    model = PPG_GAN_1D(config=para)

    print("-"*10 + " MODEL " + "-"*10 )
    model.print_structure('all') 

    print("\n" + "-"*10  + " FORWARD " + "-"*10 )
    output_g = model.forward_generator(input_z)
    output_d = model.forward_discriminator(output_g)

    print(f"Input Z shape:     {input_z.shape}")
    print(f"Output G (Signal): {output_g.shape}")
    print(f"Output D (Decision): {output_d.shape}")

    print("\n" + "-"*10  + " VISUALIZER (G) " + "-"*10 )
    model.visualizer('gen')
    
    print("\n" + "-"*10  + " VISUALIZER (D) " + "-"*10 )
    model.visualizer('dis')