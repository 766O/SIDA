import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.stats import *


def perturb_style_stats(mu_mix, std_mix, noise_shape,
                        noise_level_mu, noise_level_std):
    """Apply Gaussian perturbation to the mixed style statistics.

    mu += N(0, sigma^2), std += N(0, sigma^2). std is kept > 0 with a small clamp.
    """
    device = mu_mix.device

    noise_mu = torch.normal(mean=torch.zeros(noise_shape), std=noise_level_mu).to(device)
    noise_std = torch.normal(mean=torch.zeros(noise_shape), std=noise_level_std).to(device)
    mu_out = mu_mix + noise_mu
    std_out = std_mix + noise_std

    std_out = std_out.clamp_min(1e-6)
    return mu_out, std_out


class _Segmentation(nn.Module):
    def __init__(self, backbone,classifier):
        super(_Segmentation, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
      
    def forward(self, x,mu_t_f1=0, std_t_f1=0, transfer=False,mix=False,activation=None):
        input_shape = x.shape[-2:]
        features = {}
        features['low_level'] = self.backbone(x,trunc1=False,trunc2=False,
           trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)

        if transfer:

            mean, std = calc_mean_std(features['low_level'])
            self.size = features['low_level'].size()

            features_low_norm = (features['low_level'] - mean.expand(
                self.size)) / std.expand(self.size)
            
            if mix:
                s = torch.rand((mean.shape[0],mean.shape[1])).to('cuda').unsqueeze(-1).unsqueeze(-1)
                mu_mix = s * mean + (1-s) * mu_t_f1
                std_mix = s * std + (1-s) * std_t_f1
                features['low_level'] = (std_mix.expand(self.size) * features_low_norm + mu_mix.expand(self.size))
            else:
                features['low_level'] = (std_t_f1.expand(self.size) * features_low_norm + mu_t_f1.expand(self.size))
            features['low_level'] = activation(features['low_level'])
                

        features['out'] = self.backbone(features['low_level'],trunc1=True,trunc2=False,
            trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=True)

        x = self.classifier(features)
        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return output, features



    def forward_SIDA(self, x, source_noise=0.075, target_noise=0.075, \
                     mu_t_f1=0, std_t_f1=0, mu_t_f1_2=0, std_t_f1_2=0, \
                     transfer=False,mix=False,activation=None, \
                     target_feat_noise=True,source_feat_noise=True, patch_ST=True):


        input_shape = x.shape[-2:]
        original_features ={}
        features = {}
        features['low_level'] = self.backbone(x,trunc1=False,trunc2=False,
           trunc3=False,trunc4=False,get1=True,get2=False,get3=False,get4=False)
        original_features['low_level']=features['low_level']

        stylized_feature_mu=[]
        stylized_feature_std=[]

        if transfer:

            mean, std = calc_mean_std(features['low_level'])
            self.size = features['low_level'].size()

            # Souerce feature perturbation
            if source_feat_noise:
                feat_mean = features['low_level'].mean((2, 3), keepdim=True)
                ones_mat = torch.ones_like(feat_mean)
                alpha = torch.normal(ones_mat, source_noise * ones_mat)
                beta = torch.normal(ones_mat, source_noise * ones_mat)
                output = alpha * features['low_level'] - alpha * feat_mean + beta * feat_mean
                features['low_level'] = output
            

            features_low_norm = (features['low_level'] - mean.expand(
                self.size)) / std.expand(self.size)
            
            random_number = np.random.rand()
            patch_random_number=np.random.rand()

            if patch_ST == 'True':
                x_denormed_all=[]
                x_cat_w=[]

                H,W=192,192

                patch_num=3
                patch_h=H//patch_num

                for i in range(patch_num):
                    for j in range(patch_num):
                        
                        x_patch = features['low_level'][:, :, int(i*(H/patch_num)):int((i+1)*(H/patch_num)), int(j*(W/patch_num)):int((j+1)*(W/patch_num))]
                        
                        mu = x_patch.mean(dim=[2, 3], keepdim=True)
                        sig = (x_patch.var(dim=[2, 3], keepdim=True) + 1e-6).sqrt()
                        
                        mu, sig = mu.detach(), sig.detach()
                        
                        x_patch_normed = (x_patch - mu) / sig # x-mu / sig
                        noise_level_mu = target_noise
                        noise_level = target_noise

                        noise_shape = (self.size[0], 256, 1, 1) # final_ver

                        # mixing ration
                        s = torch.rand((mean.shape[0],mean.shape[1])).to('cuda').unsqueeze(-1).unsqueeze(-1)

                        mu_mix = s * mu_t_f1_2 + (1-s) * mu_t_f1
                        std_mix = s * std_t_f1_2 + (1-s) * std_t_f1

                        # Gaussian style perturbation
                        mu_mix, std_mix = perturb_style_stats(
                            mu_mix, std_mix, noise_shape,
                            noise_level_mu=noise_level_mu,
                            noise_level_std=noise_level,
                        )
                        
                        if patch_random_number > 0.25:
                            x_denormed = (std_mix.expand(self.size[0],256,patch_h,patch_h) * x_patch_normed + mu_mix.expand(self.size[0],256,patch_h,patch_h)) 
                        else:
                            x_denormed=x_patch
                        
                        x_denormed_all.append(x_denormed) 

                # stylized patch concat
                for k in range(patch_num):
                    x_cat_w.append(torch.cat(x_denormed_all[patch_num*k:patch_num*(k+1)], dim=3))

                # final stylized feature
                x_stylized = torch.cat(x_cat_w, dim=2) 

                features['low_level'] = activation(x_stylized.clone()) # torch.Size([8, 256, 192, 192]), lowlevel -> layer 1 통과이후 feat
            
            features['low_level'] = activation(features['low_level'].clone()) 
            
            stylized_l1_mu,stylized_l1_std=calc_mean_std(features['low_level'])



                
        # stylized final feature 
        features['out'] = self.backbone(features['low_level'],trunc1=True,trunc2=False,
            trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=True)
        # original final feature
        original_features['out'] = self.backbone(original_features['low_level'],trunc1=True,trunc2=False,
            trunc3=False,trunc4=False,get1=False,get2=False,get3=False,get4=True)


        x = self.classifier(features)
        original_x=self.classifier(original_features)


        output = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        original_output = F.interpolate(original_x,size=input_shape,mode='bilinear',align_corners=False)


        return output, features, original_output,original_features
    

    