from email import parser
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from re import L
from tqdm import tqdm
import network
import utils
import random
import argparse
import numpy as np
from torch.utils import data
from datasets import Cityscapes, gta5, fire, sand
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pickle
from utils.utils import denormalize
from torchvision.utils import save_image
import torch.nn.functional as F
from losses import UncCELoss
import time

def get_argparser():
    parser = argparse.ArgumentParser()

    # Dataset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['cityscapes','ACDC','gta5'], help='Name of dataset')
    parser.add_argument("--ACDC_sub", type=str, default="night",
                        help = "specify which subset of ACDC  to use")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet_clip',
                        choices=available_models, help='model name')
    parser.add_argument("--BB", type = str, default = "RN50",
                        help = "backbone of the segmentation network")

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=200e3,
                        help="epoch number (default: 200k)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=768)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=22,
                        help="random seed")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--forward_pass",action='store_true',default=False,
                        help="forward pass to update BN statistics")
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--freeze_BB", action='store_true',default=False,
                        help="Freeze the backbone when training")
    parser.add_argument("--ckpts_path", type = str ,
                        help="path for checkpoints saving")
    parser.add_argument("--data_aug", action='store_true', default=False)
    #validation
    parser.add_argument("--val_results_dir", type=str,help="Folder name for validation results saving")
    parser.add_argument("--val_data_root", type=str,
                        default='./datasets/data/ACDC',
                        help="root path to the ACDC dataset used for validation during training")
    #Augmented features
    parser.add_argument("--train_aug",action='store_true',default=False,
                        help="train on augmented features using CLIP")
    parser.add_argument("--path_mu_sig", type=str)
    parser.add_argument("--pin_stats_dir", type=str,
                        default='./synthesis_value',
                        help="directory containing translation_{night,snow,rain,fog}.pkl PIN statistics from synthetic images")
    parser.add_argument("--mix", action='store_true',default=False,
                        help="mix statistics")
    
    parser.add_argument("--SIDA", action='store_true',default=False,
                        help="SIDA_ver")

 


    parser.add_argument("--domain_mix", action='store_true',default=False,
                        help="Domain mix for SIDA")
    
    parser.add_argument("--source_noise",type=float ,default=0.075,help="source feature noise level")
    parser.add_argument("--target_noise",type=float ,default=0.075,help="target feature noise level")
    parser.add_argument("--threshold",type=float ,default=0.1,help="threshold for SIDA")

    parser.add_argument("--entropy_base_noise", type=float, default=0.075,
                        help="base noise level for entropy-based adaptive noise")
    parser.add_argument("--entropy_scale_factor", type=float, default=2.0,
                        help="scale factor for entropy-based noise adjustment")

    parser.add_argument("--img_num", type=int, default=3)

    
    parser.add_argument("--sub_domain", type=str,help="select which sub-domain to use for SIDA")

   

    return parser


def compute_batch_entropy(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy

def get_dataset(dataset,data_root,crop_size,ACDC_sub="night",data_aug=True):
    """ Dataset And Augmentation
    """
    if dataset == 'cityscapes':
        if data_aug:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(crop_size, crop_size)),
                et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(crop_size, crop_size)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711]),
            ])

        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = Cityscapes(root=data_root,dataset=dataset,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=data_root,dataset=dataset,
                             split='val', transform=val_transform)

    if dataset == 'ACDC':
        train_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = Cityscapes(root=data_root,dataset=dataset,
                               split='train', transform=train_transform, ACDC_sub = ACDC_sub)
        val_dst = Cityscapes(root=data_root,dataset=dataset,
                             split='val', transform=val_transform, ACDC_sub = ACDC_sub)

    if dataset == "gta5":
        
        if data_aug:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(768, 768)),
                et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        else:
            train_transform = et.ExtCompose([
                et.ExtRandomCrop(size=(768, 768)),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711]),
            ])

        val_transform = et.ExtCompose([
            et.ExtCenterCrop(size=(1046, 1914)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        train_dst = gta5.GTA5DataSet(data_root, 'datasets/gta5_list/gtav_split_train.txt',transform=train_transform)
        val_dst = gta5.GTA5DataSet(data_root, 'datasets/gta5_list/gtav_split_val.txt',transform=val_transform)
    


    if dataset =='fire':
        fire_transform = et.ExtCompose([
            et.ExtResize((768,768)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        train_dst = fire.Fire(root=data_root,dataset=dataset, transform=fire_transform)

        val_dst = fire.Fire(root=data_root,dataset=dataset, transform=fire_transform)
        
    if dataset =='sand':
        sand_transform = et.ExtCompose([
            et.ExtResize((768,768)),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        train_dst = sand.Sand(root=data_root,dataset=dataset, transform=sand_transform)

        val_dst = sand.Sand(root=data_root,dataset=dataset, transform=sand_transform)
    return train_dst, val_dst

def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    if opts.save_val_results:
        if not os.path.exists(opts.val_results_dir):
            os.mkdir(opts.val_results_dir)
        img_id = 0

    with torch.no_grad():

        for i, (im_id, tg_id, images, labels) in tqdm(enumerate(loader), total=len(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            outputs,features = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
           
            metrics.update(targets, preds)
            
            if opts.save_val_results:
                for j in range(len(images)):

                    target = targets[j]
                    pred = preds[j]

                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(target).save(opts.val_results_dir+'/%d_target.png' % img_id)
                    Image.fromarray(pred).save(opts.val_results_dir+'/%d_pred.png' % img_id)

                    images[j] = denormalize(images[j],mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
                    save_image(images[j],opts.val_results_dir+'/%d_image.png' % img_id)

                    fig = plt.figure()
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    #plt.savefig(opts.val_results_dir+'/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score


def main():
    start=time.time()
    opts = get_argparser().parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    # INIT
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
  
    train_dst,val_dst = get_dataset(opts.dataset,opts.data_root,opts.crop_size,opts.ACDC_sub,
                                    data_aug=opts.data_aug)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
    drop_last=True)  # drop_last=True to ignore single-image batches.

    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)
    
    print("Dataset: %s, Train set: %d, Val set: %d" %
        (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=19, BB= opts.BB,replace_stride_with_dilation=[False,False,True])
    model.backbone.attnpool = nn.Identity()

    #fix the backbone
    if opts.freeze_BB:
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()

    # Set up metrics
    metrics = StreamSegMetrics(19)

    # Set up optimizer
    if opts.freeze_BB:
        optimizer = torch.optim.SGD(params=[
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    else:
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': 0.001 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.9)

    # Set up criterion
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    if not opts.test_only:
        utils.mkdir(opts.ckpts_path)
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model.to(device)
    
    # ==========   Train Loop   ==========#

    if opts.test_only:
       
        model.eval()

        val_score = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics)

        print(metrics.to_str(val_score))
        print(val_score["Mean IoU"])
        print(val_score["Class IoU"])
        return

    interval_loss = 0

    if opts.train_aug and not opts.SIDA:
        text_feat_dir=os.path.join(opts.path_mu_sig,opts.ACDC_sub)
        files = [f for f in os.listdir(text_feat_dir+'/')]
    
    relu = nn.ReLU(inplace=True)
    seg_loss = UncCELoss(ignore=255).to(device)

    # Translation synth (PIN statistics derived from synthetic images)
    with open(os.path.join(opts.pin_stats_dir, 'translation_night.pkl'), 'rb') as f:
        night_dict = pickle.load(f)
    with open(os.path.join(opts.pin_stats_dir, 'translation_snow.pkl'), 'rb') as f:
        snow_dict = pickle.load(f)
    with open(os.path.join(opts.pin_stats_dir, 'translation_rain.pkl'), 'rb') as f:
        rain_dict = pickle.load(f)
    with open(os.path.join(opts.pin_stats_dir, 'translation_fog.pkl'), 'rb') as f:
        fog_dict = pickle.load(f)
    
    if opts.sub_domain == 'night':
        subdomain_dict = night_dict
    elif opts.sub_domain == 'snow':
        subdomain_dict = snow_dict
    elif opts.sub_domain == 'rain':
        subdomain_dict = rain_dict
    else:
        subdomain_dict = fog_dict
    
    dict_list=[night_dict,snow_dict,rain_dict,fog_dict]
    

    while True:  
    # =====  Train  =====
    
        if opts.freeze_BB:
            model.classifier.train()
        else:
            model.train()

        cur_epochs += 1

        for (im_id, tg_id, images, labels) in train_loader:
            
            cur_itrs += 1
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            
            optimizer.zero_grad()
            if opts.train_aug and not opts.SIDA:
                mu_t_f1 = torch.zeros([opts.batch_size,256,1,1])
                std_t_f1 = torch.zeros([opts.batch_size,256,1,1])
        
                for k in range(opts.batch_size):
                    
                    with open(text_feat_dir+'/'+random.choice(files), 'rb') as f:
                        loaded_dict = pickle.load(f)
                        mu_t_f1[k] = loaded_dict['mu_f1']
                        std_t_f1[k] = loaded_dict['std_f1']

                outputs,features = model(images,mu_t_f1.to(device),std_t_f1.to(device),
                                    transfer=True,mix=opts.mix,activation=relu)
            if opts.SIDA:
                mu_t_f1 = torch.zeros([opts.batch_size,256,1,1])
                std_t_f1 = torch.zeros([opts.batch_size,256,1,1])

                mu_t_f1_2 = torch.zeros([opts.batch_size,256,1,1])
                std_t_f1_2 = torch.zeros([opts.batch_size,256,1,1])

                for k in range(opts.batch_size):
                    with open(os.path.join(opts.pin_stats_dir, f'translation_{opts.ACDC_sub}.pkl'), 'rb') as f:
                        loaded_dict = pickle.load(f)

                        idx1 = random.randint(0,opts.img_num-1)
                        idx2 = random.randint(0,opts.img_num-1)
                        

                        mu_t_f1[k] = loaded_dict['mu_f1'][idx1]
                        std_t_f1[k] = loaded_dict['std_f1'][idx1]

                        

                        if opts.domain_mix:
                            if opts.ACDC_sub=='night':
                                mu_t_f1_2[k] = rain_dict['mu_f1'][idx2]
                                std_t_f1_2[k] = rain_dict['std_f1'][idx2]
                            elif opts.ACDC_sub=='snow':
                                mu_t_f1_2[k] = fog_dict['mu_f1'][idx2]
                                std_t_f1_2[k] = fog_dict['std_f1'][idx2]
                            elif opts.ACDC_sub=='rain':
                                mu_t_f1_2[k] = fog_dict['mu_f1'][idx2]
                                std_t_f1_2[k] = fog_dict['std_f1'][idx2]
                            else:
                                mu_t_f1_2[k] = rain_dict['mu_f1'][idx2]
                                std_t_f1_2[k] = rain_dict['std_f1'][idx2]
                        else:
                            mu_t_f1_2[k] = subdomain_dict['mu_f1'][idx2]
                            std_t_f1_2[k] = subdomain_dict['std_f1'][idx2]

            
                # Fixed noise
                current_source_noise = opts.source_noise
                current_target_noise = opts.target_noise

                outputs,features,original_outputs,original_features = model.forward_SIDA(images,current_source_noise,current_target_noise,
                                    mu_t_f1.to(device),std_t_f1.to(device),
                                    mu_t_f1_2.to(device),std_t_f1_2.to(device),
                                    transfer=True,mix=opts.mix,activation=relu,
                                    target_feat_noise="True",source_feat_noise="True",
                                    patch_ST="True")
           
         
            else:
                outputs,features = model(images)
            
            # Entropy Diff loss
            if opts.SIDA:
                loss=0.0

                valid_samples = 0
                batch_entropy = compute_batch_entropy(outputs)
                threshold = opts.threshold

                for i in range(opts.batch_size):
                    sample_entropy = batch_entropy[i]
                    if sample_entropy.mean() >= threshold:
                        loss += seg_loss(outputs[i].unsqueeze(0), labels[i], sample_entropy)
                        valid_samples += 1
                    else:
                        loss+=criterion(outputs[i].unsqueeze(0), labels[i].unsqueeze(0))
                        valid_samples += 1
                
                if valid_samples > 0:
                    loss = loss / valid_samples
                else:
                    loss = torch.tensor(0.0, device=device)
            else:
                # naive CE
                loss = criterion(outputs, labels)


            loss.backward()
            optimizer.step()
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            
            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                    (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                tmp=0
                print("validation...")
                model.eval()
                
                # ACDC val
                _,val_dst = get_dataset("ACDC", opts.val_data_root, opts.crop_size, opts.ACDC_sub, data_aug=opts.data_aug)
                
                val_loader = data.DataLoader(val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=4)

                
                val_score = validate(opts=opts, model=model, loader=val_loader,device=device, metrics=metrics)


                print(metrics.to_str(val_score))

                # update best score and save the best checkpoint
                if val_score["Mean IoU"] > best_score:
                    best_score = val_score["Mean IoU"]
                    save_ckpt(os.path.join(opts.ckpts_path,
                              "best_%s.pth" % (opts.ACDC_sub)))

                if opts.freeze_BB:
                    model.classifier.train()
                else:
                    model.train()
            
           
            if cur_itrs >= 2000:

                print(best_score)

                

                return



            

if __name__ == '__main__':
    main()