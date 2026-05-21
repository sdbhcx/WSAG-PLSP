import sys
import os
main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(main_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 指定你要使用的 GPU 编号

import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import time
import logging
import random
import numpy as np
import cv2

from models.full_model import ModelAGDsup as Model
from dataset.data import get_loader as get_loader
from models.metric import KLD, SIM, KL_loss, NSS
from models.encoder_clip import VisionTransformer as CLIP

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        INTERPOLATE_MODE = "nearest"
        torch.use_deterministic_algorithms(True)
    else:
        INTERPOLATE_MODE = "bilinear"
    return INTERPOLATE_MODE


def parse_args():
    parser = argparse.ArgumentParser(description='Finetuning on AGD20K')
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
    parser.add_argument('--seed', type=int, help='Seed', required=True)
    args = parser.parse_args()
    return args
 
  
def load_config(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def get_epoch_loss_config(base_loss_config, loss_schedule, epoch):
    if not loss_schedule:
        return base_loss_config
    effective = dict(base_loss_config)
    for stage in loss_schedule:
        start_epoch = stage.get("start_epoch", 0)
        end_epoch = stage.get("end_epoch", 10**9)
        if start_epoch <= epoch <= end_epoch:
            for key, value in stage.items():
                if key in ("start_epoch", "end_epoch"):
                    continue
                effective[key] = value
            break
    return effective


def batch_get_centers(pred_norm):
    B, H, W = pred_norm.shape
    device = pred_norm.device
    y = torch.arange(H, dtype=torch.float32, device=device) / H
    x = torch.arange(W, dtype=torch.float32, device=device) / W
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    
    centers = []
    for b in range(B):
        part_map = pred_norm[b] + 1e-3
        part_map_pdf = part_map / part_map.sum()
        y_c = (part_map_pdf * y_grid).sum()
        x_c = (part_map_pdf * x_grid).sum()
        centers.append([x_c, y_c])
    return centers


def get_variance(part_map_pdf, x_c, y_c):
    H, W = part_map_pdf.shape
    device = part_map_pdf.device
    y = torch.arange(H, dtype=torch.float32, device=device) / H
    x = torch.arange(W, dtype=torch.float32, device=device) / W
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    
    v_y = (part_map_pdf * ((y_grid - y_c) ** 2)).sum()
    v_x = (part_map_pdf * ((x_grid - x_c) ** 2)).sum()
    return v_x, v_y


def concentration_loss(pred):
    # b x h x w
    B, H, W = pred.shape
    tmp_max, tmp_min = pred.max(-1)[0].max(-1)[0].view(B, 1, 1), \
                       pred.min(-1)[0].min(-1)[0].view(B, 1, 1)

    pred_norm = ((pred - tmp_min) / (tmp_max - tmp_min + 1e-10))  # b x 28 x 28

    loss = 0
    epsilon = 1e-3
    centers_all = batch_get_centers(pred_norm)
    for b in range(B):
        centers = centers_all[b]
        # normalize part map as spatial pdf
        part_map = pred_norm[b, :, :] + epsilon  # prevent gradient explosion
        k = part_map.sum()
        part_map_pdf = part_map / k
        x_c, y_c = centers
        v_x, v_y = get_variance(part_map_pdf, x_c, y_c)
        loss_per_part = (v_x + v_y)
        loss = loss_per_part + loss
    loss = loss / B
    return loss


def main(config, seed):
    # set up logger
    os.makedirs(f"{config['work_dir']}", exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = f"{config['work_dir']}/{timestamp}.txt"
    logger = logging.getLogger("Train")
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    file_handler.setLevel("DEBUG")
    console_handler.setLevel("INFO")
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel("DEBUG")
    
    logger.info(config)

    if not os.path.exists(f"{config['work_dir']}/ckpt"):
        os.makedirs(f"{config['work_dir']}/ckpt")
    if not os.path.exists(f"{config['work_dir']}/img"):
        os.makedirs(f"{config['work_dir']}/img")
    
    args_text = yaml.safe_dump(config, default_flow_style=False)
    logger.debug(f'======================Config:======================\n{args_text}')
    logger.info(f'Set random seed to {seed}, deterministic: '
                f'{config["deterministic"]}')
    INTERPOLATE_MODE = set_random_seed(seed, deterministic=config["deterministic"])

    # build model and load checkpoint
    model_config = config['model']
    model = Model(**model_config)
    
    load_config = config["load"]
    all_ckpt, encoder_ckpt = load_config["all_ckpt"], load_config["encoder_ckpt"]
    all_ckpt, encoder_ckpt = load_config["all_ckpt"], load_config["encoder_ckpt"]
    encoder_type = str(config["model"]["encoder_type"]).lower()
    if all_ckpt:
        with open(all_ckpt, "rb") as f:
            state_dict = torch.load(f)["state_dict"]
        print("Loaded from ", all_ckpt)
        u, w = model.load_state_dict(state_dict, False)
        logger.debug(f'{u}, {w} are misaligned params in Model')
        for uu in u:
            logger.debug(uu)
        logger.debug("------------------------")
        for ww in w:
            logger.debug(ww)
    else:
        if encoder_type == "clip":
            if not encoder_ckpt:
                raise ValueError("`load.encoder_ckpt` is required when `model.encoder_type` is CLIP.")
            state_dict = torch.jit.load(encoder_ckpt, map_location='cpu').float().state_dict()
            ckpt_dict = {}
            for k, v in state_dict.items():
                if "visual" in k:
                    ckpt_dict[k.split('visual.')[1]] = v
            u, w = model.encoder.load_state_dict(ckpt_dict, False)
            logger.debug(f'{u}, {w} are misaligned params in CLIP Encoder')
        elif encoder_type in ("dino", "dinov2"):
            logger.info(
                "Using DINO/DINOv2 encoder: the main image encoder is initialized from torch.hub, "
                "and `load.encoder_ckpt` is reserved for the frozen CLIP feature branch."
            )
        else:
            raise ValueError(f"Unsupported encoder_type: {config['model']['encoder_type']}")
    
    num_parameters = sum([p.numel() for p in model.parameters()])
    logger.info(f'#Params: {num_parameters}')
    num_parameters = sum([p.numel() for p in model.encoder.parameters()])
    logger.info(f'#Encoder Params: {num_parameters}')
    num_parameters = sum([p.numel() for p in model.pred_decoder.parameters()])
    logger.info(f'#Final Decoder Params: {num_parameters}')    

    # define dataloader
    patch_grid_size = int(np.sqrt(model.encoder.num_patches))

    train_data_loader = get_loader(
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        split_file=config["split_type"],
        data_dir=config["data_dir"],
        train=True,
        exo_obj_file=os.path.join(config["data_dir"], config["split_type"], "trainset", "det_wholeobj_exo.pth"), 
        ego_obj_file=os.path.join(config["data_dir"], config["split_type"], "trainset", "det_wholeobj_ego.pth"), 
        num_exo=config["num_exo"],
        PL_mode=config["PL_mode"],
        aug4imgRatio=config["aug4imgRatio"]
    )
    val_data_loader = get_loader(
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        split_file=config["split_type"],
        data_dir=config["data_dir"],
        shuffle=False,
        train=False,
        exo_obj_file=None, 
        ego_obj_file=None, 
        no_pad_gt=True,
    )
    
    
    # build optimizer
    encoder_params_id = list(map(id, model.encoder.parameters()))
    other_params = filter(
        lambda p: (id(p) not in encoder_params_id) and p.requires_grad==True, 
        model.parameters(),
    )
    encoder_params = filter(
        lambda p: p.requires_grad==True, 
        model.encoder.parameters(),
    )
    optimizer_config = config['optimizer']
    lr = optimizer_config['lr']
    lr_encoder_coeff = optimizer_config["lr_encoder_coeff"]
    all_params = [{'params': other_params}, 
                  {'params': encoder_params, 'lr': lr*lr_encoder_coeff}]
    
    num_epochs = optimizer_config["num_epochs"]
    accum_iter = optimizer_config["accum_iter"]
    betas = optimizer_config["betas"]
    wd = optimizer_config["wd"]
    optimizer = optim.AdamW(params=all_params, lr=lr, betas=betas, weight_decay=wd)

    sche_type = optimizer_config["sche_type"]
    if sche_type == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=optimizer_config["lr_step"], gamma=optimizer_config["lr_gamma"])
    elif sche_type in ("cos", "cosine"):
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=optimizer_config["max_iter"], eta_min=1e-6)
    else:
        raise ValueError(f"Unsupported sche_type: {sche_type}")

    logger.debug("Model:")
    logger.debug(model)
    logger.debug("Optimizer:")
    logger.debug(optimizer)
    # logger.debug("Scheduler:")
    # logger.debug(lr_scheduler)
    # logger.debug("Parameters:")
    # for key, value in model.named_parameters():
    #     logger.debug(f'{key},  {value.requires_grad}')
    
    model = torch.nn.DataParallel(model).cuda()
    loss_config = config["loss"]
    loss_schedule = config.get("loss_schedule", [])
    total_iter = 0
    best_kld = 10000.
    best_sim = -10000.
    early_stop_cfg = config.get("early_stop", {})
    early_stop_enabled = early_stop_cfg.get("enabled", False)
    early_stop_patience = early_stop_cfg.get("patience", 8)
    early_stop_min_delta = early_stop_cfg.get("min_delta", 0.0)
    no_improve_epochs = 0
    
    frozen_feature = CLIP()
    state_dict = torch.jit.load(encoder_ckpt, map_location='cpu').float().state_dict()
    ckpt_dict = {}
    for k, v in state_dict.items():
        if "visual" in k:
            ckpt_dict[k.split('visual.')[1]] = v
    u, w = frozen_feature.load_state_dict(ckpt_dict, False)
    frozen_feature = frozen_feature.cuda().eval()
    
    
    for epoch in range(num_epochs):
        if total_iter >= optimizer_config["max_iter"]:
            break
        cur_loss_config = get_epoch_loss_config(loss_config, loss_schedule, epoch)
        if loss_schedule:
            logger.info(
                f"Epoch {epoch} loss coeffs: "
                f"kl={cur_loss_config.get('kl_loss_coeff')}, "
                f"sim={cur_loss_config.get('sim_loss_coeff')}, "
                f"exo_cls={cur_loss_config.get('exo_cls_coeff')}, "
                f"noun={cur_loss_config.get('noun_sim_coeff')}, "
                f"part={cur_loss_config.get('part_sim_coeff')}, "
                f"proto={cur_loss_config.get('proto_loss_coeff')}, "
                f"conc={cur_loss_config.get('conc_loss_coeff')}")
        model.train()
        all_loss = 0.0
        all_kl_loss = 0.
        all_sim_loss = 0.
        all_cls_loss = 0.0
        all_noun_sim_loss = 0.
        all_part_sim_loss = 0.
        all_proto_loss = 0.
        all_conc_loss = 0.

        all_num = 0
        acc_num = 0
        logger.info(f"============Training Epoch {epoch}============")
        for batch_data in tqdm(train_data_loader):
            if total_iter >= optimizer_config["max_iter"]:
                break
            if len(batch_data["input_image"]) == 1:
                continue # may cause cuda Bug
            
            if config["num_exo"] > 0:
                # Prepare masks for prototype contrast loss
                ego_part_mask = batch_data["gt_mask"].cuda()  # [B, 1, H, W]
                ego_obj_mask = batch_data["ego_objbox_mask"].cuda()  # Whole object mask from box
                exo_obj_mask_full = batch_data["exo_objbox_mask_patch"].cuda()  # [B*num_exo, G*G, 1]
                
                # Resize masks to match feature dimensions using max pooling to preserve small parts
                ego_part_mask_resized = F.adaptive_max_pool2d(
                    ego_part_mask, 
                    output_size=(patch_grid_size, patch_grid_size)
                ).squeeze(1)  # [B, G, G]
                ego_obj_mask_resized = F.adaptive_max_pool2d(
                    ego_obj_mask, 
                    output_size=(patch_grid_size, patch_grid_size)
                ).squeeze(1)  # [B, G, G]
                
                # Flatten masks to [B, G*G]
                ego_part_mask_flat = ego_part_mask_resized.reshape(ego_part_mask_resized.shape[0], -1)
                ego_obj_mask_flat = ego_obj_mask_resized.reshape(ego_obj_mask_resized.shape[0], -1)
                
                # Exo mask is already [B*num_exo, G*G, 1], just squeeze
                exo_obj_mask_full_flat = exo_obj_mask_full.squeeze(-1)  # [B*num_exo, G*G]
                
                aff_res, sim_loss, exo_cls_res, pred_noun, pred_part, proto_loss = model(
                    batch_data["input_image"], batch_data["sent_feats"], 
                    batch_data["exo_image"], 
                    batch_data["exo_objbox_mask_patch"], config["num_exo"],
                    ego_part_mask=ego_part_mask_flat,
                    ego_obj_mask=ego_obj_mask_flat,
                    exo_obj_mask_full=exo_obj_mask_full_flat
                )
            else:
                aff_res, pred_noun, pred_part = model(
                    batch_data["input_image"], batch_data["sent_feats"], 
                )
                proto_loss = torch.zeros(1,).cuda()
            
            noun_sim_loss = (1 - F.cosine_similarity(pred_noun, batch_data["noun_feats"].cuda(), dim=2)).mean()
            part_sim_loss = (1 - F.cosine_similarity(pred_part, batch_data["part_feats"].cuda(), dim=2)).mean()
            
            if not (aff_res.shape[2] == batch_data["gt_mask"].shape[2] and aff_res.shape[3] == batch_data["gt_mask"].shape[3]):
                r_pred = F.interpolate(
                    aff_res, 
                    size=batch_data["gt_mask"].shape[-2:],
                    mode=INTERPOLATE_MODE,
                )
            else:
                r_pred = aff_res
            kl_loss = KL_loss(r_pred, batch_data["gt_mask_prob"].cuda(), batch_data["valid_input"].cuda())
            
            vids = batch_data["vids"].long().cuda().unsqueeze(1).expand(-1, config["num_exo"],).reshape(-1)
            if config["num_exo"] > 0:
                exo_cls_loss = F.cross_entropy(exo_cls_res, vids, reduction='mean')
                sim_loss = sim_loss.mean()
            else:
                sim_loss = torch.zeros(1,).cuda()
                exo_cls_loss = torch.zeros(1,).cuda()
            
            r_prob = F.softmax(r_pred.reshape(len(r_pred), -1), dim=1)
            gt_prob = batch_data["gt_mask_prob"].reshape(len(r_pred), -1)
            
            # Handle proto_loss - may be None if masks are not available
            if proto_loss is not None:
                proto_loss = proto_loss.mean()
            else:
                proto_loss = torch.zeros(1,).cuda()
            
            conc_loss = concentration_loss(r_pred.squeeze(1))

            cur_loss = cur_loss_config["kl_loss_coeff"] * kl_loss + \
                cur_loss_config["sim_loss_coeff"] * sim_loss + \
                    cur_loss_config["exo_cls_coeff"] * exo_cls_loss + \
                        cur_loss_config["noun_sim_coeff"] * noun_sim_loss + \
                            cur_loss_config["part_sim_coeff"] * part_sim_loss + \
                            cur_loss_config.get("proto_loss_coeff", 0.1) * proto_loss + \
                                cur_loss_config.get("conc_loss_coeff", 0.01) * conc_loss
            all_num += 1
            all_loss += cur_loss.detach().item()
            all_kl_loss += kl_loss.detach().item()
            all_sim_loss += sim_loss.detach().item()
            all_cls_loss += exo_cls_loss.detach().item()
            all_noun_sim_loss += noun_sim_loss.detach().item()
            all_part_sim_loss += part_sim_loss.detach().item()
            all_proto_loss += proto_loss.detach().item()
            all_conc_loss += conc_loss.detach().item()
            
            cur_loss /= accum_iter
            cur_loss.backward()
            
            if all_num and all_num % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                acc_num += 1
                if sche_type in ("cos", "cosine"):
                    lr_scheduler.step()
                total_iter += 1
            
            
        if sche_type == "step":
            lr_scheduler.step()
        logger.info(
            f"Training loss: {all_loss / all_num}, KL loss: {all_kl_loss / all_num}, Sim loss: {all_sim_loss / all_num}, Exo CLS loss: {all_cls_loss / all_num}, \n"
            f"Noun sim loss: {all_noun_sim_loss / all_num}, Part sim loss: {all_part_sim_loss / all_num}, Proto loss: {all_proto_loss / all_num}, Conc loss: {all_conc_loss / all_num}")
        logger.info(
            f"learning rate:{optimizer.state_dict()['param_groups'][0]['lr']}\n")
        
        model.eval()
        vall_kld = 0.
        vall_sim = 0.
        vall_nss = 0.
        vall_num = 0
        vall_num_sum = 0
        
        vall_noun_sim = 0.
        vall_part_sim = 0.
        with torch.no_grad():
            for batch_data in tqdm(val_data_loader):
                aff_res, pred_noun, pred_part = model(
                    batch_data["input_image"], batch_data["sent_feats"],
                )
                pred = aff_res.detach()
                
                r_pred = F.interpolate(
                    pred, 
                    size=batch_data["gt_mask"].shape[-2:],
                    mode=INTERPOLATE_MODE,
                )
                
                noun_sim_loss = (1 - F.cosine_similarity(pred_noun, batch_data["noun_feats"].cuda(), dim=2)).mean()
                part_sim_loss = (1 - F.cosine_similarity(pred_part, batch_data["part_feats"].cuda(), dim=2)).mean()
                vall_noun_sim += noun_sim_loss.detach().item()
                vall_part_sim += part_sim_loss.detach().item()
                
                gt_prob = batch_data["gt_mask_prob"].cuda().reshape(len(pred), -1)
                r_prob = F.softmax(r_pred.reshape(len(pred), -1), dim=1)
                
                kld = KLD(r_prob, gt_prob) * len(pred)
                sim = SIM(r_prob, gt_prob) * len(pred)
                nss = NSS(r_prob, gt_prob) * len(pred)
                vall_kld += kld
                vall_sim += sim
                vall_nss += nss
                vall_num += 1
                vall_num_sum += len(pred)
                
        logger.info(
            f"Result on AGD: \nKLD={vall_kld/vall_num_sum}, SIM={vall_sim/vall_num_sum}, NSS={vall_nss/vall_num_sum}"
            f"\nnoun sim: {vall_noun_sim/vall_num}, part sim: {vall_part_sim/vall_num}")
        
        
        cur_kld = vall_kld / vall_num_sum
        cur_sim = vall_sim / vall_num_sum
        cur_nss = vall_nss / vall_num_sum
        if cur_kld < best_kld - early_stop_min_delta:
            best_kld = cur_kld
            no_improve_epochs = 0
            torch.save({'optimizer': optimizer.state_dict(),
                        'state_dict': model.module.state_dict()}, os.path.join(config['work_dir'], "ckpt", f'bestKLD.ckpt'))
            logger.info(f"New best KLD: {cur_kld}, {cur_sim}, {cur_nss}")
        else:
            no_improve_epochs += 1
            logger.info(
                f"No KLD improvement for {no_improve_epochs} epoch(s). "
                f"Best KLD: {best_kld}, Current KLD: {cur_kld}")
            if early_stop_enabled and no_improve_epochs >= early_stop_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best KLD: {best_kld}, patience: {early_stop_patience}, min_delta: {early_stop_min_delta}")
                break
        # if vall_sim/vall_num_sum > best_sim:
        #     best_sim = vall_sim/vall_num_sum
        #     torch.save({'optimizer': optimizer.state_dict(),
        #                 'state_dict': model.module.state_dict()}, os.path.join(config['work_dir'], "ckpt", f'bestSIM.ckpt'))
        
        

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(config, args.seed)
