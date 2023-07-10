import os
import sys
os.environ.update({'run_mode': 'train'})
sys.path.insert(0, os.getcwd())

import torch
import argparse
from logger import logger
from distributed import synchronize, get_rank, reduce_loss_dict, reduce_sum, get_world_size
from torch import nn, autograd, optim
from torchvision import transforms, utils
from torch.utils import data
from model import Generator, Discriminator, print_parameter, load_checkpoint
from dataset import MultiResolutionDataset
from torch.utils.tensorboard import SummaryWriter
from non_leaking import augment, AdaptiveAugment
try:
    import wandb
except ImportError:
    wandb = None
from tqdm import tqdm
import cv2
import math
import random
import numpy as np
from torch.nn import functional as F
from op import conv2d_gradfix
import torch.distributed as dist
from config import *


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device):
    # 循环处理
    loader = sample_data(loader)
    pbar = range(args.iter)
    if current_rank == 0:
        logger.info('create tqdm. current rank is ' + str(current_rank))
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # 初始化loss参数
    mean_path_length = 0
    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    # 分布式添加.module
    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            logger.info("Done!")
            break

        # 获取img
        real_img = next(loader)
        real_img = real_img.to(device)

        # 保存生成的图片
        if i == args.start_iter:
            real_img_np = np.uint8((real_img.detach().cpu().numpy() + 1) * 0.5 * 255)
            real_img_np = real_img_np[0]
            real_img_np = real_img_np.transpose(1, 2, 0)
            os.makedirs(f"{args.checkpoints_dir}/sample", exist_ok=True)
            cv2.imwrite(f"{args.checkpoints_dir}/sample/{str(i).zfill(6)}_real.png", real_img_np[..., ::-1])

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img = generator(noise, randomize_noise=debug_randomize_noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            d_reg = args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]
            d_reg.backward()
            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img = generator(noise, randomize_noise=debug_randomize_noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0
        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, randomize_noise=debug_randomize_noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if current_rank == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            writer.add_scalar('losses/g_loss', g_loss_val, global_step=i)
            writer.add_scalar('losses/d_loss', d_loss_val, global_step=i)
            writer.add_scalar('losses/r1', r1_val, global_step=i)
            writer.add_scalar('losses/path_loss', path_loss_val, global_step=i)
            writer.add_scalar('ada_aug_p', ada_aug_p, global_step=i)
            writer.add_scalar('r_t_stat', r_t_stat, global_step=i)
            writer.add_scalar('mean_path_length', mean_path_length, global_step=i)
            writer.add_scalar('fake_score', fake_score_val, global_step=i)
            writer.add_scalar('real_score', real_score_val, global_step=i)
            writer.add_scalar('path_length', path_length_val, global_step=i)

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample = g_ema([sample_z], randomize_noise=False)
                    os.makedirs(f"{args.checkpoints_dir}/sample_ema", exist_ok=True)
                    utils.save_image(
                        sample,
                        f"{args.checkpoints_dir}/sample_ema/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

                    sample_gen = generator([sample_z], randomize_noise=debug_randomize_noise)
                    os.makedirs(f"{args.checkpoints_dir}/sample_gen", exist_ok=True)
                    utils.save_image(
                        sample_gen,
                        f"{args.checkpoints_dir}/sample_gen/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

            if i % 1000 == 0:
                ckpt_dict = {
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                }
                os.makedirs(args.checkpoints_dir, exist_ok=True)

                torch.save(
                    ckpt_dict,
                    f"{args.checkpoints_dir}/{str(i).zfill(6)}.pt",
                )


def get_parser():
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures')
    parser.add_argument("--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument("--batch", type=int, default=16, help="batch sizes for each gpus")
    parser.add_argument("--n_sample", type=int, default=64, help="number of the samples generated during training")
    parser.add_argument("--size", type=int, default=256, help="image sizes for the model")
    parser.add_argument("--size_d", type=int, default=256, help="sizes for the discriminator")
    parser.add_argument("--dataset_imgsize", type=int, default=256, help="image sizes for the dataset")
    parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument("--path_regularize", type=float, default=2, help="weight of the path length regularization")
    parser.add_argument("--path_batch_shrink", type=int, default=2,
                        help="batch size reducing factor for the path length regularization (reduce memory consumption)")
    parser.add_argument("--d_reg_every", type=int, default=16, help="interval of the applying r1 regularization")
    parser.add_argument("--g_reg_every", type=int, default=4, help="interval of the applying path length regularization")
    parser.add_argument("--mixing", type=float, default=0.9, help="probability of latent code mixing")
    parser.add_argument("--ckpt", type=str, default=None, help="path to the checkpoints to resume training")
    parser.add_argument("--checkpoints_dir", type=str, default=None, help="path to the checkpoints to save")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("--channel_multiplier", type=int, default=2,
                        help="channel multiplier factor for the model. config-f = 2, else = 1")
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument("--wandb", action="store_true", help="use weights and biases logging")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    parser.add_argument("--augment", action="store_true", help="apply non leaking augmentation")
    parser.add_argument("--augment_p", type=float, default=0,
                        help="probability of applying augmentation. 0 = use adaptive augmentation")
    parser.add_argument("--ada_target", type=float, default=0.6,
                        help="target augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_length", type=int, default=500 * 1000,
                        help="target duraing to reach augmentation probability for adaptive augmentation")
    parser.add_argument("--ada_every", type=int, default=256,
                        help="probability update interval of the adaptive augmentation")
    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda"
    args = get_parser()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    logger.info('gpu num is ' + str(n_gpu))
    args.distributed = n_gpu > 1
    current_rank = 0

    # 分布式处理
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()   # 实现进程间的同步
        current_rank = get_rank()
    logger.info('current_rank is ' + str(current_rank))

    args.latent = 512
    args.start_iter = 0
    # 生成器、鉴别器、加权平均生成器 初始化
    generator = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    discriminator = Discriminator(args.size_d, channel_multiplier=args.channel_multiplier).to(device)
    g_ema = Generator(args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device)
    g_ema.eval()

    # 优化器初始化
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_optim = optim.Adam(generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),)
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),)

    # 网络加载权重参数
    if os.path.exists(args.ckpt):
        logger.info("load model: " + args.ckpt)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        load_checkpoint(generator, ckpt["g"], '', strict=False, device=device)
        load_checkpoint(discriminator, ckpt["d"], '', strict=False, device=device)
        load_checkpoint(g_ema, ckpt["g_ema"], '', strict=False, device=device)
        accumulate(generator, g_ema, 0)


    if current_rank == 0:
        argsDict = args.__dict__
        with open(f'{args.checkpoints_dir}/args_config.txt', 'w') as f:
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
                logger.info(eachArg + ' : ' + str(value))

    # 网络分布式加载
    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    # 网络参数打印
    if current_rank == 0:
        print_parameter(generator, 'generator')
        print_parameter(discriminator, 'discriminator')
        print_parameter(g_ema, 'g_ema')

    # 数据处理设置
    transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    # 数据预处理
    dataset = MultiResolutionDataset(args.path, transform, args.dataset_imgsize)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    if current_rank == 0:
        logger.info('Num Images: ' + str(len(dataset)))

    # 是否用wandb初始化
    if current_rank == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan2")

    # tensorboard写入
    if current_rank == 0:
        writer = SummaryWriter(os.path.join(args.checkpoints_dir, 'Logs'))

    print(args)
    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device)


