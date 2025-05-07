# train_ddp.py
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, random_split
from tqdm import tqdm
import wandb

from grasp_inf import GraspInf, GraspInf_GelOnly
from gel_dataset import GelDataset  


import sys
sys.path.append('/n/fs/vl/check_overheat')
import check_overheat


def setup_distributed_backend(world_size, rank):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '12355')
    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total_samples = 0
    for batch in loader:
        # vis_b   = batch['cam_before'].to(device, non_blocking=True)
        # vis_a   = batch['cam_during'].to(device, non_blocking=True)
        delta_l = batch['delta_gelA'].to(device, non_blocking=True)
        delta_r = batch['delta_gelB'].to(device, non_blocking=True)
        labels  = batch['is_gripping'].float().to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(delta_l, delta_r)
        # outputs = model(vis_b, vis_a, delta_l, delta_r)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * labels.size(0)
        preds          = (outputs >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    # aggregate across ranks
    stats = torch.tensor([total_loss, total_correct, total_samples], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, correct_sum, samples_sum = stats.tolist()
    return loss_sum / samples_sum, correct_sum / samples_sum

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_samples = 0
    for batch in loader:
        # vis_b   = batch['cam_before'].to(device, non_blocking=True)
        # vis_a   = batch['cam_during'].to(device, non_blocking=True)
        delta_l = batch['delta_gelA'].to(device, non_blocking=True)
        delta_r = batch['delta_gelB'].to(device, non_blocking=True)
        labels  = batch['is_gripping'].float().to(device, non_blocking=True)

        outputs = model(delta_l, delta_r)
        # outputs = model(vis_b, vis_a, delta_l, delta_r)
        loss    = criterion(outputs, labels)

        total_loss    += loss.item() * labels.size(0)
        preds          = (outputs >= 0.5).float()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    stats = torch.tensor([total_loss, total_correct, total_samples], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, correct_sum, samples_sum = stats.tolist()
    return loss_sum / samples_sum, correct_sum / samples_sum

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",       type=str,   required=True)
    parser.add_argument("--ckpt-dir",       type=str,   default="./checkpoints")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--lr-step",        type=int,   default=10)
    parser.add_argument("--lr-gamma",       type=float, default=0.1)
    parser.add_argument("--hidden-dim",     type=int,   default=512)
    parser.add_argument("--val-split",      type=float, default=0.1)
    parser.add_argument("--num-workers",    type=int,   default=4)
    parser.add_argument("--wandb-project",  type=str,   default="grasp_inf")
    parser.add_argument("--wandb-run-name", type=str,   default=None)
    # these get set by torchrun
    parser.add_argument("--local_rank",     type=int,   default=int(os.environ.get("LOCAL_RANK", 0)))
    args = parser.parse_args()

    # distributed setup
    world_size = int(os.environ["WORLD_SIZE"])
    rank       = int(os.environ["RANK"])
    setup_distributed_backend(world_size, rank)
    device = torch.device(f"cuda:{args.local_rank}")

    # only rank 0 logs to WandB
    if rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # dataset + samplers
    full_ds = GelDataset(args.data_dir)
    val_sz  = int(len(full_ds) * args.val_split)
    train_sz = len(full_ds) - val_sz
    train_ds, val_ds = random_split(full_ds, [train_sz, val_sz])

    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   world_size, rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, sampler=val_sampler,
                              num_workers=args.num_workers, pin_memory=True)

    # model, DDP wrap
    # model = GraspInf(mlp_hidden_dim=args.hidden_dim, pretrained=True).to(device)
    model = GraspInf_GelOnly(mlp_hidden_dim=args.hidden_dim, pretrained=True).to(device)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # loss / optimizer / scheduler
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    best_val_acc = 0.0
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # training loop
    for epoch in range(1, args.epochs + 1):
        if epoch == args.epochs // 2:
            new_lr = args.lr * 0.1
            for pg in optimizer.param_groups:
                pg['lr'] = new_lr
            if rank == 0:
                print(f"---- dropping LR to {new_lr:.1e} at epoch {epoch} ----")
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = eval_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        if rank == 0:
            print(f"[{epoch}/{args.epochs}] "
                  f"Train: loss={train_loss:.4f}, acc={train_acc:.3f} | "
                  f" Val: loss={val_loss:.4f}, acc={val_acc:.3f}")

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": scheduler.get_last_lr()[0]
            })

            # save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.module.state_dict(),
                           os.path.join(args.ckpt_dir, "best_model.pth"))
                print(f"→ Saved best model (acc={val_acc:.3f})")

    if rank == 0:
        wandb.finish()
    cleanup()

if __name__ == "__main__":
    main()
