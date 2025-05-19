import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler
from meter import Meters
from logger import Logger
from data import JSONLDataset
from model import Model, collate_fn
from functions import prepare_optimizer
import os
import opts
import torch.distributed as dist
from utils import is_main_process, logging_info

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    # --- Distributed Setup ---
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", dist.get_rank())
    else:
        device = torch.device(args.device)
    args.device = device
    logger = Logger(args.save_dir)
    meters = Meters()
    # --- AMP Setup ---
    use_amp = args.mixed_precision == 'fp16' and device.type == 'cuda'
    scaler = GradScaler(device.type, enabled=use_amp)
    
    logging_info(f"Using Automatic Mixed Precision: {use_amp}")
    logging_info(f"Distributed training: {distributed}")
    # --- End AMP Setup ---

    model = Model(args).to(device)
    original_model = model
    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=args.find_unused_parameters)

    train_dataset = JSONLDataset(args.data, args.max_seq_length)
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    scheduler, optimizer, optimizing_parameters = prepare_optimizer(args, model)

    logging_info("Starting Training...")

    model.train()
    iteration = args.start_iterations

    while iteration < args.iterations:
        if distributed:
            train_sampler.set_epoch(iteration)

        for batch in train_dataloader:
            optimizer.zero_grad()

            with autocast(device.type, enabled=use_amp):
                losses = model(batch)
                loss = sum(losses.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            meters.update(1, **losses)
            if is_main_process():
                if iteration % 20 == 0:
                    logging_info(f"Iter:{iteration:4d}| {meters}| lr: {optimizer.param_groups[0]['lr']:.6f}")
                logger.scalar_summary('train/lr', optimizer.param_groups[0]['lr'], iteration)
                for name, meter in meters:
                    logger.scalar_summary(f'train/{name}', meter.val, iteration)
                if iteration % 5000 == 0 and iteration > 0:
                    save_path = os.path.join(args.save_dir, f"{iteration}.pt")
                    logging_info(f"Saving checkpoint to {save_path}")
                    os.makedirs(args.save_dir, exist_ok=True)
                    torch.save({
                        'iteration': iteration,
                        'state_dict': original_model.state_dict(),
                        'args': args
                        }, save_path)

                    # Evaluation logic
                    model.eval()
                    with torch.no_grad():
                        with autocast(device.type, enabled=use_amp):
                            pass  # Add evaluation here
                    model.train()
                    meters.reset()

            iteration += 1
            if iteration >= args.iterations:
                break
            scheduler.step()

    logging_info("Training Finished.")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    args = opts.get_args()
    # args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    main(args)
