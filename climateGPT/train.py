import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict

import structlog
import torch
from simple_parsing import ArgumentParser

import wandb
from climateGPT.export import model_export
from climateGPT.iterate import TokenBatches, TokenIterator
from climateGPT.model import ModelArgs, Transformer
from climateGPT.tokenize import Tokenizer

log = structlog.get_logger()


# -----------------------------------------------------------------------------
# I/O
@dataclass
class EvalConfig:
    out_dir: Path = Path("out")
    eval_interval: int = 1000
    log_interval: int = 100
    eval_iters: int = 100
    always_save_checkpoint: bool = (
        False  # if True, always save a checkpoint after each eval
    )
    training_type: str = "pretraining"  # or "finetuning
    init_weights: str = "random"  # or "checkpoint"


# data
@dataclass
class BatchConfig:
    batch_size: int = (
        1  # if gradient_accumulation_steps > 1, this is the micro-batch size
    )
    gradient_accumulation_steps: int = 1  # used to simulate larger batch sizes
    num_workers: int = 0
    seed_offset: int = 0
    dataset_class: str = "iterator"  # or "batches"


@dataclass
class OptimizerConfig:
    # adamw optimizer
    learning_rate: float = 5e-4  # max learning rate
    max_iters: int = 100000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 1000  # how many steps to warm up for


# system
@dataclass
class SystemConfig:
    device: str = "mps:0"  # 'cpu', 'cuda', "mps"
    dtype: str = "float16"  # float32|bfloat16|float16
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster


# logging
@dataclass
class WandbLog:
    wandb_log: bool = False
    wandb_project: str = "climateGPT"
    wandb_run_name: str = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


# -----------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn


# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
config = {k: globals()[k] for k in config_keys}  # will be useful for logging


# -----------------------------------------------------------------------------
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, iter_batches, eval_iters: int) -> Dict[str, float]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                _ = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()  # type: ignore
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(
    it: int, warmup_iters: int, lr_decay_iters: int, min_lr: float, learning_rate: float
) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # parse all arguments
    parser = ArgumentParser(
        description="Train a Transformer model on climate data",
    )
    parser.add_arguments(
        EvalConfig,
        dest="eval_config",
    )
    parser.add_arguments(
        BatchConfig,
        dest="batch_config",
    )
    parser.add_arguments(
        OptimizerConfig,
        dest="optimizer_config",
    )

    parser.add_arguments(
        SystemConfig,
        dest="system_config",
    )

    parser.add_arguments(
        WandbLog,
        dest="wandb_log",
    )

    parser.add_arguments(
        ModelArgs,
        dest="model_config",
    )

    args = parser.parse_args()

    # instantiate all parameters

    if args.eval_config.training_type == "finetuning":
        args.eval_config.init_weights = "checkpoint"
        save_name = "ckpt_ft"
        log.info("Finetuning mode, initializing model from checkpoint")
    else:
        save_name = "ckpt"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.system_config.dtype]
    ctx = (
        nullcontext()
        if args.system_config.device != "cuda"
        else torch.amp.autocast(device_type=args.system_config.device, dtype=ptdtype)
    )

    # -----------------------------------------------------------------------------
    # Weight & Biases logging
    # logging
    if args.wandb_log.wandb_log:
        wandb.init(
            project=args.wandb_log.wandb_project,
            name=args.wandb_log.wandb_run_name,
            config=config,
        )

    # -----------------------------------------------------------------------------
    # fixing some hyperparams to sensible defaults
    lr_decay_iters = (
        args.optimizer_config.max_iters
    )  # should be ~= max_iters per Chinchilla
    min_lr = 0.0  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # -----------------------------------------------------------------------------
    tokens_per_iter = (
        args.batch_config.gradient_accumulation_steps
        * args.batch_config.batch_size  # noqa
        * args.model_config.max_context_length  # noqa
    )
    log.info(f"tokens per iteration will be: {tokens_per_iter:,}")
    log.info(
        f"""breaks down as:
            > {args.batch_config.gradient_accumulation_steps} grad accum steps  *
            >  {args.batch_config.batch_size} batch size *
            >  {args.model_config.max_context_length} context length"""
    )
    args.eval_config.out_dir.mkdir(exist_ok=True)
    # -----------------------------------------------------------------------------
    torch.manual_seed(1337 + args.batch_config.seed_offset)

    # -----------------------------------------------------------------------------
    if args.eval_config.init_weights == "random":
        iter_num = 0
        best_val_loss = 1e9
        # model init
        log.info("Initializing a new model from scratch")
        model = Transformer(args.model_config)
    elif args.eval_config.init_weights == "checkpoint":
        log.info(f"Resuming training from {args.eval_config.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = Path(args.eval_config.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=args.system_config.device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume
        #  training the rest of the attributes (e.g. dropout) can stay as desired from
        # command line
        for k in [
            "dim",
            "n_layers",
            "n_heads",
            "vocab_size",
            "multiple_of",
            "hidden_dim",
            "hidden_dim_multiplier",
            "max_context_length",
        ]:
            setattr(args.model_config, k, getattr(checkpoint_model_args, k))
        # create the model
        model = Transformer(args.model_config)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        if args.eval_config.training_type == "finetuning":
            iter_num = 0
        else:
            iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    model.to(args.system_config.device)

    # compile the model
    if args.system_config.compile:
        log.info("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    optimizer = model.configure_optimizer(
        args.optimizer_config.weight_decay,
        args.optimizer_config.learning_rate,
        (args.optimizer_config.beta1, args.optimizer_config.beta2),
        args.system_config.device,
    )

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.system_config.dtype == "float16"))

    # -----------------------------------------------------------------------------
    # Dataloader

    if args.batch_config.dataset_class == "iterator":
        iter_params = {
            "pretokenized_source": Path(
                f"climateGPT/data/tok{args.model_config.vocab_size}"
            ),
            "context_length": args.model_config.max_context_length,
            # "verbose": True,
        }

        iter_batches = partial(
            TokenIterator.iter_batches,
            batch_size=args.batch_config.batch_size,
            device=args.system_config.device,
            num_workers=args.batch_config.num_workers,
            **iter_params,
        )
    else:
        batch_params = {
            "pretokenized_source": Path(
                f"climateGPT/data/fine_tuning/vocab_{args.model_config.vocab_size}_context_{args.model_config.max_context_length}"  # noqa
            ),
            "context_length": args.model_config.max_context_length,
        }

        iter_batches = partial(
            TokenBatches.iter_batches,
            batch_size=args.batch_config.batch_size,
            device=args.system_config.device,
            num_workers=args.batch_config.num_workers,
            **batch_params,
        )
    # training
    train_batch_iter = iter_batches(split="train")
    X, Y = next(train_batch_iter)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model
    running_mfu = -1.0

    # training loop
    while True:
        # determine and set the learning rate for this iteration
        lr = (
            get_lr(
                it=iter_num,
                warmup_iters=args.optimizer_config.warmup_iters,
                lr_decay_iters=lr_decay_iters,
                min_lr=min_lr,
                learning_rate=args.optimizer_config.learning_rate,
            )
            if args.optimizer_config.decay_lr
            else args.optimizer_config.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_config.eval_interval == 0:
            losses = estimate_loss(
                model=model,
                iter_batches=iter_batches,
                eval_iters=args.eval_config.eval_iters,
            )
            log.info(
                f"Step {iter_num}",
                train_loss=f"{losses['train']:.4f}",
                val_loss=f"{losses['val']:.4f}",
            )
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    },
                    step=iter_num,
                )
            except Exception as e:
                log.info(f"logging to wandb failed: {e}")
            if losses["val"] < best_val_loss or args.eval_config.always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": args.model_config,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    log.info(f"saving checkpoint to {args.eval_config.out_dir}")
                    torch.save(
                        checkpoint,
                        os.path.join(args.eval_config.out_dir, f"{save_name}.pt"),
                    )
                    model_export(
                        raw_model,
                        os.path.join(args.eval_config.out_dir, f"{save_name}.bin"),
                        version=0,
                    )

        # forward backward update, with optional gradient accumulation

        for micro_step in range(args.batch_config.gradient_accumulation_steps):
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
                loss = (
                    loss / args.batch_config.gradient_accumulation_steps
                )  # type: ignore
            X, Y = next(train_batch_iter)  # fetch the next batch asynchrounously
            scaler.scale(loss).backward()  # type: ignore
        # clip the gradient
        if args.optimizer_config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.optimizer_config.grad_clip
            )
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.eval_config.log_interval == 0:
            lossf = (
                loss.item()
                * args.batch_config.gradient_accumulation_steps  # type: ignore # noqa
            )
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    args.batch_config.batch_size
                    * args.batch_config.gradient_accumulation_steps,  # noqa
                    dt,
                    flops_promised=2.6e12,
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            log.info(
                f"Step {iter_num}",
                loss=f"{lossf:.4f}",
                lr=f"{lr:e}",
                ms=f"{dt*1000:.2f}",
                mfu=f"{running_mfu*100:.2f}%",
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > args.optimizer_config.max_iters:
            break

    model.eval()

    # load the tokenizer
    tokenizer_model_path = f"climateGPT/models/tok{args.model_config.vocab_size}.model"
    enc = Tokenizer(tokenizer_model_path=Path(tokenizer_model_path))

    num_samples = 1  # number of samples to draw
    max_new_tokens = 100  # number of tokens generated in each sample
    temperature = (
        1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    )
    top_k = 300  # retain only the top_k most likely tokens, clamp others to have 0
    # probability
    tokenizer = ""  # override the tokenizer model path
    seed = 1337

    # encode the beginning of the prompt
    start_ids = enc.encode("Climate change is", bos=True, eos=False)
    x = torch.tensor(start_ids, dtype=torch.long, device=args.system_config.device)[
        None, ...
    ]

    log.info("Run sample generation...")
    # run generation
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(
                    x, max_new_tokens, temperature=temperature, top_k=top_k
                )
                log.info(enc.decode(y[0].tolist()))

# TODO: add MoE layer and training loop

# TODO: create new torch dataset for climate data fine tuning
# TODO: create instruct dataset
# TODO: revamp code from FastGPT repo
