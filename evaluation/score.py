import argparse
from pathlib import Path

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fastprogress.fastprogress import master_bar, progress_bar
from omegaconf import OmegaConf
from jiwer import wer, cer

from .dataset import PogDataset, collate_batch
from .model import TransFusion
from .diffusion import MultinomialDiffusion, index_to_log_onehot

import os
import time


name = time.strftime('%m_%d_%H_%M_%S', time.localtime())


# ids to text helper function
def to_text(pp, i2s: list): return ''.join([i2s[p] for p in pp if p != 0])

@torch.inference_mode()
def score_model(model: TransFusion, cfg, args, dl: DataLoader, vocab, dtype, device):
    """ Score a trained `model` on a specific dataset `dl` """

    diff = MultinomialDiffusion(cfg.model_cfg.vocab_size, 
        args.T if args.fs == '1' else cfg.model_cfg.T,
        cfg.model_cfg.diffusion_s,
        dtype=dtype,
        device=device
    )

    def reverse_diffusion(batch):
        x = batch[0]
        t = batch[1]

        if args.fs == '1':
            x_0_pred = model(batch[0], (batch[1] + 1) * (cfg.model_cfg.T // args.T) - 1, *batch[2:])
        else:
            x_0_pred = model(*batch)

        if (t == (0 if args.fs == '1' else cfg.model_cfg.T // args.T - 1)).all():
            x_tm1 = x_0_pred.argmax(dim=-1)
        else:
            log_x_t = index_to_log_onehot(x, diff.num_classes)
            if args.fs == '1':
                log_model_pred = diff.p_pred(log_x_t, t, index_to_log_onehot(x_0_pred.argmax(dim=-1), diff.num_classes) if args.clamp else F.log_softmax(x_0_pred.to(torch.float32), dim=-1)) # p(x_{t-1} | x_{t})
            else:
                log_model_pred = diff.p_pred_k(log_x_t, t, index_to_log_onehot(x_0_pred.argmax(dim=-1), diff.num_classes) if args.clamp else F.log_softmax(x_0_pred.to(torch.float32), dim=-1), cfg.model_cfg.T // args.T) # p(x_{t-1} | x_{t})
            x_tm1 = diff.log_sample_categorical(log_model_pred)

        return x_tm1, x_0_pred

    text_preds_all = []
    text_targets_all = []

    mb = master_bar(enumerate(dl), total=len(dl))

    for i, batch in mb:
        x, t, cond_emb, x_padding_mask, cond_padding_mask = batch
        cond_emb = cond_emb.to(dtype)
        cond_emb = cond_emb.to(device, non_blocking=True)
        x_padding_mask = x_padding_mask.to(device, non_blocking=True)
        cond_padding_mask = cond_padding_mask.to(device, non_blocking=True)
        batch = (x, t, 
                cond_emb, 
                cond_padding_mask, 
                x_padding_mask)

        if args.fs == '1':
            times = range(args.T - 1, -1, -1)
        else:
            times = range(cfg.model_cfg.T - 1, -1, -(cfg.model_cfg.T // args.T))

        effective_bs = batch[0].shape[0]
        x = torch.randint(0, diff.num_classes, (effective_bs, batch[0].shape[-1]), dtype=torch.long, device=batch[2].device)

        for j, t_last in progress_bar(enumerate(times), total=len(times), parent=mb):
            t = torch.ones((effective_bs,), dtype=torch.long, device=x.device) * (t_last)

            xx = (x, t, batch[2], batch[3], batch[4])
            x, x_0_pred = reverse_diffusion(xx)

        text_preds = [to_text(p, vocab['i2s']) for p in x]
        text_targets = [to_text(p, vocab['i2s']) for p in batch[0]]
        text_preds_all.extend(text_preds)
        text_targets_all.extend(text_targets)

        if (i + 1) % args.running == 0:
            # print progress
            running_wer = wer(text_targets_all, text_preds_all)
            running_cer = cer(text_targets_all, text_preds_all)
            mb.write(f"[{(i + 1):03d}/{len(dl):03d}] running cer: {running_cer:.2%} | running wer: {running_wer:.2%}")

    full_wer = wer(text_targets_all, text_preds_all)
    full_cer = cer(text_targets_all, text_preds_all)

    return full_cer, full_wer

def main():
    parser = argparse.ArgumentParser(description="Score a trained model on ASR metrics")

    parser.add_argument('--ckpt', required=True, type=str, help="model checkpoint to use.")
    parser.add_argument('--eval_csv', required=True, type=str, help="csv of audio & wavlm paths to eval on.")
    parser.add_argument('--vocab', required=True, type=str, help="path to vocab.pt")
    parser.add_argument('--device', default='cuda', type=str, help="device to use")
    parser.add_argument('--dtype', default='fp16', type=str, choices=['fp16', 'fp32'])
    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--seed', default=123, type=int, help='seed')
    parser.add_argument('--clamp', action='store_true', help='whether to use clamp trick')
    parser.add_argument('--fs', default='1', type=str, help='FS 1 or FS 2')
    parser.add_argument('--T', default=200, type=int, help='number of timesteps')
    parser.add_argument('--running', default=1000, type=int, help='how many steps to show the running cer and wer')

    args = parser.parse_args()
    # load checkpoints
    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    csv_pth = Path(args.eval_csv)
    vocab = torch.load(args.vocab)
    
    # load config
    cfg = OmegaConf.structured(ckpt['cfg_yaml'])
    print(f'\n{args}\n')

    # load model
    model = TransFusion(cfg.model_cfg, cfg.max_transcript_length).to(device)
    model.load_state_dict(ckpt['module'])
    model.eval()
    print(f"Model loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.\n")

    # create dataset
    df = pd.read_csv(csv_pth)[:100]
    ds = PogDataset(df, vocab['s2i'], vocab['i2s'], cfg.model_cfg.T, cfg.max_transcript_length)
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32

    # fp16 inference
    if dtype == torch.float16:
        model = model.half()
    
    dl = DataLoader(ds, args.bs, shuffle=False, collate_fn=collate_batch, num_workers=4)

    # set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    start = time.time()

    #score model
    cer, wer = score_model(model, cfg, args, dl, vocab, dtype, device)

    end = time.time()

    total = round(end - start)

    os.makedirs('./log/', exist_ok = True)

    with open('./log/cur.txt', mode = 'a') as fp:
        fp.write(f'{name}\n\n')
        fp.write(f'{args}\n\n')
        fp.write(f'CER: {cer:.2%}\n')
        fp.write(f'WER: {wer:.2%}\n')
        fp.write(f'TIME: {total // 60}m{total % 60}s\n\n')

    print('-'*50)
    print(f"\t CER: {cer:.2%}")
    print(f"\t WER: {wer:.2%}")
    print(f"\t TIME: {total // 60}m{total % 60}s")
    print('-'*50)

if __name__ == '__main__':
    main()
