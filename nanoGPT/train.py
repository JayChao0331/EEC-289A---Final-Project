import os
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# -----------------------------------------------------------------------------
# default config values
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
# wandb logging
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
# model
n = 3
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
# DDP settings
backend = 'nccl'
# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
data_dir = os.path.join('data', dataset)

def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r') #replace with github code
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r') #replace with test
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix]) #replace with github code
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix]) # ...
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, dropout=dropout, bias=bias,
                  vocab_size=None)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        print(f"No meta.pkl found, assuming GPT-2 encodings with vocab size of 50257")
        vocab_size = 50257
    else:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        vocab_size = meta['vocab_size']
    model_args['vocab_size'] = vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in model_args:
        model_args[k] = getattr(model.config, k)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in model_args:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
model.to(device)

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

def get_lr(iter_num):
    if iter_num < warmup_iters:
        return learning_rate * iter_num / warmup_iters
    if iter_num > lr_decay_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def collect_n_grams(X):
    n_gram_list_row = []
    for X_row in X:
        n_gram_list_row.append(generate_ngrams(X_row, n))
    return n_gram_list_row

def generate_ngrams(data, n):
    ngrams = []
    for i in range(len(data) - n + 1):
        ngrams.append(data[i:i + n])
    return ngrams

def get_ngram_probs(model, context):
    with torch.no_grad():
        context_tensor = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        logits, _ = model(context_tensor, context_tensor)
        logits = logits[:, -1, :]  # Get logits for the last token in the context
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        total_probs = probs.sum()
        probs = probs / total_probs
    return probs

def compute_shannon_entropy(probs):
    epsilon = 1e-9
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    return entropy.mean().item()

ngram_data = []
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']: # change val to test
        losses = torch.zeros(eval_iters)

        X, Y = get_batch(split)
        n_gram_list_row = collect_n_grams(X)
        with ctx:
            logits, loss = model(X, Y)

        for ngrams_row_instance in n_gram_list_row:
            # ngrams_row_instance:
            # [..., tensor([ 6, 20, 10], device='cuda:0'), tensor([20, 10, 19], device='cuda:0'),
            # tensor([10, 19,  6], device='cuda:0'),
            # tensor([19,  6, 20], device='cuda:0'),...]
            for ngram_element in ngrams_row_instance:
                ngram_prob = get_ngram_probs(model, ngram_element)  # ngram_prob: prob values for all 28 chars
                entropy = compute_shannon_entropy(ngram_prob)
                predicted_word = ngram_prob.argmax().item()
                print(predicted_word)
                ngram_data.append({
                    'predicted_word': predicted_word,
                    'probability_distribution': ngram_prob.tolist(),
                    'entropy': entropy
                })
            for entry in ngram_data:
                print(f"Predicted Word: {entry['predicted_word']}, Entropy: {entry['entropy']:.4f}")
                print(f"Probability Distribution: {entry['probability_distribution']}\n")
        for k in range(eval_iters):
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

X, Y = get_batch('train')
#for X:
#tensor([[20,  1, 24,  ..., 19,  6, 20],
        # [22, 19,  5,  ...,  6, 19,  0],
        # [16, 19,  1,  ..., 26, 16, 22],
        # ...,
        # [ 1,  7, 22,  ...,  6,  1, 17],
        # [20,  6, 19,  ..., 13,  2, 22],
        # [26,  1, 21,  ..., 19,  1, 14]], device='cuda:0')

while iter_num < max_iters:
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()

        if master_process:
            print(
                f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log and master_process:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if master_process:
                print(f"saving checkpoint to {out_dir}")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, Y) # logits unused, for: _
            loss /= gradient_accumulation_steps
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    iter_num += 1

    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        if master_process:
            print(f"iter {iter_num}: loss {lossf:.4f}")
        if wandb_log and master_process:
            wandb.log({
                "iter": iter_num,
                "train/loss": lossf
            })
if ddp:
    destroy_process_group()