import time

out_dir = 'out-shakespeare-shannon'
eval_interval = 100
eval_iters = 100
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare-shannon'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'shakespeare_shannon'
init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 1
gradient_accumulation_steps = 32
max_iters = 1000
pkl_file_path = './data/shakespeare_shannon/n_gram_list.pkl'
validation_flag = False

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
