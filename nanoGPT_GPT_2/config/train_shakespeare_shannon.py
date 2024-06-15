out_dir = 'out-shakespeare-shannon'
pt_file_name = 'ckpt_shakespeare_shannon_resume'
init_from = 'resume'


eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-shannon'
wandb_run_name = 'mini-gpt'

pkl_file_path = './data/shakespeare_shannon/n_gram_list.pkl'
dataset = 'shakespeare_shannon'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

validation_flag = False

wind = 0
n_register_token = 0
use_LLaMA_flag = False
use_abs_softmax = False

learning_rate = 5e-5 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100