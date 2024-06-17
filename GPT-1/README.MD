## Calculate theoretical entropy upper and lower bounds derived by Shannon

### Calculate the Shannon entropy upper and lower bounds using the Brown dataset
```
python3 main_brown.py
```
- line 114: Set N=2 for bi-gram, and set N=3 for tri-gram.

### Calculate the Shannon entropy upper and lower bounds using the Shakespeare dataset
```
python3 main_shakespeare.py
```
- line 126: Set N=2 for bi-gram, and set N=3 for tri-gram.

## GPT-1 experiments for N-gram prediction

### Run GPT-1 experiments on the Brown dataset
```
python3 run_lm_brown.py
```
- line 16: Set N=2 for bi-gram, and set N=3 for tri-gram.
- line 17: Set pretrained=True if use pretrained checkpoint, set pretrained=False if train from scratch.

### Run GPT-1 experiments on the Shakespeare dataset
```
python3 run_lm_shakespeare.py
```
- line 16: Set N=2 for bi-gram, and set N=3 for tri-gram.
- line 17: Set pretrained=True if use pretrained checkpoint, set pretrained=False if train from scratch.

