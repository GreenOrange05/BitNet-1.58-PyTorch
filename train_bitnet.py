import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import math
import os

# --- CONFIGURATION ---
batch_size = 64
block_size = 256
max_iters = 3000       # The "Winning" Step Count
learning_rate = 1.5e-3 # High LR for BitNet
min_lr = 1.5e-4
warmup_iters = 100
eval_interval = 250
dropout = 0.0          # No dropout needed for 1-bit models (they self-regularize)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model Architecture
n_embd = 256
n_head = 8
n_layer = 6

print(f"--- BITNET TRAINING ---")
print(f"Device: {device}")
print("-----------------------")

# --- DATA LOADING ---
file_path = 'sherlock.txt'
if not os.path.exists(file_path):
    print("Downloading dataset...")
    os.system('curl -O https://sherlock-holm.es/stories/plain-text/cano.txt')
    # Rename to generic name for ease
    if os.path.exists('cano.txt'):
        os.rename('cano.txt', 'sherlock.txt')

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- BITNET ARCHITECTURE ---

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class BitLinear(nn.Linear):
    def forward(self, x):
        w = self.weight
        RMSNorm_weight = torch.rsqrt(w.pow(2).mean().clamp(min=1e-5))
        
        # 1. Quantize Weights to {-1, 0, 1}
        w_quant = (w * RMSNorm_weight).round().clamp(-1, 1) / RMSNorm_weight
        
        # 2. Straight-Through Estimator (STE)
        w_quant = (w_quant - w).detach() + w
        
        # 3. Activation Quantization (Simple version)
        scale = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
        x_quant = (x * scale).round().clamp(-128, 127) / scale
        x_quant = (x_quant - x).detach() + x
        
        return F.linear(x_quant, w_quant)

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = BitLinear(n_embd, head_size, bias=False)
        self.query = BitLinear(n_embd, head_size, bias=False)
        self.value = BitLinear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if self.training else 0, is_causal=True)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = BitLinear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            BitLinear(n_embd, 4 * n_embd, bias=False),
            nn.GELU(),
            BitLinear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, BitLinear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

# --- TRAINING LOOP ---
def get_lr(it):
    if it < warmup_iters: return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > max_iters: return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ == '__main__':
    torch.manual_seed(1337)
    model = GPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    for iter in range(max_iters):
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter % eval_interval == 0:
            print(f"Step {iter}: Training...")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), 'bitnet_model.pth')
    print("Saved to 'bitnet_model.pth'")