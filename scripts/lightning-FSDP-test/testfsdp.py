import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.fabric.strategies import FSDPStrategy
from lightning.pytorch.demos import Transformer, WikiText2

policy = {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
strategy = FSDPStrategy(auto_wrap_policy=policy, cpu_offload=False)
fabric = L.Fabric(
    accelerator="cuda", 
    devices=1, 
    strategy=strategy,
    precision="bf16-true",
)
fabric.launch()

fabric.seed_everything(42)

with fabric.rank_zero_first():
    dataset = WikiText2()

# 1B parameters
with fabric.init_module():
    model = Transformer(vocab_size=dataset.vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

num_params = sum([p.numel() for p in model.parameters()])
print(f"expected bf16 memory usage from params: {num_params * 2 / 1e9:.2f} GB")
model, optimizer = fabric.setup(model, optimizer)
print(f"current memory usage with model on device {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

for i in range(1):
    input, target = fabric.to_device(dataset[i])
    output = model(input.unsqueeze(0), target.unsqueeze(0))
    loss = F.nll_loss(output, target.view(-1))
    fabric.backward(loss)
    print(next(model.parameters()).dtype)
    optimizer.step()
    optimizer.zero_grad()
    print(next(model.parameters()).dtype)
    print(optimizer.param_groups[0]["params"][0].dtype)
    fabric.print(loss.item())

fabric.print(torch.cuda.memory_summary())

fabric.save("bf16-checkpoint.ckpt", {"model": model})