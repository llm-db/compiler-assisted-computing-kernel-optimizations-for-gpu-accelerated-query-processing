

import torch

def masked_gather(
        x: torch.Tensor, 
        indices: torch.Tensor,
        stream: torch.Stream
    ):
    with torch.cuda.stream(stream):
        return torch.index_select(x, 0, indices)