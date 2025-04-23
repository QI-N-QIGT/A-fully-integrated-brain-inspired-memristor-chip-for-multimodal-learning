import torch

def flattern_new(input, dims):
    if isinstance(input, tuple):
        return (torch.flatten(input[0], dims), input[1])
    else:
        return torch.flatten(input, dims)
funcmapping = {torch.flatten: flattern_new}