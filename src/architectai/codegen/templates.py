CODE_TEMPLATE = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class {class_name}(nn.Module):
    def __init__(self):
        super().__init__()
{layers}

    def forward(self, x):
{forward}
        return x
"""

LAYER_TEMPLATE = "        self.{name} = {layer_type}({params})\n"
