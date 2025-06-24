import torch
import torch.nn as nn
import timm
from typing import Optional

# ---- ViT-Tiny Backbone Loader ----
def load_vit_tiny(device: torch.device) -> nn.Module:
    """
    Load a pretrained ViT-Tiny model from TIMM as a feature extractor (no classification head).
    """
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0)
    model.to(device)
    return model

# ---- Classification Head ----
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_labels)
    def forward(self, x):
        return self.fc(x)

# ---- Adapter Integration ----
def integrate_adaptors(model: nn.Module, bottleneck_dim: int, device: torch.device) -> nn.Module:
    """
    Integrate Adaptor layers into the ViT-Tiny model's attention modules.
    Only adaptor parameters are trainable; all others are frozen.
    """
    class AdaptorLayer(nn.Module):
        def __init__(self, input_dim, bottleneck_dim):
            super().__init__()
            self.down = nn.Linear(input_dim, bottleneck_dim)
            self.relu = nn.ReLU()
            self.up = nn.Linear(bottleneck_dim, input_dim)
        def forward(self, x):
            return self.up(self.relu(self.down(x)))

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Inject adaptors into each block's attention module
    for block in model.blocks:
        attn_module = block.attn
        if not hasattr(attn_module, "adaptor_q"):
            attn_module.adaptor_q = AdaptorLayer(model.embed_dim, bottleneck_dim)
        if not hasattr(attn_module, "adaptor_v"):
            attn_module.adaptor_v = AdaptorLayer(model.embed_dim, bottleneck_dim)
        for param in attn_module.adaptor_q.parameters():
            param.requires_grad = True
        for param in attn_module.adaptor_v.parameters():
            param.requires_grad = True
    model.to(device)
    return model

# ---- LoRA Integration ----
def integrate_lora(model: nn.Module, lora_config, peft_get_model, device: torch.device) -> nn.Module:
    """
    Integrate LoRA into the ViT-Tiny model using PEFT.
    """
    lora_model = peft_get_model(model, lora_config)
    lora_model.to(device)
    return lora_model 