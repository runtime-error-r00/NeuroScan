"""Grad-CAM: which spatial regions of a convolutional layer increase the score of a class."""

import torch
import torch.nn.functional as F


class GradCAM:
    """Grad-CAM on `target_layer` (e.g. model.layer4 for a ResNet)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.activations = output.detach()               # (1, C, h, w) feature maps
        output.register_hook(self._save_gradient)        # called during backward

    def _save_gradient(self, grad):
        self.gradients = grad.detach()                   # (1, C, h, w) d score / d feature maps

    def __call__(self, x, class_idx=None):
        """x: one preprocessed image (C, H, W). Returns (cam in [0, 1] of shape (H, W), class_idx, probs)."""
        device = next(self.model.parameters()).device
        self.model.eval()
        logits = self.model(x.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)                  # importance of each channel
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))      # weighted sum -> (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)                                           # scale to [0, 1]
        return cam.cpu().numpy(), class_idx, probs
