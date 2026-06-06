"""Food image classifier built on EfficientNet."""

import torch
from efficientnet_pytorch import EfficientNet


class FoodClassifier:
    """Wraps an EfficientNet model for food image classification."""

    def __init__(self, num_classes: int, model_name: str = "efficientnet-b0"):
        self.num_classes = num_classes
        self.model = EfficientNet.from_pretrained(
            model_name, num_classes=num_classes
        )

    def predict(self, image_tensor):
        """Return class logits for a preprocessed image tensor."""
        self.model.eval()
        with torch.no_grad():
            return self.model(image_tensor)
