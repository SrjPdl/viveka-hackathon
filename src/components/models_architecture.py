import torch.nn as nn
import torchvision

class InceptBaseModel(nn.Module):
    """Inception-based base model for classification.

    This model is based on the Inception v3 architecture pretrained on ImageNet.
    It consists of a modified fully connected head for binary classification.

    Example:
        >>> model = InceptBaseModel()
        >>> inputs = torch.randn(1, 3, 299, 299)
        >>> outputs = model(inputs)
    """

    def __init__(self):
        super(InceptBaseModel, self).__init__()
        self.base_model = torchvision.models.inception_v3(weights='DEFAULT')
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self,inputs):
        """Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) with sigmoid activation.
        """
        x = self.base_model(inputs)
        return x