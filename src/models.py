# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class CorruptionDetector(nn.Module):
    """ResNet-based corruption detection model"""

    def __init__(self, num_classes=5, pretrained=True):
        super(CorruptionDetector, self).__init__()

        # Load pretrained ResNet18
        if pretrained:
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)

        # Modify final layer for our classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Extract features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global average pooling
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # Add dropout before final layer
        x = self.dropout(x)
        x = self.backbone.fc(x)

        return x

    def extract_features(self, x):
        """Extract features for visualization/analysis"""
        with torch.no_grad():
            # Get features before final layer
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            features = self.backbone.avgpool(x)
            features = torch.flatten(features, 1)

            return features


class GradCAM:
    """Gradient-weighted Class Activation Mapping for visualization"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class=None):
        """Generate Class Activation Map"""

        # Forward pass
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        # Generate CAM
        gradients = self.gradients
        activations = self.activations

        # Pool gradients across spatial dimensions
        pooled_gradients = torch.mean(gradients, dim=[2, 3])

        # Weight activations by pooled gradients
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[:, i]

        # Average over feature maps
        heatmap = torch.mean(activations, dim=1).squeeze()

        # ReLU on top of heatmap
        heatmap = F.relu(heatmap)

        # Normalize heatmap
        heatmap /= torch.max(heatmap)

        return heatmap.detach()


def get_model(num_classes=5, pretrained=True):
    """Factory function to create model"""
    return CorruptionDetector(num_classes=num_classes, pretrained=pretrained)


# Model summary function
def model_summary(model, input_size=(3, 224, 224)):
    """Print model summary"""
    from torchsummary import summary

    try:
        summary(model, input_size)
    except:  # noqa: E722
        print("torchsummary not available. Install with: pip install torchsummary")
        print(f"Model: {model.__class__.__name__}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )


if __name__ == "__main__":
    # Test model creation
    model = get_model(num_classes=5, pretrained=True)
    print("Model created successfully!")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

    # Model summary
    model_summary(model)
