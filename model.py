import torch.nn as nn
import torchvision.models as models

# Define the classifier with the desired num_classes for classification

class BoneTumorClassifierResNet34(nn.Module):
    def __init__(self, num_classes):
        super(BoneTumorClassifierResNet34, self).__init__()
        # Load pre-trained ResNet34 model
        self.resnet = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT)
        
        # Freeze parameters of the ResNet34 model
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer
        self.classifier = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.resnet.fc = self.classifier
        
        # Unfreeze parameters of the last fully connected layer
        for param in self.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
