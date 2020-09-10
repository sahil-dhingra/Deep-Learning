import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        resnet = models.vgg16(pretrained=True)
        self.features = resnet.features
        for param in self.features.parameters(): #NOTE: prune:True  // finetune:False
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Linear(128, 10)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        # conv layers

        x = self.features(images)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.out(x)
        scores = F.softmax(x, dim=1)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores