import os
import torch
import torch.nn as nn


MODEL_DIR = "saved_models"
MODEL_NAME = "facenet.pt"

class ModelTrainer:
    
    def __init__(self, net, optimizer, criterion, train_loader, val_loader):
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _single_pass(self, mode, data_loader):
        if(mode == "train"):
            self.net.train()
        else:
            self.net.eval()

        running_loss = 0.0    
        num_images = 0
        for batch_i, data in enumerate(data_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor)
            images = images.type(torch.FloatTensor)
            # forward pass to get outputs
            output_pts = self.net(images)

            # calculate the loss between predicted and target keypoints
            loss = self.criterion(output_pts, key_pts)

            if(mode == "train"):
                # zero the parameter (weight) gradients
                self.optimizer.zero_grad()
                # backward pass to calculate the weight gradients
                loss.backward()
                # update the weights
                self.optimizer.step()

            running_loss += loss.item()
            num_images += len(images)

        total_loss = 100*running_loss / num_images
        return total_loss
    
    def init_weights(self):
        """Initialize Weights """
        for param in self.net.parameters():
            if(param.dim() > 1):
                nn.init.xavier_normal_(param)

    def train(self, n_epochs):
        """Train the model for n_epochs """
        for i in range(n_epochs):
            train_loss = self._single_pass("train", self.train_loader)
            val_loss = self._single_pass("validate", self.val_loader)
            print("Epoch: {}, Training Loss: {} | Validation Loss: {}".format(i, train_loss, val_loss))
    
    def save_weights(self, path=MODEL_DIR, model_name=MODEL_NAME):
        """Save model weights"""
        weights_file = os.path.join(path, model_name)
        torch.save(self.net.state_dict(), weights_file)
