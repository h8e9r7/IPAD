import torch
import torch.nn as nn

class MomentumEncoder(nn.Module):
    def __init__(self, cfg):
        super(MomentumEncoder, self).__init__()
        self.feature_dim = cfg.MODEL.EMBED_SIZE
        self.momentum = cfg.MODEL.MEMORYBANK.MOMENTUM
        self.device = cfg.DEVICE
        self.prototype = nn.Parameter(torch.zeros(self.feature_dim, device=self.device))
        self.updated = False  # To track if the prototype has been updated

    def update_prototype(self, new_feature):
        """
        Update prototype parameter with momentum
        """
        if not self.updated:
            self.prototype.data.copy_(new_feature.to(self.device))
            self.updated = True
        else:
            self.prototype.data.mul_(self.momentum).add_(new_feature.to(self.device), alpha=1 - self.momentum)

    def forward(self):
        """
        Get the current prototype
        """
        return self.prototype

    def reset_parameters(self):
        """
        Reset prototype parameter to zeros
        """
        nn.init.zeros_(self.prototype)
        self.updated = False

    def batch_update_prototype(self, batch_features):
        """
        Batch update prototype using mean of input batch features
        """
        mean_feature = torch.mean(batch_features, dim=0)
        self.update_prototype(mean_feature)

    def get_prototype(self):
        """
        Get the current prototype
        """
        return self.prototype.clone().detach()

    def to_device(self, device):
        """
        Move the prototype to the specified device
        """
        self.prototype.data = self.prototype.data.to(device)
        self.device = device

    def state_dict(self):
        """
        Return the current state of the prototype as a state dictionary
        """
        return {"prototype": self.get_prototype()}

    def load_state_dict(self, state_dict):
        """
        Load the prototype state from a state dictionary
        """
        self.prototype.data.copy_(state_dict["prototype"].to(self.device))
        self.updated = True
