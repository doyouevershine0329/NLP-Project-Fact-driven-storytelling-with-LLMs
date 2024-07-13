import torch
from torch import nn
import torch.nn.functional as F
import logging

class ContrastiveLoss(nn.Module):

    """
    Compute the modified NT-Xent loss for contrastive learning with multiple negatives.
    
    Parameters:
    - anchor: Tensor of shape (batch_size, feature_dim)
    - positive: Tensor of shape (batch_size, feature_dim)
    - negatives: Tensor of shape (batch_size, num_negatives, feature_dim)
    - temperature: A temperature scaling factor (scalar)
    
    Returns:
    - loss: Scalar tensor representing the cross-entropy loss
    """

    def __init__(self, model, temperature):
        super(ContrastiveLoss, self).__init__()
        self.model = model
        self.temperature = temperature

    
    def forward(self, anchor, positive, negatives, temperature=0.1):

        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=2)

        positive_similarity = F.cosine_similarity(anchor, positive, dim=1).unsqueeze(1)  # tensor dimensions are aligned to compare positive and negative similarity
        # bf: batch_size + feature_dim, bnf: batch_size + num_negatives + feature_dim, bn: batch_size(row) * num_negatives(col)
        # dot product between anchor and each negative in the batch
        negatives_similarity = torch.einsum('bf,bnf->bn', anchor, negatives)

        # concatenate positive and negative similarities to one tensor
        logits = torch.cat([positive_similarity, negatives_similarity], dim=1) / temperature
        # create target labels for cross-entropy loss
        # the positive class for each example in the batch is at first column (index 0) of the logits tensor
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=anchor.device)
        
        # computes the probability of the class at index 0 being the correct class. 
        loss = F.cross_entropy(logits, targets)
        
        return loss
        