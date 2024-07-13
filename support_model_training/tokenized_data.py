import torch
from torch.utils.data import Dataset

class TokenizedDataset(Dataset):
    def __init__(self, arg_tokens, pos_tokens, neg_tokens):

        """
        :param arg_tokens: Dictionary containing 2D tensors for 'input_ids' and 'attention_mask' for arguments
        :param pos_tokens: Dictionary containing 2D tensors for 'input_ids' and 'attention_mask' for positive evidences
        :param neg_tokens: Dictionary containing 3D tensors for 'input_ids' and 'attention_mask' for negative evidences (num_samples, num_neg_for_each, 512)
        """

        self.arg_tokens = arg_tokens
        self.pos_tokens = pos_tokens
        self.neg_tokens = neg_tokens

    def __len__(self):
        # Assuming all inputs are the same length
        return len(self.arg_tokens['input_ids'])

    def __getitem__(self, idx):
        
        arg_item = {key: val[idx] for key, val in self.arg_tokens.items()}
        pos_item = {key: val[idx] for key, val in self.pos_tokens.items()}
        
        # retrieve all negative evidences for the given example
        neg_items = {key: val[idx] for key, val in self.neg_tokens.items()} 
        
        '''
        neg_items = []
        for neg in self.neg_tokens:
            if idx < len(neg['input_ids']):  # Check if idx is valid for this negative batch
                neg_items.append({key: val[idx] for key, val in neg.items()})
            else:
                # Handle the error or provide a default
                neg_items.append({key: torch.tensor([]) for key in neg.keys()})
        '''
        

        return arg_item, pos_item, neg_items