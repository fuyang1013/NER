import pdb
import torch

from bert_globalpointer.data_utils import sparse_label_ids
from bert_globalpointer.gp import GlobalPointer


class BertGP(torch.nn.Module):
    def __init__(self, num_labels, ptm_model, max_seq_len) -> None:
        super().__init__()
        self.ptm_model = ptm_model
        self.globalpointer = GlobalPointer(
            self.ptm_model.config.hidden_size,
            max_seq_len,
            num_labels - 1  # no head for label 'O'
        )
        self.dropout = torch.nn.Dropout(p=0.1)
        self.apply(self.init_weight)
    
    def init_weight(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(
                mean=0.0, 
                std=self.ptm_model.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(
                mean=0.0,
                std=self.ptm_model.config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids, 
        attention_masks, 
        token_type_ids, 
        position_ids,
        label_ids=None
    ):
        token_hidden_states = self.ptm_model(
            input_ids,
            attention_masks,
            token_type_ids,
            position_ids)['last_hidden_state']
        token_hidden_states = self.dropout(token_hidden_states)
        token_logits = self.globalpointer(token_hidden_states, attention_masks)
        
        if label_ids is not None:
            label_multihead_matrices = sparse_label_ids(label_ids, self.heads)
            loss_fn = GlobalPointerLoss()
            loss = loss_fn(token_logits, label_multihead_matrices)
            return token_logits, loss
        return token_logits