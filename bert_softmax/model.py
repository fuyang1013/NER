import pdb
import torch


class BertSoftmax(torch.nn.Module):
    def __init__(self, num_labels, ptm_model) -> None:
        super().__init__()
        self.ptm_model = ptm_model
        self.linear = torch.nn.Linear(
            self.ptm_model.config.hidden_size, num_labels)
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
        label_ids
    ):
        token_hidden_states = self.ptm_model(
            input_ids,
            attention_masks,
            token_type_ids,
            position_ids)['last_hidden_state']
        token_logits = self.linear(token_hidden_states)  # [B, L, C]

        if label_ids is not None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # [B, L]
            # pdb.set_trace()
            loss = loss_fn(token_logits.permute(0, 2, 1), label_ids)  # [B, C, L] -> [B, L]
            loss = torch.mul(attention_masks, loss).sum(dim=-1).mean()
            return token_logits, loss
        else:
            return token_logits
    
