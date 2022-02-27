"""
Reimplementation of GlobalPointer in PyTorch

Notice:
To assure this work is not far away from the original tf code, a careful comparsion is made
in `tests/test_globalpointer` file
"""
import torch


class GlobalPointer(torch.nn.Module):
    def __init__(self, emb_dims, max_seq_len, heads, head_size=64):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.linear = torch.nn.Linear(emb_dims, self.heads*self.head_size*2)
        self.sin_pos, self.cos_pos = self.get_sin_and_cos_pos(max_seq_len, head_size)

    @staticmethod
    def get_sin_and_cos_pos(seq_len, head_size):
        """
        input size is [B, L, heads, head_size*2]
        params:
            seq_len: max length of sequences
            head_size
        """

        """ sinusoidal positional embedding """
        position_ids = torch.arange(seq_len).unsqueeze(dim=-1).float()
        indices = torch.arange(head_size // 2)
        indices = torch.pow(10000.0, -2 * indices.true_divide(head_size))
        indices = torch.unsqueeze(indices, dim=0)
        embeddings = torch.mul(position_ids, indices).unsqueeze(dim=0)

        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, [1, seq_len, head_size])  # [1, L, head_size]

        cos_pos = embeddings[:, :, 1::2]
        cos_pos = torch.stack([cos_pos, cos_pos], dim=-1).reshape(1, seq_len, 1, head_size)
        sin_pos = embeddings[:, :, ::2]
        sin_pos = torch.stack([sin_pos, sin_pos], dim=-1).reshape(1, seq_len, 1, head_size)
        return sin_pos, cos_pos
    
    def forward(self, inputs, mask, skip_linear=False):
        """
        params:
            inputs: [B, L, D]
            mask(IntTensor): [B, L], 
            skip_linear(bool): when compare result with tf version globalpointer, we skip 
            linear to avoid the influence caused by linear weight initializing, and the input
            will change [B, L, D] to [B, L, head*head_size*2]
        returns:
            logits: [B, heads, L, L]
        """

        if not skip_linear:
            inputs = self.linear(inputs)
        inputs = torch.split(inputs, self.head_size*2, dim=-1)
        inputs = torch.stack(inputs, dim=-2)
        qw, kw = inputs[:, :, :, :self.head_size], inputs[:, :, :, self.head_size:]

        sin_pos = self.sin_pos.to(inputs.device)
        cos_pos = self.cos_pos.to(inputs.device)

        qw2 = torch.stack([-qw[:, :, :, 1::2], qw[:, :, :, ::2]], dim=4)
        qw2 = qw2.reshape(qw.size())
        qw = qw * cos_pos + qw2 * sin_pos  # [B, L, heads, head_size]
        
        kw2 = torch.stack([-kw[:, :, :, 1::2], kw[:, :, :, ::2]], dim=4)
        kw2 = kw2.reshape(kw.size())
        kw = kw * cos_pos + kw2 * sin_pos

        qw = qw.permute(0, 2, 1, 3).unsqueeze(dim=-2)
        kw = kw.permute(0, 2, 1, 3).unsqueeze(dim=-3)
        logits = torch.mul(qw, kw).sum(dim=-1)  # [B, heads, L(qw), L(kw)]

        """ exclude padding """
        mask = torch.abs(mask - 1)  # reverse
        logits = torch.masked_fill(logits, mask.unsqueeze(dim=1).unsqueeze(dim=-1), -1e12)
        logits = torch.masked_fill(logits, mask.unsqueeze(dim=1).unsqueeze(dim=1), -1e12)

        """ exclude bottom diag """
        triu_mask = torch.triu(torch.ones_like(logits)).to(inputs.device)
        logits = logits - (1 - triu_mask) * 1e12
        return logits.true_divide(self.head_size ** 0.5)
