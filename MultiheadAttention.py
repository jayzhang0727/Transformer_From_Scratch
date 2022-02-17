import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    Allows the model to jointly attend to information
    from different representation subspaces.

    .. math:: \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim   # d_model
        self.head_dim = embed_dim // num_heads   # head_dim is d_q = d_k = d_v
        self.kdim = self.head_dim
        self.vdim = self.head_dim 
        self.num_heads = num_heads
        self.dropout = dropout

        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)    # W^Q
        # combine all num_heads weight matrices W^Q together and embed_dim = head_dim * num_heads
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)    # W^K
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)    # W^V
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # W^O   

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        Input (shape):
        query: [tgt_len, batch_size, embed_dim]
        key: [src_len, batch_size, embed_dim] 
        value: [src_len, batch_size, embed_dim]
        attn_mask: [tgt_len, src_len] or [batch_size * num_heads, tgt_len, src_len]
        key_padding_mask: [batch_size, src_len]
        
        Return (shape):
        attn_output: [tgt_len, batch_size, embed_dim]
        attn_output_weights: [batch_size, tgt_len, src_len]
        """
        
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout,
                                            out_proj=self.out_proj,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj=self.q_proj,
                                            k_proj=self.k_proj,
                                            v_proj=self.v_proj,
                                            attn_mask=attn_mask)


def multi_head_attention_forward(query,  # [tgt_len, batch_size, embed_dim]
                                 key,  # [src_len, batch_size, embed_dim]
                                 value,  # [src_len, batch_size, embed_dim]
                                 num_heads, 
                                 dropout_p,
                                 out_proj, 
                                 training=True,
                                 key_padding_mask=None,  # [batch_size, src_len]
                                 q_proj=None, 
                                 k_proj=None,  
                                 v_proj=None,  
                                 attn_mask=None,  # [tgt_len, src_len] or [num_heads*batch_size, tgt_len, src_len]
                                 ):
    q = q_proj(query)
    # [tgt_len, batch_size, embed_dim] --> [tgt_len, batch_size, num_heads * kdim]
    # embed_dim = kdim * num_heads 
    k = k_proj(key)
    # [src_len, batch_size, embed_dim] --> [src_len, batch_size, num_heads * kdim]
    v = v_proj(value)
    # [src_len, batch_size, embed_dim] --> [src_len, batch_size, num_heads * vdim]

    tgt_len, bsz, embed_dim = query.size()  # [tgt_len, batch_size, embed_dim]
    src_len = key.size(0)
    head_dim = embed_dim // num_heads  # num_heads * head_dim = embed_dim
    scaling = float(head_dim) ** -0.5
    q = q * scaling  # [tgt_len, batch_size, num_heads * kdim]

    if attn_mask is not None:   # attn_mask can be 2 or 3 dimensional
        if attn_mask.dim() == 2:  # [tgt_len, src_len]
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len, src_len]
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:  # [batch_size * num_heads, tgt_len, src_len]
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        # Now attn_mask becomes 3 dimensional

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # [batch_size * num_heads, tgt_len, kdim]
    k = k.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)  
    # [batch_size * num_heads, src_len, kdim]
    v = v.contiguous().view(src_len, bsz * num_heads, head_dim).transpose(0, 1)  
    # [batch_size * num_heads, src_len, vdim]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # [batch_size * num_heads, tgt_len, src_len]  

    if attn_mask is not None:
        attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        #[batch_size, num_heads, tgt_len, src_len]
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.view(bsz, 1, 1, src_len),  # [batch_size, 1, 1, src_len]
            float('-inf'))  
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)  
        # [batch_size * num_heads, tgt_len, src_len]

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
    attn_output = torch.bmm(attn_output_weights, v) # [batch_size * num_heads, tgt_len, vdim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    #  [tgt_len, batch_size, num_heads * kdim] = [tgt_len, batch_size, embed_dim]
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    Z = out_proj(attn_output) # Z is the Multi-head Attention, i.e. Z = MultiHead(Q, K, V)
    # shape of Z: [tgt_len, batch_size, embed_dim]

    return Z, attn_output_weights.sum(dim=1) / num_heads  # average attention weights over heads

