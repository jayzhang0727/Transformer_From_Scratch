import torch
import torch.nn as nn
import torch.nn.functional as F
from MultiheadAttention import MultiheadAttention
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoderLayer(nn.Module):
    """
    TransformerEncoderLayer is made up of self-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: [src_len, batch_size, embed_dim]
            src_mask: [src_len, src_len] or [batch_size * num_heads, src_len, src_len]
            src_key_padding_mask: [batch_size, src_len]
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0] 
        # src2: [src_len, batch_size, num_heads * kdim] where num_heads*kdim = embed_dim
        src = src + self.dropout1(src2)  
        src = self.norm1(src)  # [src_len, batch_size, embed_dim]

        src2 = self.activation(self.linear1(src))  # [src_len, batch_size, dim_feedforward]
        src2 = self.linear2(self.dropout(src2))  # [src_len, batch_size, embed_dim]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src  # [src_len, batch_size, embed_dim]


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder is a stack of N encoder layers.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)  # clone multiple encoder layers
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: [src_len,batch_size, embed_dim]
            mask: [src_len, src_len]
            src_key_padding_mask: [batch_size, src_len]
        """
        output = src
        for encoder_layer in self.layers:
            output = encoder_layer(output, src_mask=mask,
                         src_key_padding_mask=src_key_padding_mask)  
        if self.norm is not None:
            output = self.norm(output)
        return output  # [src_len, batch_size, embed_dim]


class TransformerDecoderLayer(nn.Module):
    """
    TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # Masked multihead self-attention 
        self.multihead_attn = MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)
        # Multihead attention interacted with memory (output of encoder) 

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required). 
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape: 
            tgt: [tgt_len,batch_size, embed_dim]
            memory: [src_len, batch_size, embed_dim]
            tgt_mask: [tgt_len, tgt_len]
            memory_mask: [tgt_len, src_len]
            tgt_key_padding_mask: [batch_size, tgt_len]
            memory_key_padding_mask: [batch_size, src_len]
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # [tgt_len, batch_size, embed_dim]
        tgt = tgt + self.dropout1(tgt2) 
        tgt = self.norm1(tgt)  # [tgt_len, batch_size, embed_dim]

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                        key_padding_mask=memory_key_padding_mask)[0]
        # [tgt_len, batch_size, embed_dim]
        tgt = tgt + self.dropout2(tgt2)  
        tgt = self.norm2(tgt)  # [tgt_len, batch_size, embed_dim]

        tgt2 = self.activation(self.linear1(tgt))  # [tgt_len, batch_size, dim_feedforward]
        tgt2 = self.linear2(self.dropout(tgt2))  # [tgt_len, batch_size, embed_dim]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt  # [tgt_len,batch_size,embed_dim]


class TransformerDecoder(nn.Module):
    """
    TransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        """
        Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
        tgt: [tgt_len, batch_size, embed_dim]
        memory: [src_len, batch_size, embed_dim]
        tgt_mask: [tgt_len, tgt_len]
        memory_mask: [tgt_len, src_len] 
        tgt_key_padding_mask: [batch_size, tgt_len]
        memory_key_padding_mask: [batch_size, src_len]
        """
        output = tgt  # [tgt_len,batch_size, embed_dim]

        for decoder_layer in self.layers: 
            output = decoder_layer(output, 
                                  memory,
                                  tgt_mask=tgt_mask,
                                  memory_mask=memory_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask
                                  )
        if self.norm is not None:
            output = self.norm(output)

        return output  # [tgt_len,batch_size,embed_dim]


class Transformer(nn.Module):
    """
    A transformer model. 

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
    """
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        # Encoding
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Decoding 
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        """
        Initiate parameters in the transformer model.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Take in and process masked source/target sequences.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

        Shape:
            src: [src_len, batch_size, embed_dim]
            tgt: [tgt_len, batch_size, embed_dim]
            src_mask: [src_len, src_len]
            tgt_mask: [tgt_len, tgt_len]
            memory_mask: [tgt_len, src_len]
            src_key_padding_mask: [batch_size, src_len]
            tgt_key_padding_mask: [batch_size, tgt_len]
            memory_key_padding_mask:  [batch_size, src_len]
        """
        memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        # [src_len, batch_size, embed_dim]
        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output  # [tgt_len, batch_size, embed_dim]

    