import torch
import argparse
from Transformer import Transformer

parser = argparse.ArgumentParser(description='Test transformer')

parser.add_argument('--src_len', '-s', type=int, default=5, help='the source sequence length')
parser.add_argument('--batch_size', '-bs', type=int, default=2, help='batch size')
parser.add_argument('--dmodel', '-d', type=int, default=512, help='embedding dimension')
parser.add_argument('--tgt_len', '-t', type=int, default=6, help='the target sequence length')
parser.add_argument('--num_head', '-nh', type=int, default=8, help='the number of head')
parser.add_argument('--num_encoder_layers', '-ne', type=int, default=6, 
                    help='the number of encoder layers')
parser.add_argument('--num_decoder_layers', '-nd', type=int, default=6, 
                    help='the number of decoder layers')
parser.add_argument('--dim_feedforward', '-df', type=int, default=2048, 
                    help='the dimension of the feedforward network model')

args = parser.parse_args()

if __name__ == '__main__':
    print(args)
    transformer = Transformer(d_model=args.dmodel, 
                             nhead=args.num_head, 
                             num_encoder_layers=args.num_encoder_layers,
                             num_decoder_layers=args.num_decoder_layers, 
                             dim_feedforward=args.dim_feedforward
                             )

    src = torch.rand((args.src_len, args.batch_size, args.dmodel))  # shape: [src_len, batch_size, embed_dim]
    # src_mask = transformer.generate_square_subsequent_mask(args.src_len)
    src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                         [True, True, True, True, False]])  # shape: [batch_size, src_len]

    tgt = torch.rand((args.tgt_len, args.batch_size, args.dmodel))  # shape: [tgt_len, batch_size, embed_dim]
    tgt_mask = transformer.generate_square_subsequent_mask(args.tgt_len)
    tgt_key_padding_mask = torch.tensor([[True, True, True, False, False, False],
                                         [True, True, True, True, False, False]])  # shape: [batch_size, tgt_len]

    out = transformer(src=src, 
                     tgt=tgt, 
                     tgt_mask=tgt_mask,
                     src_key_padding_mask=src_key_padding_mask,
                     tgt_key_padding_mask=tgt_key_padding_mask,
                     memory_key_padding_mask=src_key_padding_mask
                     )
    print(out.shape)
