# Transformer_From_Scratch
Construct a **transformer** from scratch with PyTorch. The architecture is based on the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 

One can use the code to build up a transformer step by step, in which the multi-head attention, encoder and decoder layers are constructed in the spirit of `torch.nn.Transformer`. So our version can be regarded as a condensed version of `torch.nn.Transformer`.

 Additionally, users can build the [BERT](https://arxiv.org/abs/1810.04805) model with corresponding parameters.

 Once the transformer is ready, use the following code to test it:
 ```
python main.py 
 ```