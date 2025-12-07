# Transformer

The Transformer is a neural network architecture designed to model relationships in sequences without using recurrence (RNNs). It relies entirely on attention mechanisms to transform input sequences into richer and more meaningful representations — hence the name Transformer.

_(The Google team who invented it is also named as team Transformer)_

## History

In 2015, researchers began using additive attention and multiplicative (Luong) attention alongside RNNs.

In 2017, the Google "Transformer" team introduced self-attention and completely removed RNNs from the architecture. They also introduced masked self-attention in the decoder for autoregressive generation.

Later, OpenAI extended the decoder part into a decoder-only Transformer, removing the encoder and cross-attention entirely.

## Brief Architecture

![alt text](image.png)

_source - Attention is All You need_

### Attention

Attention is the core idea of the Transformer.
Each token is projected into three vectors:

Query (Q)

Key (K)

Value (V)

The model computes how much each token should “attend to” every other token by comparing queries to keys, and then mixing the values accordingly.

Multi-Head Attention repeats this process across several attention heads, allowing the model to capture different types of relationships in parallel.

![alt text](transform20fps.gif)

_source - [Google research blog](https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/)_

### Tokenization

At it's core it is way of representing words in compact form such that computation, representation and retreival become efficient than it's original form.

Common tokenization strategies:

- Character-level tokens

- Word-level tokens

- Subword tokens such as:

  - BPE (Byte Pair Encoding)

  - WordPiece

  - SentencePiece

Subword tokenization is the most widely used today because it handles rare words efficiently.

### Positional Encoding

Since transformer process tokens in parallel as oppesed to RNN, hence to keep track of position of each token.

This can be achieved with serval ways,

- Absolute positional Encoding
- Relative positional encoding
- RoPE (Rotary Positional Embedding)

RoPE (Rotary Positional Embedding) uses rotation in a multi-dimensional space to encode relative positions between tokens.
It does not rely on a full repeating sinusoidal cycle; instead it rotates Q and K vectors using frequency-based rotation matrices.

### Encoder

Encoder is designed to find deep context meaning which enable it in NER, QA, context finding tasks.

The bidirectional attention allows each token to attend to all tokens in the sequence, capturing context from both directions. The encoder block contains multi-head attention, Layer Normalization, Feed forward along with residual connections.

An example of such a model is BERT, which stands for bidirectional
encoder representations from transformers.

### Decoder

The decoder uses masked self-attention so it cannot see future tokens during training. It also uses an autoregressive mode to predict the next token.

In the original Transformer, each decoder block contains:

- masked self-attention

- cross-attention (attends to encoder output)

- feed-forward network

- residual connections and layer normalization

However, GPT-style models remove the cross-attention and use only stacked masked self-attention blocks. These blocks build contextual representations but they are not an encoder.

## Reason it's so useful

Transformers offer several key advantages:

- Massive parallelism → faster training than RNNs

- Global receptive field → every token sees the entire sequence

- Scales extremely well with data and compute

- General-purpose → works for text, vision, audio, and multimodal tasks

- Excellent for transfer learning, enabling pretraining + fine-tuning

## Conclusion

The Transformer architecture revolutionized modern deep learning by providing a flexible, scalable, and powerful way to process sequences. Today it forms the foundation of almost all state-of-the-art language, vision, and multimodal models.

### Reference

_1 - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)_

_2 - [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/pdf/1508.04025)_

_3 - [Neural Machine Translation By Jointly Learning To Align And Translate](https://arxiv.org/pdf/1409.0473)_

_4 - [Transformer: A Novel Neural Network Architecture for Language Understanding]
(https://research.google/blog/transformer-a-novel-neural-network-architecture-for-language-understanding/)_
