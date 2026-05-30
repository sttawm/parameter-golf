# Experiment: Embedding-Aware Loss for BERT Fine-tuning on LAMA

**Goal**: Test whether adding an auxiliary embedding loss during fine-tuning improves factual recall accuracy on the LAMA benchmark.

## Setup
- Model: `bert-base-uncased` (full fine-tune, all layers)
- Dataset: LAMA T-REx (~34K examples)
- Evaluation: top-1 accuracy at `[MASK]` positions
- Infrastructure: Google Cloud (or similar), single GPU

## The Loss

```
L = L_CE + λ * L_embed
```

Two variants of `L_embed`, run L2 first and cosine if time allows:

```
L_embed_l2     = mean( ||e_hat - e_gt||^2 )
L_embed_cosine = 1 - cosine(e_hat, e_gt)
```

where `e_hat = E^T * p` is the softmax-weighted average of token embeddings (p is the predicted distribution at the `[MASK]` position), and `e_gt` is the ground-truth token's embedding. This rewards the model for placing probability mass on tokens that are semantically close to the correct answer, not just the exact token.

## λ Sweep
{0, 0.1, 0.5, 1.0}, where λ=0 is the CE-only baseline.

## Embedding Matrix
BERT uses tied embeddings — the input word embedding matrix and the output decoder matrix are the same tensor (`bert.embeddings.word_embeddings.weight`). Use that one.

## Reference Implementation
See `_embed_aux_loss` in `train_gpt_mlx.py` in the repo — the formula is identical, just apply it at `[MASK]` positions instead of all positions. The `embed_loss_l2` flag switches between the two variants.
