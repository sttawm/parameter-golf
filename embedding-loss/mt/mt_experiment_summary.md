# Experiment: Embedding-Aware Loss for Machine Translation Fine-tuning

**Goal**: Test whether adding an auxiliary embedding loss during fine-tuning improves translation quality, by rewarding predictions that are semantically close to the correct token (synonyms, pronoun choices, date formats) rather than treating all wrong tokens equally.

## Setup
- Model: `t5-small` (60M parameters, pre-trained but not MT-tuned)
- Dataset: IWSLT 2017 En-De (~200K sentence pairs of TED talk translations)
- Evaluation: BLEU, BERTScore, and cross-entropy loss on validation set
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

where `e_hat = E^T * p` is the softmax-weighted average of token embeddings (p is the predicted distribution at each decoder position), and `e_gt` is the ground-truth token's embedding. This rewards the model for placing probability mass on tokens that are semantically close to the correct translation, not just the exact token.

## λ Sweep
{0, 0.1, 0.5, 1.0}, where λ=0 is the CE-only baseline.

## Why These Metrics
- **BLEU**: standard MT metric, measures n-gram overlap with reference translation. Does not reward synonyms — a valid synonym counts as wrong.
- **BERTScore**: computes similarity between model output and reference in embedding space, naturally rewarding synonymous translations. Directly complements BLEU.
- **Cross-entropy**: measures how well-calibrated the model's distribution is. Will catch improvements that BLEU and BERTScore miss. If CE improves but BLEU doesn't, that is itself an interesting finding.

## Why This Task
Translation is rich in near-miss predictions — synonym choices ("big" vs "large"), pronoun alternatives, date formatting — exactly the token types where the embedding loss should help most.

## Embedding Matrix
T5 uses a single shared embedding matrix (`model.shared.weight`) across the encoder input, decoder input, and decoder output projection — no ambiguity about which matrix to use.

## Reference Implementation
See `_embed_aux_loss` in `train_gpt_mlx.py` in the repo — the formula is identical, just apply it at each decoder position instead of each causal LM position.
