# Cross-Domain-SA
A BERT-based model trying to transfer knowledge from source to target domain.

Baseline is a MLP injected after the [CLS] token embedding of BERT, trained by source labeled data.

Here take two points to improve the baseline.

- Post training

  1. Reconstruct the structure of NSP. Namely, each input contain two sentences. One from SOURCE and one from TARGET is labeled 'is_mix'=1, while both from TARGET is labeled 'is_mix'=0. It is a classification problem with 2 categories.
  2. Reconstruct the structure of MLM. Namely, ONLY mask the tokens in target domain
  
- Adversarial training

  Here the structure of DANN is taken into consideration. The thought originated from GAN. The core of it is the gradient reversal layer, which changes the min-max optimization problem to an E2E loss optimization problem with simple BP process.
