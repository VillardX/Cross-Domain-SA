# Cross-Domain-SA
A BERT-based model trying to transfer knowledge from source to target domain.

Baseline is a MLP injected after the [CLS] token embedding of BERT, trained by source labeled data.

Two strategies are taken to improve the baseline.

- Post training

  1. Reconstruct the structure of NSP. Namely, each input contain two sentences. One from SOURCE and one from TARGET is labeled 'is_mix'=1, while both from TARGET is labeled 'is_mix'=0. It is a classification problem with 2 categories.
  2. Reconstruct the structure of MLM. Namely, ONLY mask the tokens in target domain.

The above two strategies aim to make vanilla BERT focus more on source domain and target domain in our task, espcially the target domain, instead of the general corpus from all kinds of domains, which is used to pre-train the vanilla BERT.
  
- Adversarial training

  Here the structure of DANN is taken into consideration. The thought originated from GAN. The core of it is the gradient reversal layer, which changes the min-max optimization problem to an E2E loss optimization problem with simple BP process. To be more detailed, it is consisted of two units.
   
   1. sentiment classifier: trained by labeled source data.
   2. domain discriminator: check each input whether it is comes from source domain or target domain. The domain discriminator aims to clearly classify, while the BERT as feature extractor here intends to confuses the domain discriminator, aiming to get DOMAIN-INVARIANT features of both domains , which we call adversarial training. By acquiring DOMAIN-INVARIANT features, it is possible to transfer the knowledge from source domain. Via GRL, the domain discriminator and feature extractor can be trained together instead of min-max optimization. 
  
  The above two strategies are trained in an E2E way, aiming to transfer the sentiment knowledge from source to domain.
