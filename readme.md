# Continuous Diffusion for Language

An ongoing collection of experiments in continuous diffusion language modeling, centered on learned Gaussian character embeddings.

Each character is represented by a trainable Gaussian in a low-dimensional space. A transformer denoises sampled embeddings at character-specific noise levels, and Gaussian likelihoods decode its predictions back to characters. An analytic MMD prior encourages the aggregate embeddings to match a standard normal distribution, providing a starting point for generation from noise.

The current reference is [gauss_embed_dlm_mmd_token_time.ipynb](gauss_embed_dlm_mmd_token_time.ipynb). Earlier experiments are in [old/](old/), with evaluations and notes in [experiment_review/](experiment_review/).
