# Toward coherent Shakespeare text

This follow-up compares the saved experiments, then tests two changes on the same 40-work training set. *Macbeth* and *The Tempest* are held out by work. The new runs use 64-character blocks, a 128-wide transformer, four attention heads, fixed validation examples, and saved configuration and optimizer state. All reported validation scores use the held-out works.

## What the existing experiments establish

| Source | Useful result | Limit seen in saved runs |
|---|---|---|
| Eight-block Gaussian memorization test | Continuous reverse diffusion can reproduce its training blocks exactly. | It does not test language generalization. |
| 5M Gaussian Wasserstein run | A large denoiser recovers characters at low noise; context helps slightly. | Its continuous MSE rewards recovery of irreducible Gaussian draw noise, and generated text remains incoherent. |
| 10M Wasserstein log | More feedforward parameters alone are insufficient. | The logged MSE and embedding variance suggest a near-constant denoiser. This log does not match the latest source objective. |
| 16D CE plus analytic MMD | Stable Gaussian codebook with direct token supervision; low-noise reconstruction is strong. | The original 112K denoiser trails an independent-character classifier. |
| 128-wide CE plus analytic MMD | At 10K steps, its denoiser beats the independent-character classifier modestly at t=250 and 500. | Free 64-character samples remain scrambled, even with 1,000 reverse steps and eta=1. |

The earlier [experiment review](review.md) documents the older checkpoint comparisons and source-drift caveats. The most useful pieces to carry forward are the Gaussian codebook, analytic MMD prior, 128-wide contextual transformer, direct token supervision, and exact Gaussian posterior-mean calculation. The new experiments below test whether a separate token head and a different generation order can turn those pieces into language.

## New controlled runs

| Run | What changes | Parameters | Held-out result at 3,000 steps |
|---|---|---:|---|
| Contextual Gaussian diffusion | Adds a separate token head; its probabilities drive posterior-mean DDIM. Retains Gaussian embeddings, analytic MMD with weight 10, and the scaled denoiser. | 1,213,780 | CE 0.304 at t=250 and 2.761 at t=500; independent-character CE is 0.279 and 2.756 on the same blocks. Samples remain incoherent. |
| Causal point-embedding control | Predicts the next character from preceding text, with no Gaussian codebook. | 823,124 | At 10K, held-out CE 1.658 on the training-selection windows; free samples have partial sentence and dialogue structure. |
| Causal Gaussian model | Same causal task and split as the point control; uses sampled 16D Gaussian embeddings and the five-bandwidth analytic MMD prior with weight 10. | 817,236 | At 10K, held-out CE 1.706 on the training-selection windows. Samples contain stage directions and dialogue-like phrases, with remaining invented words and grammar errors. |
| 128-character Gaussian causal extension | Fine-tunes the best Gaussian model with twice the context length, retaining its codebook and prior. | 825,428 | On paired held-out targets in the second half of each block, CE 1.765 versus 1.776 for the original 64-character model. Sample quality is not clearly better. |

The two CE tasks differ: diffusion predicts the clean character from a noisy version of that character and its neighbors; causal models predict the next character without seeing it. Their CE numbers should not be compared across tasks. The Gaussian causal validation and generation use embedding means, while training draws Gaussian embeddings; this is an evaluation of the chosen deterministic inference path. The point and Gaussian causal models use the same sampling temperature and top-k settings.

The paired [neighbor-shuffle probe](coherence_context_probe.json) at diffusion step 3,000 holds each target's noisy vector fixed. At t=500, shuffling surrounding vectors raises held-out CE from 2.836 to 2.860, confirming a small contextual effect. That small gain has not produced coherent free diffusion samples.

The causal Gaussian model at step 3,000 generated a passage including “and said,” “[They exit.],” and “Enter …”; the surrounding text still has malformed words. These are the first free samples in this experiment family with recognizable Shakespeare-like structure beyond isolated words. The result identifies generation order and next-character supervision as promising factors; it does not establish which one is decisive without further ablations.

## Final check on fresh held-out windows

The [final metrics](coherence_final_metrics.json) use 512 new windows from *Macbeth* and *The Tempest*, three fixed 256-character samples per model, temperature 0.65, and top-k 12. The Gaussian validation and generation path uses embedding means; training draws from those Gaussians.

| Best 10K checkpoint | Fresh-window CE | Known training-corpus words in samples | Distinct 4-grams in samples | Exact 64-character training matches |
|---|---:|---:|---:|---:|
| Point causal | **1.660** | 87.6% | 80.1% | 0/582 windows |
| Gaussian causal + MMD | 1.713 | **88.8%** | 78.0% | 0/582 windows |

The word metric is a rough lexical check, not a language-quality score. The 64-character overlap check rules out exact long copying in these samples, not shorter memorized phrases. Point embeddings give the better held-out CE. The Gaussian version remains viable: its full training-corpus MMD² is 0.00586, within-component variance is 0.308, covariance RMS error is 0.147, and covariance eigenvalues span 0.674–1.203. It retains character information while approximately matching the aggregate normal prior.

One Gaussian sample begins: “DUKE, [to Sentonio] / O, my father, who comes, the phart of thy heart. / BENEDICK Thou shalt still …” This is locally Shakespeare-like and partly readable, with obvious grammar and spelling errors. The point model is similarly imperfect. Neither supports a claim of sustained coherent dialogue.

## Reproduction and next test

- `python3 coherent_gauss_dlm.py --resume --steps 3000` reproduces or extends the contextual diffusion run. Checkpoints and history are in `checkpoints/coherent_gauss_dlm/`.
- `python3 coherent_char_baseline.py --resume --steps 10000` handles the point control.
- `python3 coherent_gaussian_causal_lm.py --resume --steps 10000` handles the Gaussian causal run. All three scripts save `best.pt`, `latest.pt`, and a JSONL history in their own checkpoint directories.
- `python3 coherent_gaussian_causal_lm_long.py --steps 3000` reproduces the longer-context fine-tuning from the best Gaussian causal checkpoint.
- `python3 generate_coherent.py --model gaussian --prompt 'DUKE: ' --seed 2` samples the best Gaussian checkpoint; use `--model point` or `--model gaussian-long` to compare alternatives.
- `python3 -m experiment_review.coherence_context_probe` repeats the paired context intervention.
- `python3 -m experiment_review.coherence_final_eval` repeats the fresh-window and sample comparison.

The next research step is to test a stronger causal or blockwise language prior *within* the reverse diffusion process. The causal results establish that the corpus and this training budget can produce recognizable language; the diffusion denoiser still yields only small context gains. Such a hybrid should be compared against the independent-character control and this 10K causal baseline on held-out works before claiming a generative improvement.
