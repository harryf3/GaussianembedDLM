**DLM experiment review — September 25, 2026**

**Research constraint clarified:** enforcing the aggregate Gaussian prior is part of the intended model. The next experiment should preserve that prior and its coefficient while testing direct token supervision through the denoiser. The concerns below describe a tradeoff to measure, not a recommendation to abandon the prior.

My assessment is that the main obstacle is the training objective and representation making independent character denoising easy while providing comparatively little useful pressure to learn language. The saved 5M Gaussian model learns that denoising task reasonably well, with a small measurable benefit from context. The older 10M log shows a more severe, separate failure consistent with an almost constant denoiser. Increasing parameter count or extending every run would not address these two situations equally.

This review covers all 20 notebooks, the standalone 10M log, the six available checkpoints, and the old Python implementations/design notes. I also ran fresh CPU diagnostics against the six checkpoints and a paired context intervention against the two 5M checkpoints. I did not retrain models or modify the notebooks/checkpoints. The HTML file named in the IDE context was not present on disk; I inspected the saved notebook outputs instead.

The [diagnostic chart](dlm_diagnostics.png) summarizes the main measurements and is also available as a [PDF](dlm_diagnostics.pdf).

The numerical results are in [checkpoint_metrics.json](checkpoint_metrics.json) and [context_metrics.json](context_metrics.json). Reproduction scripts are [checkpoint_diagnostics.py](checkpoint_diagnostics.py) and [context_diagnostics.py](context_diagnostics.py). The tests use sampled corpus blocks, not a new held-out benchmark. Tensor dimensions and layer counts come from checkpoint shapes; attention-head counts and post-LayerNorm/GELU behavior follow notebook source because checkpoints do not record them. Historical source drift limits exact reconstruction of some older configurations. The 5M reconstructed model reproduces the saved reconstruction diagnostics closely.

**1. The strongest new evidence is the comparison with an independent-character denoiser.**

For each saved codebook, I computed the exact Gaussian-mixture posterior using each noisy character vector and corpus character frequencies. This control uses no neighboring characters, transformer, or training. Its continuous output is the posterior mean of the original sampled embedding; its token prediction is the posterior mode. It is a useful baseline because a model learning context should eventually improve on it where neighboring characters are informative.

For `gauss_embed_dlm_wass_5m.pt`, evaluated on 32 reproducibly selected 64-character blocks:

| Noise step | Transformer MSE | Independent-character MSE | Transformer token accuracy | Independent-character accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.0712 | 0.00010 | 100.0% | 100.0% |
| 100 | 0.1424 | 0.0898 | 100.0% | 100.0% |
| 250 | 0.3469 | 0.3195 | 99.7% | 99.9% |
| 500 | 0.7759 | 0.7582 | 52.3% | 55.5% |
| 750 | 0.9819 | 0.9811 | 14.0% | 17.2% |

MSE is directly comparable between the two continuous estimators. The accuracy comparison also includes a decoder difference: the model rounds its predicted continuous mean, while the analytic control takes the discrete posterior mode. This is intentional for diagnosing both denoising and decoding, but it is not an isolated decoder ablation.

Good low-noise reconstruction therefore does not establish language learning: the character itself remains recoverable without any context. The transformer currently fails to beat the independent-character control in these measurements. This is evidence of limited contextual learning, not proof that it uses no context.

I tested that distinction directly. I independently permuted the batch at each sequence position, preserving each target's noisy vector, clean target, and position while replacing its neighbors with unrelated text. Across 64 blocks and three corruption seeds, the Gaussian 5M model at t=500 achieved **54.96% accuracy with original context versus 53.79% with scrambled context**, and MSE **0.7591 versus 0.7653**. Context helps, but only modestly. These are paired descriptive measurements, not confidence intervals over independent training runs. The point-embedding `dlm_5m.pt` also shows a modest context benefit.

**2. The clean-mixture prior creates a substantial competing objective.**

The current Gaussian loss applies sliced Wasserstein matching to clean sampled `x0`, encouraging the aggregate character mixture to resemble N(0,I). That is a representation regularizer. It is not the terminal diffusion prior needed to start sampling from Gaussian noise.

For the implemented forward process, with a = alpha_bar(T):

\[
E[x_T]=\sqrt{a}\,E[x_0],\qquad
\operatorname{Cov}(x_T)-I=a\big(\operatorname{Cov}(x_0)-I\big).
\]

Here a is approximately 0.00004036. The saved 5M checkpoint's clean covariance RMS error is 0.7784, but the corresponding terminal covariance RMS error is only approximately **0.0000314**. These moments do not by themselves prove full-distribution equality, but they make the logged clean covariance error a poor argument for increasing the prior to fix terminal sampling mismatch. The distinction follows the forward-process construction in [Diffusion-LM](https://arxiv.org/pdf/2205.14217).

There is also a structural conflict. With 84 character means in 128 dimensions, between-character covariance has rank at most 83. If within-character covariance stayed at its initial 0.05I, at least 45 directions would retain variance 0.05 regardless of how the means moved. Matching an isotropic clean distribution pushes variance into within-character noise. Exact equality of a finite Gaussian mixture with a single Gaussian would require identical component distributions, which would erase character information; approximate matching creates a tradeoff rather than making that collapse inevitable.

The checkpoints show that tradeoff developing:

| Checkpoint | Total per-coordinate variance | Between-character variance | Within-character variance |
|---|---:|---:|---:|
| Gaussian Wass, 112K | 0.9792 | 0.5187 | 0.4605 |
| Gaussian Wass, 5M | 0.9865 | 0.4449 | 0.5417 |
| Gaussian Wass epsilon, 112K | 0.9918 | 0.5206 | 0.4712 |
| Point-embedding DLM, 5M | 0.6309 | 0.6163 | 0.0146 |

The Gaussian runs initialize within-character variance at 0.05. In the saved 5M model it has grown to **0.5417**, approximately **55% of all clean embedding variance**. MSE is asked to reconstruct that random draw as well as character identity. At t=500, even an oracle supplied with the true character still has approximately **0.5127 MSE** from unrecoverable embedding randomness, calculated using the actual heterogeneous variances. At t=250 that oracle residual is 0.3184, almost the entire independent-character MSE of 0.3195.

This explains why raw MSE can be poorly aligned with syntax learning. It does not prove that the prior alone caused every failed run; a controlled frozen-codebook comparison is needed. It does strongly argue against treating stronger clean Gaussian matching as an uncomplicated remedy.

**3. Your rounding concern is valid, but the important mismatch is where supervision acts.**

In the current 5M Gaussian notebook, `loss_terms` computes rounding CE on the sampled clean `x0`. That loss updates the codebook but has no path through the transformer. Clean sampled embeddings already round correctly: the saved 5M checkpoint gets 100% clean-rounding accuracy in my diagnostic. Near-zero rounding loss therefore says nothing about how well the transformer predicts contextual token probabilities.

The current 10M source already changes this to CE on `prediction` and divides Gaussian log-density scores by DIM. That is a substantive objective change; its older log cannot be used to evaluate the new implementation. A separate token head reading the denoiser's contextual hidden states would be a useful controlled experiment. Keep an explicit clean-codebook separation constraint if the Gaussian decoder is still used, and train/evaluate the same decoder at generation time.

A subtlety in the proposed SNR weighting: high-noise CE does not require perfect decoding. At zero signal, the optimum is the character-frequency distribution, with nonzero conditional entropy. A continuous posterior mean and a discrete token posterior are different quantities; pushing a single regressed vector to satisfy both can create conflict. A separate head can represent uncertainty without requiring the denoised vector to move toward one token prematurely.

I would compare unweighted auxiliary CE with bounded weighting concentrated on informative intermediate-noise examples. Multiplication by raw SNR can make already-trivial low-noise examples dominate. [Min-SNR](https://arxiv.org/abs/2303.09556) supports investigating conflicting timestep objectives, but its image-diffusion regression weighting is not an established prescription for this auxiliary token CE.

**4. The 10M legacy log shows a different failure from the saved 5M weights.**

At step 9,250 of `gauss_embed_dlm_wass_10m_prior_run.log`, weighted MSE is **0.9071** and aggregate embedding variance is **0.9071**. They also nearly coincide at steps 2,000 and 5,000. The aggregate mean approaches zero, final sample norms approach zero, and generated blocks are repeated spaces or `e`s. Meanwhile the weighted prior improves from 2.337 to 0.118.

Taken together, these observations strongly suggest an almost constant mean predictor. They are stronger evidence than repeated characters at the *start* of reverse diffusion, which can be entirely normal. Even a scalar linear denoiser with the same aggregate variance has approximately 0.664 average expected MSE under this schedule, without learning language. There is a basic signal-extraction failure in that run.

No 10M checkpoint is available to localize it. Deep post-LayerNorm optimization, insufficient input preservation, timestep conditioning, and the changed feed-forward expansion are plausible causes, not established diagnoses. I would require a fixed-codebook, low-noise reconstruction check before spending another long run on that configuration. EMA or a longer budget would not be my first response to thousands of steps at the constant baseline.

The 5M-to-10M source change leaves residual width 128, 12 layers, and 8 heads unchanged, while expanding FF_DIM from 1344 to 3008. It primarily adds position-wise feed-forward parameters; it does not double the width of the attention representation. It also changes prior weight, projection/sample counts, decoding supervision, and sampling steps, so it does not isolate capacity.

**5. Notebook state is masking what experiments actually ran.**

| Artifact | Visible notebook/log | Checkpoint or other evidence |
|---|---|---|
| Gaussian 5M | Training output interrupted after 850 steps | Every saved optimizer state is at **10,000 steps**; downstream reconstruction output closely matches that checkpoint |
| Point DLM “5m” | Current source DIM16; output reports 548,996 parameters and stops after 150 logged steps | Checkpoint is DIM128, **4,980,948 parameters**, optimizer states at 3,939/3,940 |
| Gaussian 10M | Separate log ends at 9,250/25,000, rounding approximately zero | Current source uses CE on predictions; near-zero CE is incompatible with the logged constant predictions on diverse labels; no checkpoint |
| `old/gauss_embed_dlm.ipynb` | Current source implies 517,184 parameters | Saved output reports 56,136 |
| Moment-match vs sliced-Wasserstein notebooks | Different prior source | Saved text outputs are identical, including random samples |
| Fixed vs before-likelihood notebooks | Different snapshots | Same complete 10K training metric stream |

Thus the 5M checkpoint should not be diagnosed as an 850-step model. Equally, several notebook filenames and output blocks should not be counted as independent ablations. Checkpoint optimizer counters establish accumulated optimizer updates, not uninterrupted training provenance or a particular loss configuration. The mixed DLM counters are consistent with a partially interrupted optimizer step; the exact cause is not recorded.

The Gaussian notebooks train on all 42 works, about 5.32 million characters. Their reconstruction diagnostics sample training blocks. At batch 128 × length 64, 10,000 updates represent about 81.9 million character presentations, roughly 15.4 corpus passes. That does not prove sufficient language training, but it is meaningful training rather than only the visible interrupted rerun. An actual held-out split, saved complete configuration/source hash, global step, loss implementation, and immutable run directories are needed to make future comparisons interpretable.

**6. What the older experiments establish.** Cell indices below are zero-based; observations are from stored output unless explicitly described as fresh checkpoint tests.

| Notebook(s) | Finding and interpretation |
|---|---|
| `old/gauss_embed_dlm_memorize.ipynb`, cells 0–5 | The strongest positive control: 1.236M parameters, eight blocks. At 9K and 10K, both printed samples have **0/64 mismatches** against memorized blocks. The basic Gaussian representation and deterministic reverse process can fit and generate sequences. Generalization is untested. |
| `old/fixed_gauss_embed_dlm.ipynb`; `old/gauss_embed_dlm_before_likelihood.ipynb` | Shared 10K training output ends at MSE 0.0414. Perfect low-noise recovery still gives incoherent free samples. Low MSE and clean recovery are insufficient evidence of language. |
| `old/gauss_embed_dlm.ipynb` | Source/output parameter mismatch; stored t=1 recovery only 44.6% despite MSE falling to 0.0462. A moving or shrinking codebook makes absolute MSE misleading. |
| `old/gauss_embed_dlm_gaussian_logprobs.ipynb` | Notebook shows a partial run; checkpoint optimizer shows 10K. Fresh geometry has variance ~1 but covariance error **4.44**: matching total variance permits severe anisotropy. The checkpoint also contains a token-probability buffer absent from current source. |
| `old/gauss_embed_dlm_gaussian_logprobs_weighted.ipynb` | No stored output. Frequency-weighted prior code exists, but it is not a recorded successful or failed experiment. |
| `old/gauss_embed_dlm_gaussian_logprobs_weighted_covariance.ipynb` | 10K steps, clean CE nearly zero, incoherent generation. Fresh checkpoint covariance error improves to 0.274, yet at t=500 its Gaussian rounding accuracy is about 1%; the mixture mean decodes as `5`. Clean geometry and decoding predicted means are separate problems. |
| `old/gauss_embed_dlm_momentmatch.ipynb`; `old/gauss_embed_dlm_slicedwass.ipynb` | Identical saved text outputs despite different prior source. These do not establish an empirical difference between the priors. |
| `old/gauss_embed_dlm_twowaykl.ipynb` | Initial means near zero and standard deviations near one create heavy component overlap. At 5K, variance ~0.995 coexists with CE ~1.49 and poor samples. Normal-looking aggregate variance is not enough. |
| `old/gauss_embed_dlm_twowaykl_sinusoidal.ipynb` | No stored output; also changes mixture weighting. No isolated timestep-encoding result. |
| `old/gauss_embedd_dlm_wass.ipynb` | 112K-parameter model reaches 10K. Fresh geometry is approximately isotropic, covariance error 0.091, but reconstruction trails the independent-character control and samples remain incoherent. |
| `old/gauss_embedd_dlm_wass_eps.ipynb` | 10K steps. Correct epsilon-to-x0 formula nevertheless amplifies errors by dividing by sqrt(alpha_bar). Fresh t=1000 x0 MSE is about **4,508**, consistent with stored ~4,387 and huge reverse norms. A real parameterization/optimization failure, not a general refutation of Gaussian embeddings. |
| `old/gauss_embedd_dlm_wass_v.ipynb` | No stored output. No recorded v-prediction result to compare. |
| `old/gauss_embed_dlm_wass_5m.ipynb` | Saved weights have 10K updates, excellent low-noise character recovery, weak contextual advantage, and large learned within-character variance. |
| `old/gauss_embed_dlm_wass_10m.ipynb` | Current source has prediction CE and temperature changes without saved outputs. Legacy log exhibits mean-prediction collapse and cannot evaluate those new changes. |
| `old/dlm_5m.ipynb` | Source/output/checkpoint mismatch. Saved point-embedding checkpoint has good lexical reconstruction but only modest context gains; it is not a matched, fully documented Gaussian-versus-point control. |
| `old/vldiffnlp.ipynb` | Discrete replacement corruption and different architecture; source also has an apparent timestep-table boundary error. Not a clean continuous-diffusion baseline. |
| `old/vldiffnlp_2.ipynb`, cells 12 and 14 | Training converts epsilon to x0 using **clean `x0_emb` where noisy `noisy_word` is required**; sampling uses the noisy state correctly. Its cosine-LR failure cannot rule out annealing. |
| `old/vldiffnlp_3.ipynb`, cells 5–7 | Trains a rounding head on clean embeddings but sampling/diagnostics use nearest embeddings instead. Diagnostics precede final training execution. Falling MSE/KL do not establish successful generation. |

The old scalar-moment priors also average characters uniformly while training follows corpus frequencies. They can allow common-character geometry to contract while rare characters maintain the aggregate trace. The frequency/covariance variants address parts of this, but their outputs do not establish good language modeling.

`old/model_description.md` already calls for a held-out split and matched point/Gaussian/prior controls. The old Python model implementation depends on missing `gaussian_codebook_loss.py`; a cached bytecode file is present, but the source needed for straightforward reproduction is absent.

**7. My assessment of the seven hypotheses.**

| Hypothesis | Assessment |
|---|---|
| Not enough capacity | Plausible for the smallest runs and later fluency, but not the leading explanation for the 10M constant predictor. Changing latent dimension also changes corruption difficulty; current scaling experiments are confounded. |
| Not enough training | Still plausible for language learning. The 5M checkpoint has 10K updates, not 850. Continue only when held-out intermediate-noise/context metrics improve; the legacy 10M plateau is a reason to change the experiment first. |
| Prior too weak | Weak evidence as the primary problem. Terminal diffusion moments already approach the Gaussian start distribution closely. |
| Prior too strong | Strong concern, more precisely the **clean-prior objective creates competing pressure**. Variance inflation is measured; its exact causal contribution needs a controlled ablation. |
| LR needs annealing | Sensible subsequent experiment, not a demonstrated root cause. The original Diffusion-LM uses decay; the old annealed notebook has a separate training-target bug. |
| EMA | Reasonable later stability/sampling ablation; no saved controlled evidence establishes it as the missing ingredient. Averaging a persistently collapsed denoiser will not supply absent contextual learning. |
| Token rounding / noisy-vector head | High priority. Supervise a contextual token posterior, evaluate on denoiser outputs, retain codebook separability, and avoid weighting away the useful intermediate-noise regime. |

**8. The next experiments I would run, in order.**

1. **Establish a trustworthy baseline with the prior enforced.** Use a held-out-by-work split, saved complete configuration, immutable checkpoints, and fixed evaluation corruption seeds. Keep the Gaussian prior implementation, coefficient, latent dimension, denoiser architecture, and corruption schedule identical across the next comparisons.
2. **Compare clean-only CE against added prediction CE.** Retain clean-codebook separation loss, continuous denoising MSE, and the existing prior. Add CE through the denoiser's predicted clean vectors, using the same Gaussian decoder and temperature in both arms. This directly tests whether token supervision through the generation path helps without changing the representation constraint.
3. **Test a separate contextual token head as the next controlled variant.** A head on denoiser hidden states can represent token uncertainty without forcing the continuous posterior mean toward a single character. Initially use it as auxiliary supervision and evaluate the original continuous sampler/rounder to isolate its effect on the denoiser. Separately compare using the head at the final denoising call; feeding token decisions back into reverse updates would be another sampler change.
4. **Test bounded timestep weighting only after the unweighted comparison.** Report CE and contextual benefit across noise buckets. Very noisy examples should learn uncertain posteriors; raw SNR weighting can overemphasize trivial low-noise examples. Evaluate both token accuracy and distribution calibration.
5. **Tune optimization and scale against contextual metrics while retaining the prior.** Track between-character versus within-character variance and clean separability alongside held-out denoising and generated text. Compare decay/warmup and EMA with the same seed/configuration. If the denoiser is improving on the independent-character baseline, more training and wider attention representations become informative tests.

I would log normalized MSE relative to the constant-mean and independent-character baselines, token CE/accuracy by noise level, paired context benefit, and generated character/bigram/trigram distributions. Held-out reconstruction and language-model-based generation scoring can complement qualitative samples, but good unigram frequencies alone do not establish syntax. Compare samplers and reverse-step counts only after these denoising checks pass.

The original [Diffusion-LM paper](https://arxiv.org/pdf/2205.14217) explicitly discusses easy low-noise token recovery and difficult high-noise tasks, uses a text-oriented schedule and decaying LR, and trains substantially longer than these runs. Those observations support further controlled optimization work; they do not establish a required step count for character-level Shakespeare. Its [released transformer implementation](https://raw.githubusercontent.com/XiangLi1999/Diffusion-LM/main/improved-diffusion/improved_diffusion/transformer_model2.py) also separates embedding dimension from transformer hidden width, unlike the current notebooks.

The positive result to preserve is the successful eight-block generator. The next success criterion should be a clear, reproducible contextual advantage on held-out data and improved free generation as the dataset grows. Another smaller scalar loss or cleaner aggregate Gaussian is not sufficient evidence that this is happening.
