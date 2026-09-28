# Per-character noise in the Gaussian diffusion LM

The [token-time experiment](../coherent_gauss_dlm_token_time.py) keeps the 16D diagonal Gaussian embeddings, analytic five-bandwidth MMD prior (weight 10), bidirectional 128-wide six-layer transformer, 64-character blocks, and contextual token CE. It changes the Gaussian forward process so each character has its own timestep, and gives the transformer a separate time embedding at each position. There is no causal attention or next-character objective.

Training draws 20% synchronized-timestep blocks as before. In the other 80%, timesteps vary by position: 15% of positions are nearly clean (t=1), 15% are at maximum noise (t=1000), and the rest have independently uniform timesteps. CE is still computed for every position. This is a controlled test of whether readable neighbors help recover noisy characters, not yet an optimized diffusion training objective.

The run started from random initialization, trained for 3,000 steps on 40 works, and held out *Macbeth* and *The Tempest*. It has 1,213,780 trainable parameters, the same as the previous contextual-head diffusion model. The checkpoint and training history are under `checkpoints/coherent_gauss_dlm_token_time/`. [Evaluation code](token_time_eval.py) and [raw metrics](token_time_metrics.json) reproduce the final readout.

At t=500, three fixed held-out probes each use 256 blocks and eight targets per block. The table averages their cross-entropies; shuffling changes only neighbors, keeping each target's own noisy embedding and timestep fixed. For mixed-noise neighbors, their timestep labels move with their vectors.

| Neighbors of t=500 target | Original CE | Shuffled CE | Context gap |
|---|---:|---:|---:|
| Nearly clean (t=1) | 2.699 | 2.846 | 0.147 |
| Independently mixed timesteps | 2.732 | 2.800 | 0.068 |
| All t=500 | 2.751 | 2.763 | 0.013 |

The independent-character Gaussian classifier scores 2.747 on these targets. Thus mixed-noise training induces contextual dependence when some neighbors are clearer, while the fully synchronized case remains almost independent-character denoising. The shuffle gaps are descriptive across fixed probes of one training run; they are not confidence intervals across training seeds. At t=250 the independent-character classifier is still better than the learned model, and at t=750 the contextual gain remains small.

Positionwise posterior-mean DDIM starts every character at t=1000 and advances positions at different rates. Neither 100 nor 250 reverse steps produced coherent free 64-character text. A sample after 3,000 updates is `sl,tul.yhrohdb,e\ntIeosls.o e\na\n n E,.C\nHlga Inaeeye'\n ,lettsd e `. The result supports per-character noise as a way to make the denoiser use context, but it does not show that the current posterior-mean reverse process can turn that conditional skill into coherent unconditional text.

The next controlled test should isolate reverse decoding: compare posterior-mean updates with a component-sampling or rounding update using this *same* checkpoint, and measure both sample quality and the noise/geometry of the resulting states. A further training change could target heavily noised positions more strongly, but that should be a separate ablation.
