# Anima Compact Style LoRA Training — Experimental Findings and Configuration Report

## 1. Objective

The objective of this experiment series is to develop a **compact Anima style LoRA** that can reproduce an artist's distinctive visual style while preserving the base model's ability to generate accurate anatomy, varied poses, and unseen characters.

The primary goals are:

- Strong artist-style fidelity.
- Accurate reproduction of lineart and painterly rendering.
- Preservation of hair, skin, lighting, and color characteristics.
- Stable fine details such as eyes, eyelashes, fingers, and nails.
- Good generalization to unseen characters and poses.
- Minimal artifacts such as stray lines, noisy lineart, melting, malformed anatomy, and unstable facial details.
- Small LoRA size, approximately **33 MB**.
- Avoid unnecessary increases in network capacity when dataset quality/composition is the actual bottleneck.

A recurring theme throughout the experiments is that **style fidelity, local detail fidelity, and anatomical/generalization stability compete for limited LoRA representation capacity**.

---

# 2. Initial Findings and General Training Behavior

The initial experiments established several important relationships between training hyperparameters and visual output.

## 2.1 Learning Rate

Learning rate was one of the strongest controls for the balance between style strength and artifacting.

### LR ≈ 4.0e-4

Observed behavior:

- Very strong style reproduction.
- High similarity to the artist.
- More fine-detail artifacts.
- More stray lines.
- More noisy lineart.
- More malformed fingers.
- Increased melting in difficult areas.

Interpretation:

> The LoRA learns aggressively enough to reproduce strong style characteristics, but fine details become less stable.

### LR ≈ 3.5e-4

Observed behavior:

- Large reduction in artifacts.
- Strong style fidelity retained.
- Cleaner lineart.
- Better balance between style strength and stability.

This became an important candidate operating point.

### LR ≈ 3.3e-4

Observed behavior:

- More stable than 3.5e-4.
- Cleaner fine details.
- Slightly weaker style reproduction.
- Still retained strong stylistic characteristics.

### LR ≈ 3.0e-4

Observed behavior:

- Cleaner linework.
- Less melting.
- More stable fine details.
- Noticeably weaker style strength.

### Conclusion

There is a clear tradeoff:

```text
Higher LR
    ↓
Stronger style imprint
    ↓
More aggressive fine-detail learning
    ↓
Greater artifact risk

Lower LR
    ↓
More stable details
    ↓
Cleaner anatomy/lineart
    ↓
Weaker style imprint
```

The mature configuration therefore settled around **3.8–3.9e-4**, where the style remained strong without the artifact level seen at 4e-4.

---

# 3. Weight Decay Findings

Weight decay had a strong effect on fine-detail stability.

## WD = 0.07

Observed:

- More fine detail.
- Stronger lineart/style reproduction.
- More artifacts.
- More melting.
- More unstable fine features.

## WD = 0.09

Observed:

- Slightly more detailed than WD 0.1.
- Slightly more artifact-prone.

## WD = 0.10

Observed:

- Best overall stability.
- Fewer stray lines.
- Strong style reproduction.
- Some fine detail suppressed compared with lower WD.

## Very high WD

Experiments around approximately 1.0–1.13 were excessively regularizing.

### Conclusion

**WD ≈ 0.1** became the preferred value.

The behavior suggests that WD is acting as an important stabilizing constraint against excessive fine-detail adaptation.

---

# 4. Network Dimension Experiments

Network dimension was tested to determine whether additional LoRA capacity improved style representation.

## Earlier larger-rank experiments

Higher dimensions, including around dim 20, demonstrated:

- Strong style reproduction.
- Greater ability to encode detail.
- Less desirable pose diversity.
- More fixed-looking output compositions/postures.
- Increased concern about memorization-like behavior.

This established an important principle:

> More LoRA capacity does not automatically produce a better style LoRA.

For a style LoRA, excessive capacity can reduce the base model's freedom.

---

# 5. Conv Dimension and Alpha Experiments

Conv parameters were investigated separately from standard network rank.

The mature configuration eventually settled at:

```text
conv_dim   = 12
conv_alpha = 10
```

## Conv alpha progression

Increasing Conv alpha from approximately 6 → 8 → 10 improved:

- Fine feature representation.
- Local rendering.
- Artist-specific detail.

However, increasing local transformation strength too aggressively also increased artifact risk.

## Conv dimension 16 / alpha 8

Observed:

- Better pupils and some small details.
- Some additional artifacts.

## Conv dimension 20 / alpha 10

Observed:

- Cleaner overall images.
- Slightly weaker style.
- More training may have been required.
- Did not fundamentally solve the fine-detail artifact problem.

### Conclusion

The best balance became:

> **Conv 12 / 10**

rather than increasing Conv capacity further.

---

# 6. Warmup and Scheduler

Cosine scheduling consistently performed well.

The preferred setup became:

```text
lr_scheduler = cosine
warmup_ratio  = 0.05
```

An experiment using approximately 9% warmup combined with increased crop repeats produced tighter/lower losses, but the visual result was worse:

- Less texture.
- Less toning.
- Less desirable fine-detail representation.
- Greater instability in some details.

This reinforced an important finding:

> Lower training loss does not necessarily correspond to better visual style reproduction.

---

# 7. Loss Interpretation

Loss curves were repeatedly found to be insufficient as a standalone measure of LoRA quality.

Two models can have nearly identical loss curves while producing noticeably different:

- lineart,
- eyes,
- fingers,
- skin shading,
- brush strokes,
- lighting,
- artifact patterns.

Loss is averaged over the training objective and therefore does not directly measure:

- perceptual style fidelity,
- anatomical correctness,
- local feature stability,
- artifact severity,
- generalization.

Therefore, visual checkpoint evaluation remains essential.

---

# 8. Resolution Experiments

Training at 1280 resolution produced:

- Better style fidelity.
- Better high-frequency representation.
- More fine-detail artifacts at aggressive learning rates.

At 1280:

### WD 0.07

Improved fine detail and lineart but caused:

- more artifacts,
- more melting,
- increased instability.

### LR 3.5e-4

Substantially reduced these artifacts.

### LR 3.3e-4

Produced highly accurate style with approximately 3/10 evaluated images showing occasional fine-detail artifacts.

The overall conclusion was:

> Higher resolution provides useful high-frequency information, but also increases the opportunity for the LoRA to learn unstable details.

---

# 9. Targeted Feature Crops

One of the most successful dataset improvements was the introduction of **targeted crops**.

These are not random crops. They deliberately emphasize high-frequency regions such as:

- eyes,
- pupils,
- eyelashes,
- hair shine,
- hair shading,
- skin texture,
- fine rendering,
- color toning.

An example dataset increased from approximately:

```text
386 original images
+
66 targeted crops
```

The cropped model demonstrated:

- Better pupils.
- More stable hair shading.
- Better skin shading.
- Better representation of high-frequency features.

The best checkpoint was sometimes found before the final training step, demonstrating that visual convergence and loss convergence do not necessarily occur simultaneously.

---

# 10. Why Crops Are Particularly Important for Anima

Anima already possesses a strong prior for common anime structures.

Therefore, simply showing more ordinary full-body images does not necessarily provide the most useful information.

The LoRA instead benefits from seeing **high-resolution examples of how this particular artist renders familiar structures**.

The important distinction is:

> The LoRA does not necessarily need to learn what an eye is. It needs to learn how this artist renders an eye.

This is especially important when the artist's rendering differs substantially from Anima's prior.

---

# 11. Current Artist Dataset Characteristics

The current artist has:

- Stylized anime artwork.
- Strong lineart.
- Detailed skin shading.
- Distinctive lighting tones.
- Brush-like backgrounds.
- Unique hair rendering.
- Characteristic hair shine.
- Distinctive palette.
- Unusual eye design.
- Distinctive eyelash texture.
- Artist-specific skin texture.
- Characteristic scene lighting.

However, the artist heavily censors NSFW artwork.

As a result:

### Well represented

- Face rendering.
- Hair.
- Eyes.
- Eyelashes.
- Skin shading.
- Lighting.
- Palette.
- Brushwork.
- Painterly rendering.

### Poorly represented

- Uncensored anatomy.
- Fingers/nails in some contexts.
- Certain fine anatomical structures.

A small regularization dataset of fewer than 10 images was therefore introduced.

---

# 12. Regularization Dataset

The small tagged regularization set (without artist trigger token) was found to have **minimal impact on overall style**.

However, it improved:

- General anatomy.
- Fine anatomical stability.
- Preservation of base-model behavior.

Its role is therefore best understood as an **anchor**, rather than a source of style information.

Conceptually:

```text
Artist dataset
     ↓
learn artist style

Regularization dataset
     ↓
preserve general model behavior
```

This is particularly valuable because the artist dataset is biased toward censored artwork.

The regularization set should therefore remain.

---

# 13. Face Crop Tradeoff

A major discovery was that increasing face crops improves the artist's eyes, but can negatively affect other fine features.

The artist's eyes are unusual:

- Painterly blending.
- Low contrast.
- Pupil is only slightly darker than the eye base.
- Pupil/iris boundary is subtle.
- The pupil can visually merge into the surrounding painterly rendering.

When face crops were reduced:

> The pupil became less distinct and could merge into a painterly blob.

Increasing face crops improved:

- pupil visibility,
- eye rendering,
- facial fidelity.

However, excessive face-crop emphasis can reduce representation allocated to:

- fingers,
- nails,
- anatomy,
- other fine features.

Therefore:

> The objective is not simply to maximize face crops. The goal is to maximize **information density per crop**.

---

# 14. Style Evolution and Dataset Weighting

Experiments with artists whose styles changed over time demonstrated that a dataset can contain multiple stylistic sub-distributions.

For one artist, approximately:

- 70% represented an older, highly rendered style.
- 30% represented a newer, flatter style.

The older work contained:

- richer backgrounds,
- smoother skin toning,
- bloom,
- reflective highlights,
- stronger rendering.

The newer work contained:

- stronger lineart,
- flatter rendering,
- reduced detail.

Later checkpoints naturally shifted toward the newer style.

Increasing the weight of the older images moved the LoRA back toward the richer rendering while retaining cleaner lineart.

### General conclusion

When an artist's style evolves:

> **Weight the desired era rather than simply deleting the other era.**

---

# 15. Mixed Monochrome and Colour Datasets

When an artist has a predominantly monochrome dataset with a smaller richly colored subset, the smaller color subset can be repeated to prevent the monochrome majority from suppressing:

- saturation,
- color interactions,
- richer shading,
- color toning.

However, excessive repetition can cause oversaturation.

This is fundamentally a dataset-statistics problem rather than an optimizer problem.

The preferred solution is therefore:

- adjust repeats,
- add representative lower-saturation examples,
- preserve diversity,

rather than immediately changing LR or WD.

---

# 16. Complex Backgrounds

Complex environment/multi-subject images can contribute useful style information:

- lighting,
- color interactions,
- atmosphere,
- composition,
- brushwork.

However, they should not dominate a character-style LoRA.

A useful approximate composition is:

```text
70–85% character-focused
10–20% alternative compositions / lighting
≤10% large environmental or multi-subject scenes
```

unless the environment itself is a defining part of the artist's style.

Targeted crops from complex scenes can retain useful character/detail information without allowing the environment to dominate training.

---

# 17. Mature Baseline Configuration — Config A

The experiments eventually converged on the following compact baseline:

```toml
network_dim   = 12
network_alpha = 12

conv_dim      = 12
conv_alpha    = 10

optimizer     = AdamW8bit

learning_rate = 0.00039
lr_scheduler  = "cosine"

weight_decay  = 0.1
betas         = "0.9,0.9985"

warmup_ratio  = 0.05

batch_size    = 4
resolution    = 1024

mixed_precision = "bf16"

weighting_scheme = "logit_normal"

network_train_unet_only = true
```

Other established settings include:

```text
max_train_steps = 1000
gradient_checkpointing = true
cache_latents = true
cache_latents_to_disk = true
xformers = true
```

The LoRA remains approximately **33 MB**.

---

# 18. Config A — Characteristics

Config A:

```text
Network: 12 / 12
Conv:    12 / 10
```

is currently the mature compact baseline.

Its strengths are:

- Strong overall style fidelity.
- Good balance of style and base-model flexibility.
- Good anatomical stability.
- Good generalization.
- Controlled artifact levels.
- Good hair rendering.
- Good skin shading.
- Good lighting.
- Compact model size.

Its limitation is that some extremely fine artist-specific details remain unstable, particularly:

- pupil/iris distinction,
- certain fine anatomical features,
- very localized high-frequency details.

---

# 19. Config B — Network Dimension 14 Experiment

The next controlled experiment changed only network dimension:

### Config A

```text
network_dim   = 12
network_alpha = 12
conv_dim      = 12
conv_alpha    = 10
```

### Config B

```text
network_dim   = 14
network_alpha = 14
conv_dim      = 12
conv_alpha    = 10
```

The purpose was to determine whether the remaining fine-detail problems were caused by insufficient LoRA capacity.

---

# 20. Config A vs Config B — Loss

The loss curves were extremely similar.

At step 925:

```text
A: 0.1004
B: 0.1000
```

This represents only approximately a **0.4% reduction**.

Therefore:

> The additional rank did not fundamentally change average optimization performance.

This is important because the visual differences are therefore unlikely to be explained by simply saying "B trained better."

Instead, the extra rank appears to change **representation allocation**.

---

# 21. Config B — Visual Improvements

The corrected observations show several genuine improvements.

### Painterly brush strokes

B captures more of the artist's painterly brush strokes.

### Eyelashes

B produces:

- more detailed eyelash brush strokes,
- more artist-specific eyelash rendering.

### Hair and eyelashes + lighting

B better captures the interaction between:

- hair,
- eyelashes,
- highlights,
- light,
- local shading.

This is an important finding because it indicates that the additional network capacity is learning **useful stylistic information**, not merely producing random artifacts.

### Skin

B's skin shading is:

- slightly more detailed,
- less flat,
- better at capturing some sketch-like aesthetic lines.

---

# 22. Config B — Remaining Problems

Despite the increased rank, some fine details remain unstable.

Most importantly:

### Pupil / iris distinction

The pupil/iris boundary remains unstable.

This is particularly significant because the eyes are one of the most artist-specific features in the dataset.

Therefore:

> Increasing network dimension does not automatically solve the eye problem.

### Uncensored anatomy

B can produce more detail in the finer uncensored anatomical regions.

However, this detail is less stable and can compete with the learned artist style.

Config A tends to produce **less detail but more reliable anatomical structure**.

Config B can therefore be characterized as:

> **More expressive, but not necessarily more accurate.**

---

# 23. Revised Interpretation of Network Dimension

The experiments now suggest that network dimension affects more than simply "how much style" the LoRA learns.

A better model is:

```text
Higher network dimension
        ↓
More representational freedom
        ↓
More high-frequency stylistic information can be encoded
        ↓
BUT
        ↓
More freedom to encode poorly constrained information
```

Therefore, additional capacity can produce both:

### Desired

- brush strokes,
- eyelashes,
- sketch-like skin lines,
- hair/light interactions,
- intricate rendering.

### Undesired

- unstable anatomy,
- incorrect fine details,
- artifacts,
- unstable pupil/iris structures.

---

# 24. Important Distinction: Detail vs Accuracy

One of the strongest findings from Config B is:

> **More detail does not necessarily mean more accuracy.**

For example:

```text
Config A
    ↓
less anatomical detail
    ↓
more anatomically stable
```

versus:

```text
Config B
    ↓
more anatomical detail
    ↓
less stable / less accurate
```

This distinction is critical when evaluating LoRA quality.

A style LoRA should not be optimized simply for maximum visible detail.

The desired target is:

> **Artist-specific detail + structural correctness + generalization.**

---

# 25. Frequency-Dependent Capacity Hypothesis

The current experiments suggest that Config B is particularly effective at a middle/high-frequency level.

### Successfully improved by B

- Eyelash brush strokes.
- Hair brushwork.
- Hair/light interaction.
- Eyelash/light interaction.
- Skin sketch lines.
- Painterly local rendering.

### Still unstable

- Pupil/iris distinction.
- Some extremely localized anatomical details.
- Some poorly represented uncensored structures.

This suggests that there may be multiple levels of high-frequency representation:

```text
Global style
    ↓
Skin / lighting / palette
    ↓
Painterly rendering
    ↓
Hair / eyelashes / brush strokes
    ↓
Extremely localized structures
    ↓
Pupil / iris boundary
    ↓
Tiny anatomical structures
```

Config B clearly improves some of the middle levels, while the very finest structures remain difficult.

---

# 26. Why the Pupil/Iris Problem Is Probably Not Simply Rank

The fact that B has more capacity but still produces unstable pupil/iris boundaries is strong evidence against a simple:

> "dim 12 is too small."

explanation.

The problem is likely related to the training signal itself.

The artist's pupil is:

- low contrast,
- subtly differentiated from the iris,
- painterly,
- spatially tiny,
- sometimes visually blended.

The model can therefore learn:

> "Artist uses a painterly eye with soft internal rendering."

without reliably learning:

> "The pupil must remain distinguishable from the iris."

This is a **representation/data problem**, not necessarily a raw capacity problem.

---

# 27. Current Hypothesis: Cleaner Data Would Make Higher Rank More Valuable

The strongest current hypothesis is:

> **Increasing network dimension makes the LoRA more sensitive to fine-grained information. A cleaner and more consistently represented dataset would allow the additional capacity to be used more effectively.**

Conceptually:

```text
Dim 12 + current dataset
    ↓
Strong style
    ↓
Relatively constrained details


Dim 14 + current dataset
    ↓
More intricate style
    ↓
More painterly information
    ↓
More sensitivity to poorly represented details


Dim 14 + cleaner dataset
    ↓
More intricate style
    ↓
More accurate fine detail
    ↓
Potentially superior result
```

The difficulty is that the current artist's dataset may not realistically be capable of becoming sufficiently clean.

The artist's censorship itself creates an unavoidable limitation in anatomical supervision.

---

# 28. Config A vs B — Current Overall Assessment

| Category | Config A | Config B |
|---|---|---|
| Overall style | Excellent | Excellent |
| Style stability | **Higher** | Slightly lower |
| Painterly brushwork | Good | **Better** |
| Eyelash brushwork | Good | **Better** |
| Hair rendering | Good | **More intricate** |
| Hair/light interaction | Good | **More accurate** |
| Skin shading | Good | **More detailed** |
| Sketch-like skin lines | Some | **Better** |
| Lighting | ≈ equivalent | ≈ equivalent |
| Pupil/iris distinction | Unstable | **Still unstable** |
| Fine anatomy | **More stable** | More detailed but less stable |
| Artifact sensitivity | Lower | **Higher** |
| Generalization | **Safer** | Requires more validation |
| Capacity | Lower | **Higher** |
| LoRA size | Smaller | Slightly larger |
| Loss | ≈ same | ≈ same |

---

# 29. Current Best Interpretation

The experiments do **not** support the conclusion:

> "Dim 14 is worse."

Nor do they support:

> "Dim 14 is simply better."

The stronger conclusion is:

> **Dim 14 provides useful additional representational capacity for intricate artist-specific rendering, particularly painterly brushwork, eyelashes, hair/light interaction, and subtle skin rendering. However, this increased capacity also makes the LoRA more sensitive to poorly represented or ambiguous fine details.**

Config A therefore remains the safer general-purpose baseline, while Config B is a promising higher-capacity configuration that may benefit from better dataset supervision.

---

# 30. Recommended Next Experiment

The next experiment should avoid broad hyperparameter changes.

Keep Config B fixed:

```text
Network: 14 / 14
Conv:    12 / 10
LR:      3.9e-4
WD:      0.1
Warmup:  5%
Scheduler: cosine
```

Change **only the face-crop dataset**.

The objective should not be simply "more face crops."

Instead, select crops with high information density around:

- pupil,
- iris,
- eyelashes,
- eyelid,
- surrounding skin,
- broader face context.

Maintain variation in:

- character,
- expression,
- viewing angle,
- lighting,
- hairstyle,
- eye orientation.

Avoid excessive repetition of nearly identical eye structures.

The desired outcome is:

```text
B's improved painterly rendering
        +
B's improved eyelashes/hair/light interaction
        +
more stable pupil/iris distinction
        +
preserved anatomy
```

---

# 31. Decision Criteria for the Next Experiment

The experiment should be evaluated using four independent criteria.

### 1. Artist-specific style

Does the model retain:

- brushwork,
- palette,
- skin rendering,
- hair rendering,
- lighting?

### 2. High-frequency style

Does it improve:

- eyelashes,
- hair highlights,
- skin texture,
- painterly details?

### 3. Structural accuracy

Does it preserve:

- eyes,
- fingers,
- nails,
- anatomy,
- pose structure?

### 4. Generalization

Does it work on:

- unseen characters,
- unseen poses,
- different compositions,
- different image structures?

The optimal LoRA is not necessarily the one with the strongest style imprint.

It is the configuration that gives the best **balance across all four dimensions**.

---

# 32. Final Configuration Philosophy

The entire experiment series has led to a general training philosophy:

### First stabilize the optimizer

Use:

- moderate LR,
- WD ≈ 0.1,
- cosine scheduling,
- modest warmup,
- AdamW8bit.

### Then optimize dataset composition

Use:

- representative images,
- targeted high-frequency crops,
- controlled repeats,
- regularization examples,
- temporal/style weighting when appropriate.

### Only then increase capacity

Higher rank should be used when the dataset demonstrates that there is useful information that the current rank cannot adequately represent.

The Config B experiment is particularly useful because it demonstrates that this additional information can indeed exist:

- eyelashes,
- brush strokes,
- hair/light interactions,
- skin sketch lines.

However, it also demonstrates the danger:

> **Extra capacity can expose weaknesses in the dataset rather than fixing them.**

---

# 33. Current Recommended Baseline

For now, the most reliable production baseline remains:

```text
                CONFIG A

Network Dim:       12
Network Alpha:     12

Conv Dim:          12
Conv Alpha:        10

Learning Rate:     3.8–3.9e-4
Weight Decay:      0.10
Scheduler:         cosine
Warmup:            5%

Batch Size:        4
Resolution:        1024
Precision:         bf16

Optimizer:         AdamW8bit
Betas:             0.9, 0.9985

Weighting:         logit_normal

Training:          U-Net only
```

**Config B (14/14) should remain an experimental candidate**, particularly if further crop curation can improve the unstable fine features.

---

# 34. Overall Conclusion

The experiments demonstrate that compact Anima LoRA training is fundamentally a problem of **representation allocation** rather than simply maximizing training strength.

The mature Config A provides a strong equilibrium:

> **Style fidelity + detail + anatomy + generalization + stability**

Config B demonstrates that additional rank can unlock genuinely useful artist-specific information:

> **More painterly brushwork + better eyelashes + more intricate hair/light interaction + richer skin rendering**

but also increases sensitivity to poorly represented details:

> **More unstable fine anatomy + persistent pupil/iris instability + greater artifact sensitivity**

The most important remaining problem is therefore not simply lack of LoRA capacity.

The current evidence suggests:

> **The LoRA has sufficient capacity to learn many of the artist's intricate rendering characteristics, but the dataset does not provide equally strong supervision for every fine-scale feature.**

This makes **dataset composition and crop quality the next highest-value optimization target**.

In particular, the pupil/iris problem should be treated as a specialized representation problem rather than assuming that increasing rank alone will solve it.

The ideal future configuration may ultimately resemble **Config B with a better-curated dataset**, but until that is demonstrated, **Config A remains the safer and more balanced baseline.**