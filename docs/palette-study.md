# Palette representation feasibility study

Reviewed 2026-10-02. This bounded LOCAL-11/15 investigation used the same sixteen
final-GAN KDE-bandwidth-3.16 generated skins from the sampling study. It compares
posthoc palette mapping with the original PNG exports. No dataset, model, source
image, or package implementation was changed.

Keep palette mapping as an optional styling control. Thirty-two colors produced
the smallest color error of the tested palettes, but this batch does not justify
changing the default output or adopting a learned discrete model.

## Protocol

The input was `sampling-study/gan-kde-3-16/generated.npy` under the local ignored
`images/results/roadmap-2026-10-02/` experiment directory. Its SHA256 is
`65f606799db78ca0902a87d6085f42cef8ce6dc0fcd27e26ef15e5a2bc696511`.

Export the normalized array by clipping/rounding to uint8 RGBA. Fit a separate
global 8-, 16-, or 32-color palette using only exported RGB pixels whose exported
alpha is greater than zero. The same 40,627 visible pixels across sixteen images
are used for fitting and measurement; this is an in-batch representation study,
not a generalization evaluation. Palette fitting does not weight pixels by alpha.

`MiniBatchKMeans` uses seed 1976, one initialization, at most twenty iterations,
batch size 1,024, initialization size 3,072, ten non-improving steps for early
stopping, reassignment ratio 0.01, and zero tolerance. Round/clip centers to uint8
and apply their hex RGB values through the existing `creator.recolor_skin` API.
It maps visible pixels to the nearest RGB center and preserves alpha and RGB at
fully transparent pixels exactly.

The CPU command was externally limited to 120 seconds, with BLAS/OpenMP and
`threadpoolctl` restricted to one thread. Computation/export took 0.34 seconds,
excluding interpreter startup and final artifact hashing. All fits stopped after
two iterations, at 58, 53, and 59 minibatch steps respectively. A separate fresh
process refit reproduced all three rounded palettes exactly in the same pinned
environment.

## Measured results

Color error is original-alpha-weighted RGB MSE in normalized `[0,1]` units.
Within-batch variation is mean RGBA MSE over the same 120 image pairs; it measures
pixel difference, not semantic diversity or originality. The exported PNG batch
is the baseline, so float-to-PNG quantization is separate from palette error.

| Representation | Visible RGB colors | Visible RGB MSE | Pairwise RGBA MSE | Variation retained | Unique images |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original export | 31,032 | 0 | 0.141614 | 100% | 16/16 |
| Global palette 8 | 8 | 0.011829 | 0.132672 | 93.69% | 16/16 |
| Global palette 16 | 16 | 0.007043 | 0.137187 | 96.87% | 16/16 |
| Global palette 32 | 32 | 0.004064 | 0.139011 | 98.16% | 16/16 |

All sixty-four PNGs decoded as 64×64 RGBA. Alpha and hidden RGB bytes matched the
original export for every palette image. Mean alpha stayed 0.497667 and the
fraction of pixels with intermediate exported alpha stayed 0.274765. All palettes
used exactly their requested number of distinct colors; none produced exact
duplicate images in this sixteen-image batch.

## Visual findings and decision

The four sheets use the same sample order and 4×4 layout, each skin enlarged from
64×64 to 256×256 with nearest-neighbor resizing and checkerboard compositing.
They were inspected after creation:

- [Original](../images/results/roadmap-2026-10-02/palette-study/original/sheet.png)
- [8 colors](../images/results/roadmap-2026-10-02/palette-study/palette-8/sheet.png)
- [16 colors](../images/results/roadmap-2026-10-02/palette-study/palette-16/sheet.png)
- [32 colors](../images/results/roadmap-2026-10-02/palette-study/palette-32/sheet.png)

The 8- and 16-color sheets show posterized bands and lost or shifted hues; the
32-color sheet retains more of this batch's hue range. Semitransparent softness
and incoherent details remain. These visual observations are unblinded atlas
inspection, not a measured improvement in crispness, wearable quality, or game
acceptance. Fewer colors alone cannot support those claims.

LOCAL-15 now has a measured palette-control effect: predictable color reduction
with increasing color error and a modest reduction in pixel variation. Keep it
opt-in and let creators compare the original with the mapped image. No palette
size is established as a universal quality threshold.

LOCAL-11's palette-aware representation alternative has a bounded feasibility
result. This study does not train quantized latents or restore missing spatial
detail. Defer a VQ implementation or production migration until a separate
controlled study demonstrates benefit against the evaluation rubric; this result
does not prove or disprove a learned discrete model's potential.

## Artifacts and limits

A follow-up nearest-training review (local-only `../artifacts/roadmap-palette-neighbors.json`)
compared all sixteen exported images in each representation against all 106,248
training images using normalized RGBA MSE. Mean nearest distances were 0.033212
for the original export, 0.033519 for 8 colors, 0.033409 for 16 colors and 0.033118
for 32 colors. The original differs slightly from the sampling report because
this comparison uses exported uint8 pixels rather than pre-export floats.
Individual neighbor identities/distances are retained as copying-review flags;
these small mean differences establish neither originality nor copying safety.
The first monolithic attempt hit its 180-second cap (exit 124). A declared
300-second retry preserved each completed representation and finished in 293.36
seconds with the same sixteen samples and full training cohort.

The local-only summary (local-only `../artifacts/roadmap-palette-study.json`) records learned
palettes, protocol, versions, measured results, source/report/sheet hashes, and
verification. The ignored local [full report](../images/results/roadmap-2026-10-02/palette-study/study.json)
hashes all 73 local artifacts, including the study script, PNGs, arrays, and sheets.
Those hashes were independently checked, and the input array hash remained
unchanged. The local artifacts are unavailable from a fresh clone.

Limits: one sampling seed and sixteen generated images; fitting and measurement
share the same pixels; global palettes can suppress image-specific hues; no
blinded creator rubric, unseen samples, human copying assessment, model retraining,
alpha-threshold intervention, or learned VQ experiment. No new dependency or
package code was introduced, so no CI test was required for this research-only
change.
