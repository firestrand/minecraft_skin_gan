# Skin diagnostics: renderer profile, not game certification

`minecraft_skin_gan.skin_checks.skin_diagnostics(pixels, model_type="classic")`
accepts exactly a `(64, 64, 4)` NumPy `uint8` array. Choose `classic` or `slim`
explicitly. It leaves every input byte unchanged and returns a JSON-compatible,
versioned diagnostic report. It does not repair alpha, enforce seamless colors,
or certify upload acceptance, copyright, or wearable quality.

## Verified source profile

The atlas rectangles, arm widths, and base/overlay distinction follow the pinned
[skinview3d `setUVs` and `SkinObject` implementation](https://github.com/bs-community/skinview3d/blob/84906e983a2cf325f33f515b9a71c871f799054b/src/model.ts).
That renderer constructs six cuboid parts with separate base and overlay
materials; its overlay material enables transparency. This is evidence about
that renderer, not Mojang's alpha preprocessing or account upload service.

Face orientation follows the pinned
[Three.js `BoxGeometry` implementation](https://github.com/mrdoob/three.js/blob/9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8/src/geometries/BoxGeometry.js).
The existing `face_pixels` helper reverses bottom-face rows. Diagnostics compare
all twelve geometric edges of each part and layer with the necessary direction
reversals; they do not compare unrelated atlas-neighbor pixels.

[Minecraft's own skin guide](https://www.minecraft.net/en-us/article/what-is-minecraft-skin)
describes importing an unwrapped PNG. This review did not obtain a primary
Mojang source proving all modern base/overlay alpha acceptance rules. Therefore
`game_acceptance` always says `unverified`. The array contract does not prove
that a source file is a PNG; file decoding checks belong to dataset/export
boundaries. Legacy 64×32 skins, HD textures, capes, custom geometry, and automatic
model guessing are outside this profile.

## Diagnostic meanings

Preview rendering has an explicit `alpha_mode` choice. The default `original`
preserves supplied face RGBA bytes. `opaque-base` renders embedded base-face
alpha as 255 while retaining RGB and every overlay alpha value. This approximates
the base opacity distinction in the pinned skinview3d material configuration:
its base material does not enable transparency, while its overlay does. The
pinned [Three.js `Material` defaults](https://github.com/mrdoob/three.js/blob/9b4a2ac29c63ccb43fd51c5661f2f873ac2c39b8/src/materials/Material.js)
set `transparent=false` and `opacity=1`. Both the HTML label and metadata disclose
the chosen mode. This is a local rendering option, not full shader/game
equivalence, alpha repair, or upload certification; source PNG and export bytes
are unchanged.

`layer_masks` returns fresh disjoint base/overlay masks containing used texels.
Each classic layer contains 1,632 texels; each slim layer contains 1,568.
Unused atlas texels are excluded from opacity statistics. Each layer reports
opaque (`255`), transparent (`0`), and partial-alpha (`1..254`) counts and mean
alpha divided by 255. `base_all_opaque` is only an observation, not a pass/fail
verdict. A transparent or partially transparent overlay is permitted by the
report; it is never rewritten.

Each edge reports its two oriented face edges, reversal flag, texel-pair count,
alpha MAE, minimum-alpha visible weight, and visible RGB MAE. For paired bytes
`a` and `b`, normalize channels by 255 and set `w = min(a.alpha, b.alpha)`.
RGB MAE is `sum(w * abs(a.rgb-b.rgb)) / (3 * sum(w))`; it is `null` when no
pair is jointly visible. Alpha MAE is `sum(abs(a.alpha-b.alpha))/pair_count`.
The overall summary weights by pair count or visible weight respectively;
it is not an unweighted mean of edge means. Float64 subtraction prevents
unsigned-byte wraparound.

Edges sample boundary texel centers on neighboring faces. They diagnose color
or alpha jumps, rather than estimating a physically continuous texture. Corner
texels participate in multiple edges. Interior detail, antialiasing, material
lighting, inter-part joins, overlay compositing, and pose-dependent visibility
are not measured. Legitimate clothing boundaries and contrasting designs can
produce large seam errors. There is deliberately no threshold that labels a
skin invalid or high quality.

## Verification

`tests/test_skin_checks.py` independently constructs face colors from world
coordinates. Every paired edge agrees for both model types, including reversed
back/top/bottom edges and narrow arms. Additional fixtures check alpha masks,
hidden RGB, partial-alpha mismatches, nonzero discontinuities, unchanged input
bytes, JSON serialization, and invalid shape/dtype/model choices.

A second geometric oracle constructs each face's four world-space corners from
the six pinned `BoxGeometry.buildPlane` calls. It discovers adjacency by grouping
coincident corner endpoints, independently of the production seam table. Every
body part/layer must contain each of the twelve discovered edges exactly once,
with its correct direction reversal. This catches missing or repeated pairings
as well as reversed edge order. The world-coordinate color fixture separately
checks every texel along those edges, not only the corners.

Run the focused checks with:

```sh
scripts/uv.sh run --locked pytest tests/test_skin_checks.py --no-cov
scripts/uv.sh run --locked ruff check minecraft_skin_gan/skin_checks.py tests/test_skin_checks.py
scripts/uv.sh run --locked ty check minecraft_skin_gan/skin_checks.py
```

Primary-source review performed 2026-10-02. In-game render and upload verification
remains necessary before promoting this renderer-profile report to a release
acceptance gate.
