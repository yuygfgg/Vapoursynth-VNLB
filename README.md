# VapourSynth-VNLB

Video Non-Local Bayes denoising filter for VapourSynth.

## Description

- `vnlb.Basic` builds the basic estimate contribution stack.
- `vnlb.Final` builds the final estimate contribution stack using a reference clip, usually the aggregated basic estimate.
- `vnlb.Aggregate` turns contribution stacks into normal video frames.

Supported input formats:

- `GrayS`
- `YUV444PS`

Note: It is recommended to convert your source into OPP colorspace before calling VNLB, as the plugin does not perform any internal color-space conversion.

## Installation

```
pip install -U vapoursynth-vnlb
```

## Experimental CUDA plugin

CUDA Toolkit with `nvcc`, cuSOLVER and cuBLAS is required for building the CUDA plugin.

```sh
cmake -S . -B build/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVNLB_ENABLE_CUDA=ON \
  -DVNLB_BUILD_PLUGIN=ON \
  -DVNLB_ENABLE_TESTS=ON \
  -DVNLB_CUDA_ARCHITECTURES=120-real
cmake --build build/cuda --target vnlbcu
ctest --test-dir build/cuda --output-on-failure
```

Load the resulting module as a separate VapourSynth plugin:

```python
core.std.LoadPlugin(path="/absolute/path/to/libvnlbcu.so")

basic = core.vnlbcu.Basic(src, sigma=8.0)
final = core.vnlbcu.Final(src, ref=basic, sigma=8.0)
```

`vnlbcu.Basic` and `vnlbcu.Final` return normal video frames directly. Do not pass their output to `vnlb.Aggregate`.

## Usage

### VNLB Functions

#### basic estimate of VNLB denoising filter

This basic estimate produces a decent estimate of the noise-free image, as a reference for the final estimate.

```python
vnlb.Basic(clip clip, float sigma[, int block_size=8, int block_step=8, int group_size=8, float cap_factor=4.0, float model_cap_factor=1.0, int bm_range=9, int patch_time=1, int radius=1, int search_bwd=1, int search_fwd=1, int rank=8, float beta=1.0, float tau=0.0, float variance_threshold=1.1, float weight_alpha=0.75, float weight_beta=0.35, float weight_gamma=1.0, float weight_epsilon=1e-6, float membership_noise_floor=0.25, int chroma=1, clip mvfw=None, clip mvbw=None])
```

- clip:
    The input clip, the clip to be filtered. It must be constant-format `GrayS` or `YUV444PS`. Integer sources should be converted before calling VNLB.

- sigma:
    The strength of denoising in 8-bit sample units (e.g., `5.1` means about 5.1 luma levels on an 8-bit scale). The plugin divides it by `255` internally for float clips. Technically, this is the standard deviation of i.i.d. zero mean additive white Gaussian noise in 8 bit scale.

- block_size:
    The size of a spatial block is block_size x block_size, representing a local patch. A value of `8` uses `8x8` patches. Generally, larger blocks capture broader structure but cost more and can blur small details if the rank is too low.

- block_step:
    Sliding step to process every next reference block (anchor block spacing). Lower values process more anchor positions and reduce sparse-anchor artifacts; higher values are faster. The default is `8`. Set `block_step=0` to use the automatic step, which is `block_size`.

- group_size:
    Maximum number of similar blocks in each group before threshold expansion. More patches improve averaging stability but increase cost and may smooth texture when the search admits weak matches.

- cap_factor:
    Soft cap for threshold-expanded groups. VNLB can keep additional patches after the best `group_size` matches when `tau` is active; this limits the expanded group to at most `ceil(group_size * cap_factor)` patches. Use `0.0` to disable this soft cap and keep all candidates admitted by the search threshold. Non-zero values must be at least `1.0`.

- model_cap_factor:
    Soft cap for the patches used to estimate the Bayesian model mean and covariance. The model uses at most `ceil(group_size * model_cap_factor)` retained patches. The default `1.0` keeps model construction fast while still allowing extra threshold-expanded patches to be filtered and aggregated. Use `0.0` to build the model from all retained patches. Non-zero values must be at least `1.0`.

- bm_range:
    Spatial search radius for block-matching around the guided center. The size of the search window is `(bm_range * 2 + 1) x (bm_range * 2 + 1)`. `3` means candidates are searched inside a `7x7` window. Larger is slower, with more chances to find similar patches.

- patch_time:
    Number of consecutive frames in each temporal patch. A value of `2` estimates patches spanning the anchor frame and the next frame.

- radius:
    The temporal radius for denoising. Symmetric temporal search radius. It sets both `search_bwd` and `search_fwd`. Explicit `search_bwd` or `search_fwd` values override the shorthand for their respective directions.

- search_bwd, search_fwd:
    Number of previous and next patch origins to search. With `patch_time=2` and `radius=1`, each anchor can use patch origins from one frame before through one frame after, while still reading the extra frame needed by the temporal patch.

- rank:
    Retained Bayesian model rank. Lower ranks remove more high-frequency variation; higher ranks preserve more detail and noise. For `YUV444PS` with `chroma=1`, the estimator dimension is `3 * block_size * block_size * patch_time`, so useful ranks can be higher than Gray-only settings.

- tau:
    Distance expansion threshold in 8-bit squared-distance units. VNLB first selects the best `group_size` patches, then keeps any additional candidates whose distance is at or below `max(kth_distance, tau)`. Set `tau=0.0` to keep only the nearest `group_size` patches.

- beta:
    Multiplier applied to the noise variance used in Bayesian shrinkage. Higher values increase denoising strength.

- variance_threshold:
    Dimensionless cutoff applied to the internal normalized noise variance. Components below `variance_threshold * beta * sigma^2` are discarded. Higher values usually smooth more.

- weight_alpha:
    Strength of group-confidence weighting during aggregation. Higher values give more influence to groups whose Bayesian model is more confident, usually reducing weak-match blur and improving stable detail. Too high can over-favor very smooth or very confident groups and suppress subtle texture. Set `0.0` to disable this term.

- weight_beta:
    Strength of patch membership weighting during aggregation. Higher values reject patches that fit the current group poorly, which can reduce bad temporal matches and texture smearing. Too high can make aggregation too selective, leaving more residual noise or a slightly patchy look. Set `0.0` to disable this term.

- weight_gamma:
    Strength of the spatial patch aggregation window. Higher values favor patch centers over patch borders, usually reducing blocking or grid-like artifacts. Too high can make overlapping patches contribute less evenly. Set `0.0` to disable the window.

- weight_epsilon:
    Stabilizer for group-confidence weighting. Most users should leave it as default.

- membership_noise_floor:
    Noise tolerance floor for patch membership weighting. Higher values make membership weighting less strict, allowing more averaging and smoother output. Lower values make it stricter, which can reject bad matches more aggressively but may leave more noise or uneven aggregation. Most users should leave it as default.

> [!NOTE]
> Formula reference:
>
> $$
w_{k,i,u}
=
\tau_k^{-\alpha}
\cdot
\exp(-\beta M_{k,i})
\cdot
a(u)^\gamma
$$
>
> where `weight_alpha`, `weight_beta`, and `weight_gamma` are $\alpha$, $\beta$, and $\gamma$.
>
> The group precision term is:
>
>   $$
    \tau_k
    =
    \varepsilon
    + \frac{d}{n_{\mathrm{eff},k}}
    + \sum_j
      \frac{\lambda_{k,j}}{\lambda_{k,j}+\sigma_{\mathrm{est}}^2}
    $$
>
>    The membership penalty is:
>
>   $$
    M_{k,i}
    =
    \max(0, Z_{k,i}-Z_{k,\mathrm{ref}})
    + \frac{1}{d_m}
      \sum_j \log\left(1+\frac{\lambda_{k,j}}{\sigma_{\mathrm{eff}}^2}\right)
    $$
>
>    with:
>
>    $$
    Z_{k,i}
    =
    \frac{D_{k,i}-d_m}{\sqrt{2d_m}},
    \qquad
    Z_{k,\mathrm{ref}}
    =
    \min_i Z_{k,i}
    $$
>
> Symbol notes:
>
> - $k$ is the group index, $i$ is the patch index inside that group, and $u$ is the pixel position inside the patch.
> - $w_{k,i,u}$ is the aggregation weight applied to that patch contribution.
> - $\tau_k$ is the group-confidence denominator; smaller values give group $k$ more weight.
> - $M_{k,i}$ is the membership penalty for patch $i$ under group $k$; larger values reduce that patch's weight.
> - $a(u)$ is the normalized patch window value at position $u$.
> - $d$ is the estimator patch dimension, and $d_m$ is the dimension used for membership scoring.
> - $n_{\mathrm{eff},k}$ is the number of patches used to estimate group $k$'s model.
> - $\lambda_{k,j}$ is the signal covariance eigenvalue of component $j$ in group $k$.
> - $\sigma_{\mathrm{est}}^2$ is the noise variance used by the Wiener shrinkage term.
> - $\sigma_{\mathrm{eff}}^2$ is the membership noise variance after applying `membership_noise_floor`.
> - $\varepsilon$ is `weight_epsilon`.
> - $D_{k,i}$ is the Mahalanobis distance of patch $i$ to group $k$'s model.
> - $Z_{k,i}$ is the chi-square normalized form of $D_{k,i}$, and $Z_{k,\mathrm{ref}}$ is the best score inside the group.

- chroma:
    Defaults to `1`. For `YUV444PS`, `chroma=1` estimates one coupled multi-channel model across Y, U, and V. Use `chroma=0` to process planes independently. `GrayS` clips ignore this setting.

- mvfw, mvbw:
    Optional MVTools vector clips. They guide temporal search centers but do not replace exhaustive patch search inside `bm_range`. `mvfw` must come from `isb=False`, and `mvbw` from `isb=True`, both with `delta=1` and dimensions matching `clip`.

#### final estimate of VNLB denoising filter

It takes the basic estimate as a reference. This final estimate can be realized as a refinement. It can significantly improve the denoising quality, keeping more details and fine structures that were removed in basic estimate.

```python
vnlb.Final(clip clip, clip ref, float sigma[, ...])
```

- clip:
    The input clip, the clip to be filtered.

- ref:
    The reference clip, this clip is used in block-matching and as the reference in filtering. It must match `clip` format, dimensions, and frame count. In the usual two-stage chain this is the aggregated Basic estimate.

- sigma_basic:
    Final-stage reference noise level in 8-bit sample units. Models residual noise in `ref`. Use `0.0` when treating the Basic/reference clip as clean; increasing it makes Final discount noisy reference variance and can smooth more.

- gamma:
    Flat-area detector multiplier. It is used only when `flat_areas=1`; a group is treated as flat when its variance is below `gamma * sigma^2`.

- flat_areas:
    Enables the flat-area path. Can improve very flat regions, but may smooth low-contrast texture. Final defaults to `1`.

- *Other parameters*:
    Same as those in vnlb.Basic. Final uses different defaults for `tau=400.0`, `variance_threshold=1.7`, `weight_alpha=1.0`, and `weight_beta=0.5`.

#### aggregation of VNLB denoising filter

```python
vnlb.Aggregate(clip clip, clip src[, int patch_time=1, int radius=1, int search_bwd=1, int search_fwd=1])
```

- clip:
    Contribution stack returned by `Basic` or `Final`. Stack height is `src.height * 2 * (search_bwd + search_fwd + patch_time)`.

- src:
    Original input clip used for dimensions and fallback pixels where no VNLB contribution is available.

- patch_time, radius, search_bwd, search_fwd:
    Must receive the same temporal settings used to create its contribution stack in `vnlb.Basic` or `vnlb.Final`.

### CUDA Functions

#### basic estimate of CUDA VNLB

```python
vnlbcu.Basic(clip clip, float sigma[, int block_size=8, int block_step=0, int group_size=15, float cap_factor=4.0, float model_cap_factor=1.0, int bm_range=9, int patch_time=1, int radius=1, int search_bwd=1, int search_fwd=1, int rank=8, float beta=1.0, float tau=0.0, float variance_threshold=1.1, float weight_alpha=0.75, float weight_beta=0.35, float weight_gamma=1.0, float weight_epsilon=1e-6, float membership_noise_floor=0.25, int chroma=1, int device_id=0, int num_streams, int chunk_size=256])
```

The denoising parameters have the same meaning as in `vnlb.Basic`, with these CUDA-specific defaults and options:

- `block_step=0` selects `max(1, block_size // 2)`; it is `4` for the default `8x8` patch.
- `group_size` defaults to `15`.
- `device_id` selects the CUDA device.
- `num_streams` defaults to `min(VapourSynth threads, 4)` and accepts `1..32`.
- `chunk_size` is the maximum number of anchor groups submitted together and defaults to `256`.

#### final estimate of CUDA VNLB

```python
vnlbcu.Final(clip clip, clip ref, float sigma[, int block_size=8, int block_step=0, int group_size=17, float cap_factor=4.0, float model_cap_factor=1.0, int bm_range=9, int patch_time=1, int radius=1, int search_bwd=1, int search_fwd=1, int rank=8, float beta=1.0, float tau=400.0, float variance_threshold=1.7, float weight_alpha=1.0, float weight_beta=0.5, float weight_gamma=1.0, float weight_epsilon=1e-6, float membership_noise_floor=0.25, int chroma=1, int device_id=0, int num_streams, int chunk_size=256, float sigma_basic=0.0, float gamma=0.95, int flat_areas=1])
```

`ref`, `sigma_basic`, `gamma` and `flat_areas` have the same meaning as in `vnlb.Final`.

The main default differences are:

| Parameter | CUDA Basic / Final | CPU Basic / Final |
| --- | --- | --- |
| `group_size` | `15 / 17` | `8 / 8` |
| automatic `block_step` | `block_size // 2` | `block_size` |
| output | normal video frame | contribution stack |
| `chroma` | `1` only | `0` or `1` |
| `model_cap_factor` | `1.0` only | configurable |
| motion vectors | not supported | optional MVTools guidance |

#### Additional restrictions for CUDA VNLB

- Only coupled processing is supported, `chroma=0` is not supported.
- `model_cap_factor` must be `1.0`.
- The effective model group `K=min(group_size, candidate count)` must satisfy `K<=128`, `rank<=K`, and `K<D`, where `D=channels * block_size^2 * patch_time`.
- The fused one-block matcher accepts at most `2048` candidates per anchor.
- MVTools guidance is not available.


## Examples

### Basic Example

```python
import vapoursynth as vs

core = vs.core

params = dict(
    sigma=5.1,
    block_size=8,
    block_step=8,
    patch_time=1,
    bm_range=9,
    radius=1,
    group_size=8,
    rank=8,
)

basic_stack = core.vnlb.Basic(src, **params)
basic = core.vnlb.Aggregate(
    basic_stack,
    src=src,
    patch_time=params["patch_time"],
    radius=params["radius"],
)

final_stack = core.vnlb.Final(src, ref=basic, **params)
final = core.vnlb.Aggregate(
    final_stack,
    src=src,
    patch_time=params["patch_time"],
    radius=params["radius"],
)

final.set_output()
```

For `YUV444PS`, `chroma` defaults to `1`, estimating a coupled multi-channel model. Set `chroma=0` to process planes independently.

### MVTools Motion Guidance Example

VNLB can run without motion vectors. In that mode each temporal search window is centered at the same `(x, y)` location. When `mvfw` and `mvbw` are passed, the vectors are used as block-motion guidance for block-matching centers.

```python
src = core.resize.Bicubic(src, format=vs.GRAYS)

# MVTools does not operate on float clips.
mvsrc = core.resize.Bicubic(src, format=vs.GRAY8)

sup = core.mv.Super(mvsrc, pel=2, hpad=16, vpad=16)
mvfw = core.mv.Analyse(sup, isb=False, delta=1, blksize=8, overlap=4)
mvbw = core.mv.Analyse(sup, isb=True,  delta=1, blksize=8, overlap=4)

params = dict(
    sigma=5.1,
    block_size=8,
    block_step=8,
    patch_time=1,
    bm_range=9,
    radius=1,
    group_size=8,
    rank=8,
    mvfw=mvfw,
    mvbw=mvbw,
)

basic_stack = core.vnlb.Basic(src, **params)
basic = core.vnlb.Aggregate(
    basic_stack,
    src=src,
    patch_time=params["patch_time"],
    radius=params["radius"],
)

final_stack = core.vnlb.Final(src, ref=basic, **params)
final = core.vnlb.Aggregate(
    final_stack,
    src=src,
    patch_time=params["patch_time"],
    radius=params["radius"],
)
```

### Scene Cuts Example

Invalid MVTools vector frames fall back to zero motion automatically. For scene cuts, VNLB checks scene-change props on the clip passed to `vnlb.Basic` / `vnlb.Final`: `_SceneChangePrev`, `_SceneChangeNext`, `Scenechange`.

Use `mv.SCDetection()` on the integer motion source, then copy the props to the float clip used by VNLB:

```python
src = core.resize.Bicubic(src, format=vs.GRAYS)
mvsrc = core.resize.Bicubic(src, format=vs.GRAY8)

sup = core.mv.Super(mvsrc, pel=2, hpad=16, vpad=16)
mvfw = core.mv.Analyse(sup, isb=False, delta=1, blksize=8, overlap=4)
mvbw = core.mv.Analyse(sup, isb=True,  delta=1, blksize=8, overlap=4)

mvsrc_sc = core.mv.SCDetection(mvsrc, mvfw)
mvsrc_sc = core.mv.SCDetection(mvsrc_sc, mvbw)

src = core.std.CopyFrameProps(
    src,
    prop_src=mvsrc_sc,
    props=["_SceneChangePrev", "_SceneChangeNext", "Scenechange"], # MVTools does not produce `Scenechange`, but other ones might (vapoursynth-wwxd).
)

basic_stack = core.vnlb.Basic(
    src,
    sigma=5.1,
    patch_time=2,
    radius=1,
    mvfw=mvfw,
    mvbw=mvbw,
)
```

If `_SceneChangePrev` or `Scenechange` is set, `mvfw` for that frame is ignored. If `_SceneChangeNext` is set, `mvbw` for that frame is ignored.

## Build

Configure and build the release plugin:

```bash
cmake --preset release
cmake --build --preset release
```

## Tests

Run the native and VapourSynth integration tests:

```bash
cmake --preset dev
cmake --build --preset dev
ctest --preset dev --output-on-failure
```

```bash
python3 tests/vs/integration_bestsource.py \
  build/release/libvnlb.dylib \
  . \
  build/release/bestsource-cache
```

## Notes

- The implementation does not copy the original VNLB reference source, so output is not bit-exact.
- MVTools support is nearest-block sampling of the finest vector level.

## References

- Pablo Arias and Jean-Michel Morel, "Video Denoising via Empirical Bayesian
  Estimation of Space-Time Patches", Journal of Mathematical Imaging and
  Vision, 60(1), 70-93, 2018.
  <https://doi.org/10.1007/s10851-017-0742-4>
- <https://github.com/pariasm/vnlb>

## License

`AGPL-3.0-or-later`
