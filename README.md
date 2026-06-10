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

## Usage

### VNLB Functions

#### basic estimate of VNLB denoising filter

This basic estimate produces a decent estimate of the noise-free image, as a reference for the final estimate.

```python
vnlb.Basic(clip clip, float sigma[, int block_size=8, int block_step=8, int group_size=8, float cap_factor=4.0, float model_cap_factor=1.0, int bm_range=9, int patch_time=1, int radius=1, int search_bwd=1, int search_fwd=1, int rank=8, float beta=1.0, float tau=0.0, float variance_threshold=1.1, int chroma=1, clip mvfw=None, clip mvbw=None])
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
    Length of the side of the search neighborhood for block-matching around the guided center. The size of the search window is `(bm_range * 2 + 1) x (bm_range * 2 + 1)`. `3` means candidates are searched inside a `7x7` window. Larger is slower, with more chances to find similar patches.

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
    Same as those in vnlb.Basic. Final uses different defaults for `tau=400.0` and `variance_threshold=1.7`.

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
