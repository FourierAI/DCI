# Dataset setup for the paper protocol

The repository contains label catalogs and lightweight evaluation metadata, but
does not redistribute dataset images. Obtain every dataset from its official
source and follow its license.

Current manuscript summary data are in [`results/`](results/README.md). Dataset
catalogs here describe evaluation inputs; they are separate from experimental
prediction logs and from the reported aggregate results.

| Dataset | Official source | Paper split |
|:--|:--|:--|
| CIFAR-100 | <https://www.cs.toronto.edu/~kriz/cifar.html> | complete 10,000-image test split |
| CUB-200-2011 | <https://www.vision.caltech.edu/datasets/cub_200_2011/> | complete 5,794-image official test split |
| Food-101 | <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/> | complete 25,250-image official test split |
| ImageNet-1K / ImageNet-21K | <https://www.image-net.org/> | see below |

The loader validates split and vocabulary sizes before inference. This prevents
an accidental train/full-dataset evaluation from being reported as the paper's
test protocol.

## Expected layouts

### CIFAR-100

Materialize the 10,000 official test images with the relative filenames used by
`metadata/cifar100/labels.json`, or pass a JSON mapping through `--metadata`.

```text
data/images/cifar100/
└── <10,000 test images>
```

### CUB-200-2011

Keep the official metadata beside the image directory. `--image-root` may point
to either `CUB_200_2011/` or `CUB_200_2011/images/`.

```text
CUB_200_2011/
├── classes.txt
├── image_class_labels.txt
├── images.txt
├── train_test_split.txt
└── images/
    └── 001.Black_footed_Albatross/...
```

The runner selects the rows whose official split flag is `0`, yielding 5,794
test images and 200 candidate classes.

### Food-101

Keep the official `meta/test.txt` file beside `images/`. `--image-root` may
point to either `food-101/` or `food-101/images/`.

```text
food-101/
├── meta/
│   └── test.txt
└── images/
    └── apple_pie/...
```

The runner appends `.jpg` to the entries in `test.txt`, yielding the official
25,250-image test split and 101 candidate classes.

### ImageNet-1K

The bundled `metadata/imagenet1k/split_TAI_imagenet_val.json` indexes all 50,000
official validation images. Point `--image-root` at the directory containing
those files. For the paper's candidate-scaling protocol, the runner samples
classes independently for every run and every candidate count, then evaluates
all 50 validation images from each selected class.

Two ImageNet-1K pairs share a human-readable name (`crane` and `maillot`). The
runner disambiguates only these collisions with their numeric class ID so the
candidate vocabulary retains all 1,000 classes.

### ImageNet-21K full-vocabulary stress test

The paper samples images from an available ImageNet-21K pool and evaluates them
against all 21,843 synsets. It does not use ImageNet-1K validation images as a
substitute. Arrange images under WNID directories:

```text
imagenet21k/
├── n00004475/...
├── n00005787/...
└── ...
```

Create a reusable image index before evaluation:

```bash
dci-index-imagenet21k \
  --image-root /path/to/imagenet21k \
  --output /path/to/imagenet21k-index.json
```

The index format is a JSON object from image path relative to `--image-root` to
WNID:

```json
{
  "n01440764/example.JPEG": "n01440764"
}
```

The candidate catalog comes from `metadata/imagenet21k/im21K.txt`. The runner
uses the full comma-separated synset description and appends a WNID only when
two descriptions are identical, preserving exactly 21,843 unique candidate
strings. Each run samples 1,000 distinct indexed images uniformly without
replacement; samples may overlap across runs.

## Custom JSON metadata

For CIFAR-100 or another standard dataset configuration, `--metadata` accepts a
mapping from relative image path to either a label string or an object with a
`class_name` field. A custom mapping must still satisfy the configured paper
protocol counts. ImageNet-21K instead expects a relative-path-to-WNID mapping as
shown above.
