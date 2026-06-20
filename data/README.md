# Dataset setup

This directory contains evaluation metadata and class vocabularies only. The
original images are not redistributed.

Place each dataset under `data/images/<dataset>/`, or pass an existing location
with `--image-root`. File names and relative paths must match the bundled JSON
metadata.

| Dataset | Official source |
|:--|:--|
| CIFAR-100 | <https://www.cs.toronto.edu/~kriz/cifar.html> |
| CUB-200-2011 | <https://www.vision.caltech.edu/datasets/cub_200_2011/> |
| Caltech-256 | <https://data.caltech.edu/records/nyy15-4j048> |
| Food-101 | <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/> |
| ImageNet | <https://www.image-net.org/> |

Users are responsible for obtaining datasets from their official sources and
complying with the corresponding terms and licenses. ImageNet-21K experiments
use ImageNet-1K validation images with the expanded 21K candidate vocabulary.
