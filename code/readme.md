## Code

In this section, the code contains two pieces, the decision boundary visualization code and the model lineage closeness analysis  code.

1. The code of visualizing decision boundaries references the following repository: [somepago/dbViz: The official PyTorch implementation - Can Neural Nets Learn the Same Model Twice? Investigating Reproducibility and Double Descent from the Decision Boundary Perspective (CVPR'22).](https://github.com/somepago/dbViz)
2. The code of measuring model lineage closeness.

## Implementation environment

I have a bit too many packages inside my virtual environment and I'm not sure which ones are necessary, so I've listed the full package and version information in ''requirements.txt''.

Among the things to be aware of are:

- The packages needed for decision boundary visualisation can be found in the repository described in the link above. Of course it is fine to use my environment.
- The more important ones in the environment are torch\==1.7.1+cu110 and torchvision\==0.8.2+cu110, consider configuring these two packages first and then see what other packages the code needs.

## Others

The way to get dataset and models is to check the "Benchmark".