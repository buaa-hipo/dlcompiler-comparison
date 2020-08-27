# DL compiler comparison

The performance comparison acrossing widely used deep learning compilers (e.g., TVM, nGraph, Tensor Comprehension, Glow and XLA)

For more information about the DL compilers, please refer to our survey paper **The Deep Learning Compiler: A Comprehensive Survey** [on arXiv](https://arxiv.org/abs/2002.03794).

We have compared the end-to-end and per-layer (convolution) performance among DL compilers on CNN models. We upload the corresponding scripts in this repo, and we hope to save time for the practitioners.

## Usage
Please refer to the `README` in the following directories. 
```
|-- TVM
|-- nGraph
|-- TC_perlayer
|-- Glow
|-- XLA
|-- micro-models
```
