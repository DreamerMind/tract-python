# tract-python

 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 [![PyPI version](https://badge.fury.io/py/tract_python.svg)](https://badge.fury.io/py/tract_python)
 [![CI](https://github.com/DreamerMind/tract-python/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/DreamerMind/tract-python/actions/workflows/CI.yml)

[Tract inference engine](https://github.com/sonos/tract) bindings in Python via FFI.
It support Neural Network inference from NNEF or ONNX

## Why

`tract-cli` is very feature-full but reloading a model each time you wish
to do an inference is computationaly costy and slow.

## Install

Install using pip:
```
pip install tract_python
```


## Usage

```python
import tract_python

tract_model = tract_python.TractModel.load_plan_from_path(
  # This parameter can be an ONNX or NNEF filepath (in case of NNEF it can be a dir or a tgz)
  './test_simple_nnef/' # simple graph that mul input by 2
)
results = tract_model.run(input_0=np.arange(6).reshape(1, 2, 3).astype(np.float32))
print(results)
#{'output_0': array([[[ 0.,  2.,  4.],
#       [ 6.,  8., 10.]]], dtype=float32)}

```

## Status

This project is in alpha state.

## Scope

My personnal usecase is to be able to run benchmarks (+10M inferences) with 'tract' engine.

Ideally I would like to support some others `tract-cli` features:
- [X] load NNEF dir and .tgz or ONNX
- [X] run simple plan
- [ ] computing: number of FMA operations
- [ ] computing: profiling infos

We do not plan to support Tensorflow nor Kaldi models for now but it could be added if there is need for this.
That said I do not have the bandwith to do it up-front.
I welcome any contributor that would wish to add more features.
