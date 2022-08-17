# tract-python

 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Tract inference engine](https://github.com/sonos/tract) bindings in Python via FFI.

## Why

Tract Cli is very feature-full but reloading a model each time you wish
to do an inference is computationaly costy and slow.

## Usage

```python
import tract_python

tract_model = tract_python.TractModel.load_plan_from_nnef_dir(
  './test_simple_nnef/'
)
results = tm.run(input_0=np.arange(6).reshape(1, 2, 3).astype(np.float32))
print(results)
#{'output_0': array([[[ 0.,  2.,  4.],
#       [ 6.,  8., 10.]]], dtype=float32)}

```

## Status

This project is in alpha state.

## Scope

My personnal usecase is to be able to run benchmarks (+10M inferences) with 'tract' engine.

Ideally I would like to support most of the `tract-cli` features:
- [X] load NNEF dir
- [X] run simple plan
- [ ] load all supported formats: ONNX, TF, from HTTP, from .tgz
- [ ] expose computations of model informations:
    - [ ] number of parameters
    - [ ] size on disk
    - [ ] profiling infos

That said I do not have the bandwith to do it up-front.
I welcome any contributor that would wish to add more features.
