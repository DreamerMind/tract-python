from pathlib import Path
import tract_python
import numpy as np
from urllib import request

test_dir = Path(__file__).parent
assets_dir = test_dir / "assets"


def simple_model_load_and_execute_n_times(mul_model_path):
    tm = tract_python.TractModel.load_from_path(mul_model_path)

    init_input = np.arange(6).reshape(1, 2, 3).astype(np.float32)
    expected_output = np.arange(6).reshape(1, 2, 3).astype(np.float32) * 2
    results = tm.run(input_0=init_input)
    assert np.allclose(
        results["output_0"],
        expected_output,
    )

    results2 = tm.run(input_0=init_input * 2)
    assert np.allclose(
        results2["output_0"],
        expected_output * 2,
    )


def test_mul2_nnef():
    simple_model_load_and_execute_n_times(assets_dir / "test_simple_nnef")


def test_mul2_onnx():
    return simple_model_load_and_execute_n_times(assets_dir / "mul.onnx")


def test_load_onnx_tract_unable():
    local_model_path = assets_dir / "resnet50.onnx"
    request.urlretrieve(
        "https://huggingface.co/OWG/resnet-50/raw/main/onnx/model.onnx",
        local_model_path,
    )
    try:
        tract_python.TractModel.load_from_path(local_model_path)
    except RuntimeError as exp:
        # expect fail at load time since tract error
        assert "invalid wire type value: 6" in exp.args[0]


def test_wrong_inputs_name():
    tm = tract_python.TractModel.load_from_path(assets_dir / "test_simple_nnef")
    init_input = np.arange(6).reshape(1, 2, 3).astype(np.float32)
    try:
        tm.run(my_wrong_input_name=init_input)
    except RuntimeError as exp:
        assert 'No node found for name: "my_wrong_input_name"' in exp.args[0]


def test_missing_input():
    tm = tract_python.TractModel.load_from_path(assets_dir / "test_simple_nnef")
    try:
        tm.run()
    except RuntimeError as exp:
        assert 'input with id: \\"input_0\\" not provided' in exp.args[0]


def test_wrong_input_type():
    tm = tract_python.TractModel.load_from_path(assets_dir / "test_simple_nnef")
    init_input = np.arange(6).reshape(1, 2, 3)
    try:
        tm.run(input_0=init_input)
    except RuntimeError as exp:
        assert (
            'Error while running plan: "Evaluating #0 \\"input_0\\" Source: output 0,'
            " expected 1,S,3,F32, got 1,2,3,I64"
        ) in exp.args[0]
