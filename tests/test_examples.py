import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_example_module(relative_path: str):
    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example_scripts_run_and_export(tmp_path):
    modules = [
        "examples/image_classification/discover_cifar10.py",
        "examples/multimodal/discover_vision_language.py",
        "examples/edge/discover_mobile_net.py",
        "examples/nlp/discover_transformer.py",
    ]

    for relative_path in modules:
        module = load_example_module(relative_path)
        output_path = tmp_path / f"{Path(relative_path).stem}.py"
        result = module.main(output_path=str(output_path))
        assert output_path.exists(), f"{relative_path} did not export a file"
        assert result is not None
