from pathlib import Path

from architectai import Architect
from architectai.core.dsl import ArchitectureDSL
from architectai.hardware.constraints import HardwareConstraints
from architectai.hardware.edge_optimizer import EdgeOptimizer
from architectai.hardware.quantization import Quantizer


def test_full_pipeline(tmp_path):
    # 1. Build via DSL
    dsl = ArchitectureDSL("pipeline_test")
    graph = dsl.input([3, 32, 32]).conv2d(32).relu().linear(10).build()

    # 2. Hardware optimization & Quantization
    edge_opt = EdgeOptimizer(target_device="jetson_nano")
    opt_graph = edge_opt.optimize(graph)

    quantizer = Quantizer(precision="int8")
    quant_graph = quantizer.quantize(opt_graph)

    # 3. Discovery run via Architect orchestrator
    architect = Architect()
    constraints = HardwareConstraints(max_params=1_000_000, max_memory_mb=512)
    discovered = architect.discover(
        task="image_classification", iterations=3, constraints=constraints
    )

    target_graph = discovered or quant_graph

    # 4. Code compilation & export
    output_file = tmp_path / "compiled_pipeline.py"
    architect.export(target_graph, str(output_file))

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")
    assert "class SDNNGeneratedModel" in content or "import torch" in content
