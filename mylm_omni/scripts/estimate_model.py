"""Estimate parameter count and first-order training memory for a MyLMConfig JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mylm_omni.modeling import MyLMConfig, MyLMForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--dtype-bytes", type=float, default=2.0,
                        help="parameter/gradient bytes; e.g. 2 for BF16, 4 for FP32")
    parser.add_argument("--optimizer-state-bytes", type=float, default=8.0,
                        help="bytes per parameter for two FP32 Adam moments")
    parser.add_argument("--master-weight-bytes", type=float, default=0.0,
                        help="additional master weights bytes per parameter, if used")
    parser.add_argument("--activation-gb", type=float, default=8.0,
                        help="rough user-supplied allowance for activations, CUDA workspaces, and cache")
    args = parser.parse_args()

    raw = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    config = MyLMConfig(**raw)
    model = MyLMForCausalLM(config)
    count = model.num_parameters()
    parameters_gb = count * args.dtype_bytes / 1e9
    gradients_gb = count * args.dtype_bytes / 1e9
    optimizer_gb = count * args.optimizer_state_bytes / 1e9
    master_gb = count * args.master_weight_bytes / 1e9
    total = parameters_gb + gradients_gb + optimizer_gb + master_gb + args.activation_gb
    print(json.dumps({
        "parameter_count": count,
        "parameter_count_billions": count / 1e9,
        "config_vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "unique_layers": config.num_layers,
        "shared_recurrent_depth": config.max_recurrent_steps,
        "image_feature_size": config.image_feature_size,
        "memory_estimate_gb": {
            "parameters": parameters_gb,
            "gradients": gradients_gb,
            "adam_moments": optimizer_gb,
            "master_weights": master_gb,
            "activation_and_workspace_allowance": args.activation_gb,
            "total_very_rough": total,
        },
        "warning": "First-order estimate only. Measure a real forward/backward step and add >=15-25% headroom.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
