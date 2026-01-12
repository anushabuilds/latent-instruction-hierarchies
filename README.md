# Instruction Hierarchy: Mechanistic Interpretability of Conflicting Instructions

A research codebase for investigating how transformer language models handle conflicting instructions with different priorities through mechanistic interpretability and activation steering.

## Overview

This project explores:
- How language models resolve conflicts between high-priority and low-priority instructions
- Representation of instruction hierarchy in transformer residual streams
- Activation steering techniques to control model behavior in conflict scenarios
- Layer-wise analysis of instruction priority encoding

## Features

- **Modular architecture** for reproducible experiments
- **Activation extraction** from transformer layers
- **Steering direction computation** via contrastive activations
- **Multi-layer steering analysis** with comprehensive evaluation metrics
- **Visualization tools** for dose-response curves and heatmaps
- **Colab-compatible notebook** for interactive experimentation

## Installation

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/instruction-hierarchy.git
cd instruction-hierarchy

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .

# Or install with extras
pip install -e ".[dev,jupyter]"
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-compatible GPU (recommended for model inference)

See [requirements.txt](requirements.txt) for full dependency list.

## Quick Start

### Using the Python API

```python
from instruction_hierarchy.models import load_model_and_tokenizer, get_last_token_residual
from instruction_hierarchy.prompts import build_priority_prompt, first_line
from instruction_hierarchy.steering import compute_steering_direction, generate_with_steering

# Load model
model, tokenizer = load_model_and_tokenizer("google/gemma-3-1b-it")

# Create conflicting prompts
prompt_alpha = build_priority_prompt("ALPHA", style=0)
prompt_beta = build_priority_prompt("BETA", style=0)

# Extract activations
layer = 17
resid_alpha = get_last_token_residual(model, tokenizer, prompt_alpha, layer)
resid_beta = get_last_token_residual(model, tokenizer, prompt_beta, layer)

# Compute steering direction
direction = compute_steering_direction(
    torch.stack([resid_alpha]),
    torch.stack([resid_beta])
)

# Generate with steering
output = generate_with_steering(
    model, tokenizer, prompt_alpha,
    layer=layer, direction=direction, scale=10.0
)
print(first_line(output))
```

### Using the Jupyter Notebook

Open [notebooks/instruction_conflicts_demo.ipynb](notebooks/instruction_conflicts_demo.ipynb) in Jupyter or Colab:

```bash
jupyter notebook notebooks/instruction_conflicts_demo.ipynb
```

Or click here to open in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/instruction-hierarchy/blob/main/notebooks/instruction_conflicts_demo.ipynb)

## Project Structure

```
instruction-hierarchy/
├── src/
│   └── instruction_hierarchy/
│       ├── models/           # Model loading and activation extraction
│       ├── steering/         # Activation steering utilities
│       ├── prompts/          # Prompt generation and templates
│       ├── evaluation/       # Metrics and analysis
│       └── utils/            # Visualization and helpers
├── notebooks/
│   └── instruction_conflicts_demo.ipynb  # Interactive demo
├── scripts/
│   ├── run_steering_sweep.py    # Layer/scale sweep experiments
│   └── analyze_results.py       # Result analysis and visualization
├── configs/
│   └── experiment_config.yaml   # Experiment configurations
├── requirements.txt
├── setup.py
└── README.md
```

## Experiments

### Basic Steering Experiment

```bash
python scripts/run_steering_sweep.py \
    --model google/gemma-3-1b-it \
    --layers 1 10 17 25 35 \
    --scales -50 -20 0 20 50 \
    --output results/steering_sweep.json
```

### Full Layer Sweep

```bash
python scripts/run_steering_sweep.py \
    --model google/gemma-3-1b-it \
    --sweep-all-layers \
    --output results/full_sweep.json
```

## Configuration

Experiments can be configured via YAML files in [configs/](configs/):

```yaml
model:
  name: "google/gemma-3-1b-it"
  device_map: "auto"

steering:
  layers: [1, 10, 17, 25, 35]
  scales: [-50, -20, -10, 0, 10, 20, 50]

prompts:
  num_paraphrases: 10
  conflict_types: ["priority", "temporal"]
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{instruction-hierarchy-2024,
  title={Mechanistic Interpretability of Instruction Hierarchy in Language Models},
  author={Anusha Mujumdar},
  year={2026},
  url={https://github.com/anushabuilds/latent-instruction-hierarchies.git}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

<!-- ## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request -->

## Acknowledgments

- Built with [Transformers](https://huggingface.co/transformers/) by Hugging Face
- Uses [Gemma](https://ai.google.dev/gemma) models from Google

## Contact

For questions or collaborations please open an issue.
