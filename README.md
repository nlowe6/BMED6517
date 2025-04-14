# GRN Reconstruction Toolkit

A comprehensive toolkit for Gene Regulatory Network (GRN) reconstruction using various methods, including traditional statistical approaches and deep learning techniques.

## Features

- Multiple GRN reconstruction methods:
  - Correlation-based methods (Pearson, Spearman)
  - Mutual Information
  - LASSO regression
  - GENIE3 (Random Forest)
  - ARACNE
  - CLR (Context Likelihood of Relatedness)
  - TIGRESS
  - WGCNA
  - Deep Neural Network (DNN)

- GPU acceleration support for neural network methods
- Comprehensive evaluation metrics
- Visualization tools for results analysis
- Support for multiple dataset formats

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/grn-reconstruction.git
cd grn-reconstruction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```python
from grn_methods import GRNReconstructor
from data_loader import GRNDataLoader
from evaluation import GRNEvaluator

# Load data
loader = GRNDataLoader("path/to/dataset")
expression_matrix, gene_ids = loader.load_expression_matrix()
tf_expression, target_expression, tf_ids, target_ids = loader.get_tf_target_split(expression_matrix, gene_ids)

# Initialize reconstructor
reconstructor = GRNReconstructor(method='dnn')  # Choose from: correlation, mutual_info, lasso, genie3, aracne, clr, tigress, wgcna, dnn

# Reconstruct network
predicted_matrix = reconstructor.fit(tf_expression, target_expression)

# Evaluate results
evaluator = GRNEvaluator(ground_truth_matrix)
metrics = evaluator.evaluate(predicted_matrix)
```

## Project Structure

```
grn-reconstruction/
├── data/                    # Example datasets
├── results/                 # Output directory for results
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── grn_methods.py      # GRN reconstruction methods
│   ├── evaluation.py       # Evaluation metrics
│   └── visualization.py    # Visualization tools
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## GPU Acceleration

The toolkit supports GPU acceleration for neural network methods. To use GPU:

1. Install CUDA-compatible PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. The code will automatically use GPU if available.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{grn_reconstruction_toolkit,
  title = {GRN Reconstruction Toolkit},
  author = {Your Name},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/grn-reconstruction}
}
``` 