# LGFormer

Official PyTorch implementation of **[LGFormer: Integrating Local and Global Representations for EEG Decoding](https://iopscience.iop.org/article/10.1088/1741-2552/adc5a3)**.

## Introduction

LGFormer is a general-purpose convolutional-transformer model for Electroencephalography (EEG) decoding. The architecture consists of:
- Temporal-Spatial Encoder (TSE)
- Local-Enhanced Transformer (LET)
- Classification head

The lightweight design efficiently combines local and global representations for fast training and accurate EEG signal decoding.

## Quick Start

### Input Format
Ensure your input data has the shape **(B, 1, channel, sequence_length)**, where:
- **B**: Batch size
- **channel**: Number of EEG electrode channels
- **sequence_length**: Number of temporal sample points

### Implementation
To use LGFormer, specify your EEG decoding task parameters: number of channels (`in_channel`), sequence length (`seq_len`), and number of classes (`num_classes`):

```python
from LGFormer import LGFormer

# Initialize the model
model = LGFormer(in_channel=22, seq_len=1000, num_classes=4)

# Forward pass
output = model(input_data)  # input_data shape: [B, 1, channel, sequence_length]
```

### Example Configurations
Configurations used in the paper:

| Dataset | Channels | Sequence Length | Classes | Code |
|---------|----------|----------------|---------|------|
| BCI42A | 22 | 1000 (4s @ 250Hz) | 4 | `LGFormer(in_channel=22, seq_len=1000, num_classes=4)` |
| BCI42B | 3 | 1000 | 2 | `LGFormer(in_channel=3, seq_len=1000, num_classes=2)` |
| Cognitive workload | 28 | 400 | 3 | `LGFormer(in_channel=28, seq_len=400, num_classes=3)` |
| ERN | 56 | 280 | 2 | `LGFormer(in_channel=56, seq_len=280, num_classes=2)` |

The implementation includes default hyperparameters from the paper, and we alse provide interfaces that can be quickly adjusted for your specific task requirements.

## Citation
If you find our paper/code useful, please consider citing our work:
```bibtex
@article{yang2025lgformer,
  title={LGFormer: Integrating local and global representations for EEG decoding},
  author={Yang, Wenjie and Wang, Xingfu and Qi, Wenxia and Wang, Wei},
  journal={Journal of Neural Engineering},
  year={2025}
}
```

