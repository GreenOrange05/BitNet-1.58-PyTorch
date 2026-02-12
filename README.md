# BitNet b1.58 Implementation in PyTorch

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

A clean, from-scratch implementation of the **Microsoft BitNet b1.58** architecture.

This project benchmarks a 1.58-bit quantized Transformer against a standard FP32 Transformer, demonstrating that extreme quantization (weights restricted to `{-1, 0, 1}`) can match the perplexity of full-precision models while offering massive memory efficiency.

**Trained on:** The complete Sherlock Holmes canon.

---

## The Results: 1.58-bit vs. FP32

I ran a controlled experiment comparing a standard GPT-style Transformer against my BitNet implementation. Both models were trained for equivalent epochs with identical parameters (~5M params).

| Metric | BitNet b1.58 (This Repo) | Standard FP32 (Baseline) |
| :--- | :--- | :--- |
| **Convergence Speed** | **2,750 Steps** | 4,750 Steps |
| **Final Val Loss** | **1.1573** | 1.1596 |
| **Model Size (Uncompressed)** | **19 MB** | 19 MB |
| **Model Size (Theoretical 2-bit)** | **~1.2 MB** | N/A |
| **Memory Footprint** | **16x Smaller** | Baseline |

### Training Loss Comparison
![Loss Curve](images/bitnet_victory.png)

> **Key Finding:** The BitNet model converged to the target loss (1.15) nearly **2x faster** than the Standard model and produced subjectively better text generation with fewer hallucinations.

---

## Technical Implementation

This is not a wrapper around existing libraries. The core 1.58-bit quantization is implemented from scratch:

* **Custom Autograd Function:** Implements a `BitLinear` layer with a **Straight-Through Estimator (STE)** to allow gradients to flow through the non-differentiable `round()` function.
* **RMSNorm:** Replaced standard LayerNorm to stabilize quantized training.
* **Activation:** Switched to GELU for smoother gradient flow in the quantized latent space.

### Core Code Snippet (`BitLinear`)
```python
class BitLinear(nn.Linear):
    def forward(self, x):
        # Weight Quantization: {-1, 0, 1}
        w = self.weight
        scale = 1.0 / (w.abs().mean().clamp(min=1e-5))
        w_quant = (w * scale).round().clamp(-1, 1) / scale
        
        # Straight-Through Estimator (Magic happens here)
        w_quant = (w_quant - w).detach() + w
        return F.linear(x, w_quant)
