# Google Colab Setup Guide - Fastest Settings

## 🚀 Recommended Colab Runtime Settings

### Hardware Accelerator
- **Free Tier**: Select **T4 GPU** (15GB VRAM)
- **Paid/Compute Units**: Select **A100 GPU** (40GB VRAM) for 2-3x faster training

### Runtime Configuration
1. **Runtime type**: Python 3
2. **Hardware accelerator**: T4 GPU (or A100 if available)
3. **Runtime version**: Latest (recommended)

## ⚡ Performance Optimizations (Auto-Applied)

The code automatically detects Colab and optimizes settings:

### Automatic Optimizations:
- ✅ **Mixed Precision Training (AMP)**: Enabled by default - ~2x faster training
- ✅ **Pin Memory**: Enabled for faster GPU data transfer
- ✅ **Batch Size**: Auto-increased on A100 (128) or T4 (96)
- ✅ **DataLoader Workers**: Set to 0 (Colab doesn't support multiprocessing)

### Manual Optimizations Available:

In `Config` class, you can adjust:

```python
batch_size = 64          # Increase to 128-256 on A100 if memory allows
use_amp = True           # Mixed precision (already enabled)
compile_model = False    # Set True for PyTorch 2.0+ (20-30% speedup)
```

## 📊 Expected Performance

### T4 GPU (Free Tier):
- **Training Speed**: ~2-3 seconds per epoch
- **Total Time (20 epochs)**: ~1-2 minutes per fold
- **7 Folds**: ~10-15 minutes total

### A100 GPU (Paid):
- **Training Speed**: ~0.5-1 second per epoch  
- **Total Time (20 epochs)**: ~20-40 seconds per fold
- **7 Folds**: ~3-5 minutes total

## 🔧 Additional Colab Tips

1. **Check GPU Availability**:
   ```python
   import torch
   print(f"CUDA Available: {torch.cuda.is_available()}")
   print(f"GPU: {torch.cuda.get_device_name(0)}")
   ```

2. **Monitor GPU Usage**:
   - Runtime → Manage sessions → Monitor GPU usage
   - Or use: `!nvidia-smi`

3. **Save Output Files**:
   - Files are saved to Colab's `/content/` directory
   - Download `research_report.html` after completion
   - Or mount Google Drive to save directly

4. **Install Dependencies** (if needed):
   ```python
   !pip install torch torchvision arch statsmodels yfinance scikit-learn
   ```

## ⚠️ Common Issues

### Out of Memory (OOM):
- Reduce `batch_size` to 32 or 16
- Reduce `num_paths` for Monte Carlo simulations
- Reduce `hidden_dim` to 64

### Slow Training:
- Ensure GPU is selected (not CPU)
- Check GPU utilization with `nvidia-smi`
- Enable `compile_model = True` if PyTorch 2.0+

### Import Errors:
- Run: `!pip install arch statsmodels` if baselines fail
- The code will gracefully skip missing baselines

## 📈 Performance Comparison

| Setting | CPU | T4 GPU | A100 GPU |
|---------|-----|--------|----------|
| Epoch Time | ~30-60s | ~2-3s | ~0.5-1s |
| 20 Epochs | ~10-20 min | ~1-2 min | ~20-40s |
| 7 Folds | ~70-140 min | ~10-15 min | ~3-5 min |

**Recommendation**: Always use GPU for training. The speedup is 10-30x compared to CPU.

