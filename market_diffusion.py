import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats
import copy
import json
from datetime import datetime

# Try to import optional dependencies for baselines
try:
    from arch import arch_model  # For GARCH
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("Warning: 'arch' package not found. GARCH baseline will be disabled.")

try:
    from statsmodels.tsa.ar_model import AutoReg  # For AR(1)
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: 'statsmodels' package not found. AR(1) baseline will be disabled.")

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION
# ==========================================

def detect_colab():
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

class Config:
    use_mock_data = False     # Set to True if you don't have internet
    ticker = "SPY"            # Target ETF (what we predict)
    
    # Multi-feature conditioning - additional signals the model can learn from
    # NOTE: Multi-feature requires more training data to be effective.
    # Start with single feature, enable multi-feature once base model works.
    #
    # Options:
    #   ["SPY"]                           - Single feature (recommended to start)
    #   ["SPY", "^VIX"]                   - Add volatility signal
    #   ["SPY", "^VIX", "TLT", "GLD"]     - Full multi-feature
    #
    feature_tickers = ["SPY"]  # Start simple - single feature mode
    
    # Uncomment below for multi-feature (requires more epochs/data):
    # feature_tickers = ["SPY", "^VIX", "TLT", "GLD"]
    
    seq_length = 128          # Total window size (64 past + 64 future)
    cond_length = 64          # How much history the model sees
    pred_length = 64          # How much future the model generates
    
    # Model Hyperparameters - SIMPLIFIED
    timesteps = 200           # Fewer diffusion steps (simpler)
    batch_size = 128           # Increase to 128-256 on GPU if memory allows
    lr = 1e-3                 # Higher LR for simple model
    epochs = 50               # Simple model converges fast
    hidden_dim = 64           # Simple architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Noise schedule
    noise_schedule = 'cosine'
    
    # Sampling settings - SIMPLIFIED
    sampling_method = 'ddim'  # 'ddim' (fast) or 'ddpm' (slow but potentially better)
    ddim_eta = 1.0            # 1.0 = fully stochastic
    ddim_steps = 20           # Fast sampling
    sampling_temperature = 1.0  # Standard temperature
    
    # Uncertainty calibration
    # Options:
    #   'calibrated' - Use validation-calibrated scale (more accurate but slower)
    #   'auto'       - Regime-aware scaling based on recent volatility (fast, default)
    #   'none'       - No scaling
    #   float        - Fixed multiplier (e.g., 1.5)
    variance_scale = 'none'  # No post-processing - trust the model
    
    # Diversity noise: Add small noise to final samples
    diversity_noise_std = 0.0  # No artificial noise
    
    # Coverage adjustment: Small multiplier applied to path spread 
    # This helps reach target 90% coverage without heavy post-processing
    coverage_boost = 1.0  # No boost - trust the model
    
    # Monte Carlo Dropout - DISABLED for speed (was causing 3x slowdown)
    use_mc_dropout = False    # Disable for faster inference
    mc_samples = 1            # Only 1 forward pass
    
    # Model type: 'diffusion' or 'probabilistic'
    # For generative model projects, use 'diffusion'
    model_type = 'diffusion'
    
    # Colab/GPU Optimizations
    use_amp = True            # Mixed precision training (faster on GPU)
    pin_memory = True         # Faster data transfer to GPU
    num_workers = 0 if detect_colab() else 2  # 0 for Colab (multiprocessing issues), 2-4 for local
    compile_model = False     # Set True for PyTorch 2.0+ (can speed up 20-30%)
    
    # Monte Carlo Settings
    num_paths = 100           # Number of simulations
    
    # ==========================================
    # WALK-FORWARD CROSS-VALIDATION SETTINGS
    # ==========================================
    # 
    # Walk-Forward CV is the gold standard for time series validation.
    # Unlike k-fold CV, it respects temporal ordering to prevent data leakage.
    #
    # How it works:
    # ┌──────────────────────────────────────────────────────────────────┐
    # │ Fold 1: [====TRAIN====][=VAL=][TEST]                             │
    # │ Fold 2: [======TRAIN======][=VAL=][TEST]                         │
    # │ Fold 3: [========TRAIN========][=VAL=][TEST]                     │
    # │ Fold 4: [==========TRAIN==========][=VAL=][TEST]                 │
    # │ Fold 5: [============TRAIN============][=VAL=][TEST]             │
    # └──────────────────────────────────────────────────────────────────┘
    #
    # The training window EXPANDS with each fold, but we NEVER train on
    # future data. This simulates real-world deployment where you'd
    # retrain periodically on all available historical data.
    #
    # Our folds (13 total, testing 2008-2024):
    #   Fold 1:  Train 2000-2006, Val 2007, Test 2008 (GFC crash!)
    #   Fold 2:  Train 2000-2007, Val 2008, Test 2009 (recovery)
    #   Fold 3:  Train 2000-2008, Val 2009, Test 2010
    #   Fold 4:  Train 2000-2009, Val 2010, Test 2011
    #   Fold 5:  Train 2000-2012, Val 2013, Test 2014
    #   Fold 6:  Train 2000-2014, Val 2015, Test 2016
    #   Fold 7:  Train 2000-2016, Val 2017, Test 2018
    #   Fold 8:  Train 2000-2017, Val 2018, Test 2019
    #   Fold 9:  Train 2000-2018, Val 2019, Test 2020 (COVID crash!)
    #   Fold 10: Train 2000-2019, Val 2020, Test 2021
    #   Fold 11: Train 2000-2020, Val 2021, Test 2022
    #   Fold 12: Train 2000-2021, Val 2022, Test 2023
    #   Fold 13: Train 2000-2022, Val 2023, Test 2024
    #
    use_walk_forward_cv = True  # Set False to use simple single split
    
    # Walk-forward fold definitions: (train_end, val_end, test_end)
    # Each tuple defines the END year for each period
    walk_forward_folds = [
        # Early folds - includes 2008 crisis testing
        ("2006-12-31", "2007-12-31", "2008-12-31"),  # Test on 2008 GFC!
        ("2007-12-31", "2008-12-31", "2009-12-31"),  # Test on 2009 recovery
        ("2008-12-31", "2009-12-31", "2010-12-31"),
        ("2009-12-31", "2010-12-31", "2011-12-31"),
        # Mid folds
        ("2012-12-31", "2013-12-31", "2014-12-31"),
        ("2014-12-31", "2015-12-31", "2016-12-31"),
        ("2016-12-31", "2017-12-31", "2018-12-31"),
        ("2017-12-31", "2018-12-31", "2019-12-31"),
        # Late folds - includes COVID and rate hikes
        ("2018-12-31", "2019-12-31", "2020-12-31"),  # Test on COVID crash
        ("2019-12-31", "2020-12-31", "2021-12-31"),
        ("2020-12-31", "2021-12-31", "2022-12-31"),
        ("2021-12-31", "2022-12-31", "2023-12-31"),
        ("2022-12-31", "2023-12-31", "2024-12-31"),
        ("2023-12-31", "2024-12-31", "2025-11-28"),  # Test on 2025 (current year)
    ]
    
    data_start = "2000-01-01"  # Include dot-com crash (2000-02) and GFC (2008) for better tail modeling
    
    # Number of test windows to evaluate per fold
    num_test_windows = 30  # More windows = more reliable statistics
    
    # Early stopping patience (stop if val loss doesn't improve)
    early_stopping_patience = 25

# ==========================================
# 2. DATA PIPELINE
# ==========================================
def fetch_data(ticker="SPY", start_date="2010-01-01", end_date=None, feature_tickers=None):
    """
    Fetches data for the target ticker and optional feature tickers.
    Returns raw data WITHOUT scaling (scaling done per-fold to prevent leakage).
    
    Args:
        ticker: Primary target ticker (SPY)
        start_date: Start date for data
        end_date: End date for data  
        feature_tickers: List of additional tickers for multi-feature conditioning
                        If None, only fetches the primary ticker
    
    Returns:
        DataFrame with 'returns' column for target and 'feat_X_returns' for features
    """
    # Determine all tickers to fetch
    if feature_tickers is None or len(feature_tickers) <= 1:
        all_tickers = [ticker]
    else:
        all_tickers = feature_tickers if ticker in feature_tickers else [ticker] + feature_tickers
    
    print(f"Downloading data for {all_tickers} from {start_date} to {end_date or 'now'}...")
    
    # Download all tickers at once (more efficient)
    try:
        df = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    except Exception as e:
        print(f"  Warning: Multi-ticker download failed ({e}). Falling back to single ticker.")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        all_tickers = [ticker]
    
    # Handle single ticker case
    if isinstance(df, pd.Series):
        df = df.to_frame(name=ticker)
    
    # Ensure column names are strings (not tuples)
    df.columns = [str(c) for c in df.columns]
    
    # Create result dataframe
    data = pd.DataFrame(index=df.index)
    
    # Primary target: SPY price and returns
    primary_col = ticker if ticker in df.columns else df.columns[0]
    data['price'] = df[primary_col]
    data['returns'] = np.log(data['price'] / data['price'].shift(1))
    
    # Additional features as returns (percentage changes, scaled similarly)
    valid_features = 0
    for i, t in enumerate(all_tickers):
        if t == ticker or t == primary_col:
            continue
        col_name = t if t in df.columns else None
        if col_name is None:
            # Try to find matching column
            for c in df.columns:
                if t in c:
                    col_name = c
                    break
        
        if col_name is not None and col_name in df.columns:
            # Use log returns for price series, raw values for VIX
            if t == "^VIX":
                # VIX is already a volatility measure - use normalized level
                data[f'feat_{i}_raw'] = df[col_name]
                data[f'feat_{i}_returns'] = (df[col_name] / df[col_name].rolling(20).mean() - 1)
            else:
                data[f'feat_{i}_returns'] = np.log(df[col_name] / df[col_name].shift(1))
            valid_features += 1
    
    # Forward-fill then backward-fill missing feature values (handles different start dates)
    # This is important because VIX, HYG etc may have different date ranges
    feature_cols = [c for c in data.columns if c.startswith('feat_')]
    if feature_cols:
        data[feature_cols] = data[feature_cols].ffill().bfill()
    
    # Only drop rows where PRIMARY returns are NaN (not features)
    data = data.dropna(subset=['returns'])
    
    # Ensure index is datetime
    data.index = pd.to_datetime(data.index)
    
    # Report what we got
    print(f"  Loaded {len(data)} rows with {valid_features} additional features")
    
    # Warn if no data
    if len(data) == 0:
        raise ValueError(f"No data available for {ticker} in the specified date range")
    
    return data


def split_data_by_date(df, train_end, val_end, test_end, data_start="2010-01-01"):
    """
    Split dataframe by date ranges for walk-forward CV.
    
    CRITICAL: This prevents data leakage by ensuring strict temporal ordering.
    - Training data: [data_start, train_end]
    - Validation data: (train_end, val_end]
    - Test data: (val_end, test_end]
    
    The scaler is FIT ONLY on training data, then applied to val/test.
    Handles both single-feature and multi-feature cases.
    """
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    test_end = pd.to_datetime(test_end)
    data_start = pd.to_datetime(data_start)
    
    # Split by date
    train_df = df[(df.index >= data_start) & (df.index <= train_end)].copy()
    val_df = df[(df.index > train_end) & (df.index <= val_end)].copy()
    test_df = df[(df.index > val_end) & (df.index <= test_end)].copy()
    
    # Fit scaler ONLY on training data (prevents leakage!)
    scaler = StandardScaler()
    train_returns = train_df['returns'].values.reshape(-1, 1)
    
    # Safety check: ensure we have data
    if len(train_returns) == 0:
        raise ValueError(f"No training data available for period {data_start} to {train_end}")
    
    scaler.fit(train_returns)
    
    # Transform primary returns
    train_df['returns_scaled'] = scaler.transform(train_df['returns'].values.reshape(-1, 1))
    val_df['returns_scaled'] = scaler.transform(val_df['returns'].values.reshape(-1, 1))
    test_df['returns_scaled'] = scaler.transform(test_df['returns'].values.reshape(-1, 1))
    
    # Scale additional features independently (each has its own scaler)
    feature_cols = [c for c in df.columns if c.startswith('feat_') and c.endswith('_returns')]
    feature_scalers = {}
    
    for feat_col in feature_cols:
        # Get non-NaN values for fitting
        train_feat_values = train_df[feat_col].dropna().values.reshape(-1, 1)
        
        if len(train_feat_values) == 0:
            # Skip this feature if no valid training data
            continue
            
        feat_scaler = StandardScaler()
        feat_scaler.fit(train_feat_values)
        feature_scalers[feat_col] = feat_scaler
        
        scaled_col = feat_col.replace('_returns', '_scaled')
        
        # Fill NaN before scaling, then scale
        for split_df in [train_df, val_df, test_df]:
            feat_values = split_df[feat_col].fillna(0).values.reshape(-1, 1)
            split_df[scaled_col] = feat_scaler.transform(feat_values)
    
    # Store feature scalers in the main scaler object for later use
    scaler.feature_scalers = feature_scalers
    
    return train_df, val_df, test_df, scaler


class FinancialDataset(Dataset):
    """
    Dataset that creates overlapping windows for diffusion model training.
    
    IMPORTANT: This should only be instantiated with data from a SINGLE temporal split
    (train, val, or test) to prevent data leakage across time periods.
    
    Supports multi-feature conditioning where additional features (VIX, TLT, etc.)
    are provided as extra channels in the history tensor.
    """
    def __init__(self, dataframe, seq_len, cond_len, use_scaled=True):
        self.seq_len = seq_len
        self.cond_len = cond_len
        self.data = dataframe
        
        # Primary returns (target)
        col = 'returns_scaled' if use_scaled and 'returns_scaled' in dataframe.columns else 'returns'
        self.returns = torch.tensor(dataframe[col].values, dtype=torch.float32).unsqueeze(-1)
        
        # Additional features for conditioning
        feat_cols = sorted([c for c in dataframe.columns if '_scaled' in c and c != 'returns_scaled'])
        
        if feat_cols:
            # Stack features: (N, num_features)
            feat_data = dataframe[feat_cols].values.astype(np.float32)
            self.features = torch.tensor(feat_data, dtype=torch.float32)
            self.num_features = len(feat_cols) + 1  # +1 for primary returns
        else:
            self.features = None
            self.num_features = 1
        
        # Store raw returns for baseline comparisons
        self.raw_returns = dataframe['returns'].values

    def __len__(self):
        return max(0, len(self.returns) - self.seq_len)

    def __getitem__(self, idx):
        # Full window: History + Future (primary returns only for prediction)
        full_window = self.returns[idx : idx + self.seq_len]
        
        # Split primary returns
        x_history_primary = full_window[:self.cond_len]  # (64, 1)
        x_future = full_window[self.cond_len:]           # (64, 1)
        
        # Build history with all features
        if self.features is not None:
            # Additional features for history window
            feat_window = self.features[idx : idx + self.cond_len]  # (64, num_feat)
            # Concatenate: primary returns + features -> (64, 1 + num_feat)
            x_history = torch.cat([x_history_primary, feat_window], dim=-1)
        else:
            x_history = x_history_primary
        
        # Permute to (Channels, Length) for 1D Conv
        # History: (num_features, 64), Future: (1, 64)
        return x_history.permute(1, 0), x_future.permute(1, 0)
    
    def get_raw_window(self, idx):
        """Get raw (unscaled) returns for a window - used for baseline comparisons."""
        return self.raw_returns[idx : idx + self.seq_len]

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
class EMA:
    """
    Exponential Moving Average for model parameters.
    Standard technique for diffusion models to stabilize training and improve generation.
    """
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]


# ==========================================
# ALTERNATIVE: PROBABILISTIC FORECASTER
# ==========================================
# This is a more appropriate model for financial time series.
# Instead of diffusion (which learns to denoise), this directly
# predicts distribution parameters and optimizes CRPS.

class GaussianCRPSLoss(nn.Module):
    """
    Closed-form CRPS loss for Gaussian distributions.
    
    CRPS(N(μ,σ), y) = σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
    where z = (y - μ) / σ, φ is standard normal PDF, Φ is CDF
    
    This is differentiable and directly optimizes what we evaluate on!
    Reference: Gneiting & Raftery (2007)
    """
    def __init__(self):
        super().__init__()
        self.inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
    
    def forward(self, mu, sigma, target):
        """
        Args:
            mu: (B, L) predicted means
            sigma: (B, L) predicted stds (must be positive)
            target: (B, L) actual values
        Returns:
            Scalar CRPS loss (non-negative, lower is better)
        """
        # Ensure sigma is positive
        sigma = torch.clamp(sigma, min=1e-6)
        
        # Standardized residual
        z = (target - mu) / sigma
        
        # Standard normal PDF: φ(z) = exp(-z²/2) / √(2π)
        phi = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
        
        # Standard normal CDF: Φ(z) = 0.5 * (1 + erf(z/√2))
        Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        
        # CRPS formula: σ * [z*(2Φ(z)-1) + 2φ(z) - 1/√π]
        crps = sigma * (z * (2 * Phi - 1) + 2 * phi - self.inv_sqrt_pi)
        
        return crps.mean()


class ProbabilisticForecaster(nn.Module):
    """
    Direct probabilistic forecaster that predicts (mean, std) for each timestep.
    
    Architecture:
    - Transformer encoder for history
    - Autoregressive decoder with distribution output
    - Trained with CRPS loss (proper scoring rule)
    
    This is simpler, faster, and more appropriate for financial forecasting.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden = config.hidden_dim
        self.pred_length = config.pred_length
        self.cond_length = config.cond_length
        
        # History encoder
        self.input_proj = nn.Linear(1, self.hidden)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.cond_length, self.hidden) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=4,
            dim_feedforward=self.hidden * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Volatility-aware context
        self.vol_proj = nn.Sequential(
            nn.Linear(3, self.hidden),  # vol features
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        
        # Decoder: predicts all future steps at once (non-autoregressive for speed)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden * 2, self.hidden * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden * 2, self.hidden * 2),
            nn.SiLU(),
            nn.Linear(self.hidden * 2, config.pred_length * 2),  # mean and log_std for each step
        )
        
    def _compute_vol_features(self, history):
        """Extract volatility features from history."""
        # history: (B, L)
        std = history.std(dim=-1, keepdim=True)
        recent_std = history[:, -16:].std(dim=-1, keepdim=True)
        mean = history.mean(dim=-1, keepdim=True)
        return torch.cat([std, recent_std, mean], dim=-1)  # (B, 3)
    
    def forward(self, history):
        """
        Args:
            history: (B, 1, L) or (B, L) - historical returns
        
        Returns:
            mu: (B, pred_length) - predicted means
            sigma: (B, pred_length) - predicted stds
        """
        # Handle input shape
        if history.dim() == 3:
            history = history.squeeze(1)  # (B, L)
        
        B, L = history.shape
        
        # Encode history
        x = history.unsqueeze(-1)  # (B, L, 1)
        x = self.input_proj(x)  # (B, L, hidden)
        x = x + self.pos_encoding[:, :L, :]
        
        encoded = self.encoder(x)  # (B, L, hidden)
        
        # Pool to get context vector (use last position + mean)
        context = torch.cat([encoded[:, -1, :], encoded.mean(dim=1)], dim=-1) / 2
        context = encoded.mean(dim=1)  # Simple mean pooling
        
        # Volatility context
        vol_feat = self._compute_vol_features(history)
        vol_context = self.vol_proj(vol_feat)  # (B, hidden)
        
        # Combined context
        combined = torch.cat([context, vol_context], dim=-1)  # (B, hidden*2)
        
        # Decode to distribution parameters
        output = self.decoder(combined)  # (B, pred_length * 2)
        
        # Split into mean and log_std
        mu = output[:, :self.pred_length]
        log_sigma = output[:, self.pred_length:]
        
        # Convert log_sigma to sigma (ensure positive)
        sigma = torch.exp(log_sigma.clamp(-5, 2))  # Clamp for stability
        
        return mu, sigma
    
    def sample(self, history, num_samples=100):
        """
        Generate Monte Carlo samples for evaluation.
        
        Args:
            history: (B, 1, L) historical returns
            num_samples: number of paths to generate
        
        Returns:
            samples: (B, num_samples, pred_length)
        """
        mu, sigma = self.forward(history)  # (B, pred_length), (B, pred_length)
        
        # Sample from predicted Gaussians
        B, L = mu.shape
        eps = torch.randn(B, num_samples, L, device=mu.device)
        samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps
        
        return samples


def train_probabilistic_model(train_dataset, config, verbose=True):
    """Train the probabilistic forecaster with CRPS loss."""
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=config.pin_memory if config.device == "cuda" else False,
        num_workers=config.num_workers if config.device == "cuda" else 0,
        drop_last=True
    )
    
    model = ProbabilisticForecaster(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # CRPS loss - directly optimizes what we evaluate on!
    crps_loss = GaussianCRPSLoss()
    
    # Learning rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
    
    if verbose:
        print(f"    Training Probabilistic Forecaster (CRPS-optimized)")
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        for history, future in dataloader:
            history = history.to(config.device)
            future = future.to(config.device).squeeze(1)  # (B, pred_length)
            
            # Forward pass
            mu, sigma = model(history)
            
            # CRPS loss
            loss = crps_loss(mu, sigma, future)
            
            # Also add a small NLL term for stability
            nll = 0.5 * (torch.log(sigma ** 2) + ((future - mu) / sigma) ** 2).mean()
            loss = loss + 0.1 * nll
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"    Epoch {epoch+1}/{config.epochs} | CRPS Loss: {avg_loss:.6f}")
    
    return model


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SelfAttention1D(nn.Module):
    """Self-attention for 1D sequences - critical for capturing long-range dependencies."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        residual = x
        
        x = self.norm(x)
        qkv = self.qkv(x)  # (B, 3C, L)
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, L)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, L, L)
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # (B, heads, L, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, L)  # (B, C, L)
        out = self.proj(out)
        
        return out + residual


class CrossAttention1D(nn.Module):
    """Cross-attention between future (query) and history (key/value) - for conditioning."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm_q = nn.GroupNorm(8, channels)
        self.norm_kv = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.kv = nn.Conv1d(channels, channels * 2, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, context):
        # x: (B, C, L) - query (future)
        # context: (B, C, L) - key/value (history)
        B, C, L = x.shape
        residual = x
        
        x = self.norm_q(x)
        context = self.norm_kv(context)
        
        q = self.q(x).reshape(B, self.num_heads, self.head_dim, L).permute(0, 1, 3, 2)
        kv = self.kv(context).reshape(B, 2, self.num_heads, self.head_dim, L)
        k, v = kv[:, 0].permute(0, 1, 3, 2), kv[:, 1].permute(0, 1, 3, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, L)
        out = self.proj(out)
        
        return out + residual


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, emb_dim, use_attention=False, num_heads=4):
        super().__init__()
        self.use_attention = use_attention
        
        # Pre-norm design (more stable training)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        
        # Time/conditioning embedding projection
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, channels * 2),  # Scale and shift
        )
        
        self.act = nn.SiLU()
        
        # Optional attention layers
        if use_attention:
            self.self_attn = SelfAttention1D(channels, num_heads)
            self.cross_attn = CrossAttention1D(channels, num_heads)

    def forward(self, x, embedding, context=None):
        residual = x
        
        # First conv block with pre-norm
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        
        # Add time/conditioning embedding (FiLM: scale and shift)
        emb_out = self.emb_proj(embedding)
        scale, shift = emb_out.chunk(2, dim=-1)
        x = x * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        
        # Second conv block
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        
        x = x + residual
        
        # Attention layers (if enabled)
        if self.use_attention:
            x = self.self_attn(x)
            if context is not None:
                x = self.cross_attn(x, context)
        
        return x


# Note: VolatilityEncoder is replaced by MultiScaleVolatilityEncoder in MarketDiffusion
# Kept as alias for backwards compatibility
VolatilityEncoder = None  # See MultiScaleVolatilityEncoder above


class SimpleMarketDiffusion(nn.Module):
    """
    SIMPLE Diffusion Model for Financial Time Series.
    
    Key design principles:
    - Minimal complexity - just predict noise conditioned on time and history
    - No attention (overkill for this task)
    - Simple MLP-based architecture
    - History encoded via simple 1D convolutions
    """
    def __init__(self, config, num_features=1):
        super().__init__()
        hidden = config.hidden_dim
        self.hidden = hidden
        pred_len = config.pred_length
        cond_len = config.cond_length
        
        # Time embedding (simple)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        
        # History encoder: conv layers to extract features from conditioning
        self.history_encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden // 2, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # Global pooling -> (B, hidden, 1)
        )
        
        # Noisy input encoder
        self.input_encoder = nn.Sequential(
            nn.Conv1d(1, hidden // 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        
        # Main denoising network: simple residual blocks
        self.denoise_net = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        
        # Conditioning projection (time + history -> scale/shift)
        self.cond_proj = nn.Linear(hidden * 2, hidden * 2)
        
        # Output
        self.output = nn.Conv1d(hidden, 1, kernel_size=3, padding=1)
        
    def forward(self, x, t, history, return_variance=False):
        """
        Args:
            x: (B, 1, L) noisy future
            t: (B,) timestep
            history: (B, C, L) conditioning history
        Returns:
            Predicted noise (B, 1, L)
        """
        B = x.size(0)
        
        # 1. Time embedding
        t_emb = self.time_embed(t)  # (B, hidden)
        
        # 2. History embedding
        h_emb = self.history_encoder(history).squeeze(-1)  # (B, hidden)
        
        # 3. Combined conditioning
        cond = torch.cat([t_emb, h_emb], dim=-1)  # (B, hidden*2)
        cond = self.cond_proj(cond)  # (B, hidden*2)
        scale, shift = cond.chunk(2, dim=-1)  # Each (B, hidden)
        
        # 4. Encode noisy input
        h = self.input_encoder(x)  # (B, hidden, L)
        
        # 5. Apply conditioning (FiLM)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        
        # 6. Denoise
        h = self.denoise_net(h)
        
        # 7. Output
        noise_pred = self.output(h)
        
        if return_variance:
            return noise_pred, torch.zeros_like(noise_pred)
        return noise_pred


# Alias for compatibility
MarketDiffusion = SimpleMarketDiffusion

# ==========================================
# 4. DIFFUSION UTILITIES
# ==========================================
def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
    Better for time series than linear schedule - more gradual noise at start/end.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1.0):
    """
    Sigmoid schedule - smoother transitions that can be better for financial data.
    
    Reference: "Improved Denoising Diffusion Probabilistic Models" appendix
    
    Args:
        timesteps: Number of diffusion steps
        start: Start of sigmoid input range (more negative = slower start)
        end: End of sigmoid input range
        tau: Temperature (higher = more uniform)
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    
    # Sigmoid transformation
    v_start = torch.sigmoid(torch.tensor(start / tau))
    v_end = torch.sigmoid(torch.tensor(end / tau))
    
    alphas_cumprod = torch.sigmoid((t / timesteps * (end - start) + start) / tau)
    alphas_cumprod = (v_end - alphas_cumprod) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DiffusionManager:
    def __init__(self, config):
        self.timesteps = config.timesteps
        self.device = config.device
        self.ddim_steps = getattr(config, 'ddim_steps', 100)  # DDIM sampling steps (increased)
        
        # Use configurable schedule (sigmoid often better for financial data)
        noise_schedule = getattr(config, 'noise_schedule', 'sigmoid')
        if noise_schedule == 'sigmoid':
            self.betas = sigmoid_beta_schedule(self.timesteps).to(self.device)
        else:
            self.betas = cosine_beta_schedule(self.timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]])
        
        # For DDPM sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def add_noise(self, x_start, t):
        """Forward process: q(x_t | x_0)"""
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        
        noise = torch.randn_like(x_start)
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise, noise

    def _predict_x0_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and predicted noise."""
        return (x_t - self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise) / self.sqrt_alphas_cumprod[t][:, None, None]

    @torch.no_grad()
    def sample_ddpm(self, model, history, temperature=1.0):
        """Standard DDPM sampling (slower but potentially higher quality)."""
        model.eval()
        batch_size = history.shape[0]
        seq_len = history.shape[2]
        
        # Initialize with temperature-scaled noise
        x = torch.randn((batch_size, 1, seq_len)).to(self.device) * temperature
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            predicted_noise = model(x, t, history)
            
            # Relaxed clamping
            predicted_noise = torch.clamp(predicted_noise, -10.0, 10.0)
            
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x) * temperature
            else:
                noise = torch.zeros_like(x)
            
            # DDPM update
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise) + torch.sqrt(beta) * noise
            
            # Relaxed clamping
            x = torch.clamp(x, -20.0, 20.0)
            
        return x
    
    @torch.no_grad()
    def sample_ddim(self, model, history, eta=1.0, num_steps=None, temperature=1.0, 
                    use_mc_dropout=False, mc_samples=1):
        """
        DDIM sampling - fast and effective.
        
        Args:
            model: Trained diffusion model
            history: (B, C, L) conditioning history
            eta: Stochasticity level (1.0 = DDPM-like, 0.0 = deterministic)
            num_steps: Number of DDIM steps
            temperature: Initial noise temperature
            use_mc_dropout: Ignored (kept for API compatibility)
            mc_samples: Ignored (kept for API compatibility)
        """
        model.eval()
        batch_size = history.shape[0]
        seq_len = history.shape[2]
        num_steps = num_steps or self.ddim_steps
        
        # Create evenly spaced timesteps for DDIM
        step_ratio = self.timesteps // num_steps
        timesteps = torch.arange(0, self.timesteps, step_ratio).long().flip(0).to(self.device)
        
        # Initialize with temperature-scaled noise for diversity
        x = torch.randn((batch_size, 1, seq_len)).to(self.device) * temperature
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Simple forward pass
            predicted_noise = model(x, t_batch, history, return_variance=False)
            
            # Clamp to prevent numerical issues
            predicted_noise = torch.clamp(predicted_noise, -10.0, 10.0)
            
            # Predict x_0
            alpha_cumprod_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Clamp x0
            x0_pred = torch.clamp(x0_pred, -15.0, 15.0)
            
            if i < len(timesteps) - 1:
                # Get previous timestep
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.alphas_cumprod[t_prev]
                
                # Standard DDIM sigma
                sigma = eta * torch.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * 
                    (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                )
                
                # Direction pointing to x_t
                pred_dir = torch.sqrt(torch.clamp(1 - alpha_cumprod_t_prev - sigma ** 2, min=0)) * predicted_noise
                
                # Random noise
                noise = torch.randn_like(x) * temperature if eta > 0 else torch.zeros_like(x)
                
                x = torch.sqrt(alpha_cumprod_t_prev) * x0_pred + pred_dir + sigma * noise
                
                # Clamp
                x = torch.clamp(x, -20.0, 20.0)
            else:
                x = x0_pred
        
        return x
    
    @torch.no_grad()
    def sample(self, model, history, method='ddim', **kwargs):
        """
        Main sampling interface - defaults to DDIM for speed.
        
        Args:
            method: 'ddim' (fast, default) or 'ddpm' (slow, potentially higher quality)
        """
        if method == 'ddim':
            return self.sample_ddim(model, history, **kwargs)
        else:
            return self.sample_ddpm(model, history, **kwargs)


# ==========================================
# 5. BASELINE MODELS
# ==========================================
# These baselines are CRITICAL for academic rigor.
# Your diffusion model must beat these to be considered useful.
# 
# | Baseline           | Description                                      |
# |--------------------|--------------------------------------------------|
# | Historical Bootstrap | Random 64-day windows from training data       |
# | Random Walk        | Returns ~ N(μ, σ) from historical stats          |
# | GARCH(1,1)         | Classic volatility clustering model              |
# | AR(1)              | Simple autoregressive model                      |
# ==========================================

class BaselineModel:
    """Base class for all baseline models."""
    def __init__(self, name):
        self.name = name
        self.fitted = False
    
    def fit(self, train_returns):
        """Fit model on training data."""
        raise NotImplementedError
    
    def generate(self, history, num_paths, pred_length):
        """Generate future paths given history."""
        raise NotImplementedError


class HistoricalBootstrap(BaselineModel):
    """
    Historical Bootstrap Baseline
    
    Simply samples random 64-day windows from the training data.
    This is a surprisingly strong baseline - if your model can't beat this,
    it's not learning anything beyond what's in the historical distribution.
    """
    def __init__(self):
        super().__init__("Historical Bootstrap")
        self.train_returns = None
        self.pred_length = None
    
    def fit(self, train_returns, pred_length=64):
        self.train_returns = train_returns
        self.pred_length = pred_length
        self.fitted = True
        return self
    
    def generate(self, history, num_paths, pred_length=None):
        """Sample random windows from training data."""
        pred_length = pred_length or self.pred_length
        paths = []
        
        for _ in range(num_paths):
            # Random start index
            max_start = len(self.train_returns) - pred_length
            if max_start <= 0:
                # Not enough data, just sample with replacement
                path = np.random.choice(self.train_returns, size=pred_length, replace=True)
            else:
                start_idx = np.random.randint(0, max_start)
                path = self.train_returns[start_idx : start_idx + pred_length]
            paths.append(path)
        
        return np.array(paths)  # (num_paths, pred_length)


class RandomWalkModel(BaselineModel):
    """
    Random Walk (Geometric Brownian Motion) Baseline
    
    Generates returns from N(μ, σ) where μ and σ are estimated from training data.
    This is the simplest possible model - assumes returns are i.i.d. normal.
    """
    def __init__(self):
        super().__init__("Random Walk")
        self.mu = None
        self.sigma = None
    
    def fit(self, train_returns, **kwargs):
        self.mu = np.mean(train_returns)
        self.sigma = np.std(train_returns)
        self.fitted = True
        return self
    
    def generate(self, history, num_paths, pred_length=64):
        """Generate random walk paths."""
        paths = np.random.normal(self.mu, self.sigma, size=(num_paths, pred_length))
        return paths


class GARCHModel(BaselineModel):
    """
    GARCH(1,1) Baseline
    
    Classic volatility model that captures volatility clustering.
    This is a strong baseline because financial returns exhibit heteroskedasticity.
    
    Model: r_t = μ + ε_t, where ε_t = σ_t * z_t, z_t ~ N(0,1)
           σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
    """
    def __init__(self):
        super().__init__("GARCH(1,1)")
        self.model = None
        self.fitted_model = None
        self.mu = None
    
    def fit(self, train_returns, **kwargs):
        # Scale returns to percentage for GARCH stability
        returns_pct = train_returns * 100
        
        try:
            self.model = arch_model(returns_pct, vol='Garch', p=1, q=1, mean='Constant')
            self.fitted_model = self.model.fit(disp='off', show_warning=False)
            self.mu = np.mean(train_returns)
            self.fitted = True
        except Exception as e:
            print(f"GARCH fitting failed: {e}. Using fallback.")
            # Fallback to simple random walk
            self.mu = np.mean(train_returns)
            self.sigma = np.std(train_returns)
            self.fitted = False
        
        return self
    
    def generate(self, history, num_paths, pred_length=64):
        """Generate paths using GARCH volatility forecasts."""
        if not self.fitted or self.fitted_model is None:
            # Fallback to random walk
            return np.random.normal(self.mu, self.sigma, size=(num_paths, pred_length))
        
        paths = []
        
        # Get GARCH parameters
        params = self.fitted_model.params
        omega = params.get('omega', 0.01)
        alpha = params.get('alpha[1]', 0.1)
        beta = params.get('beta[1]', 0.85)
        mu = params.get('mu', 0)
        
        for _ in range(num_paths):
            path = []
            # Initialize with last observed variance
            sigma2 = self.fitted_model.conditional_volatility[-1]**2 if hasattr(self.fitted_model, 'conditional_volatility') else 1.0
            eps_prev = 0
            
            for t in range(pred_length):
                # Update variance
                sigma2 = omega + alpha * eps_prev**2 + beta * sigma2
                sigma = np.sqrt(sigma2)
                
                # Generate return
                z = np.random.normal(0, 1)
                eps = sigma * z
                r = mu / 100 + eps / 100  # Convert back from percentage
                
                path.append(r)
                eps_prev = eps
            
            paths.append(path)
        
        return np.array(paths)


class AR1Model(BaselineModel):
    """
    AR(1) Autoregressive Baseline
    
    Simple model: r_t = φ * r_{t-1} + ε_t
    
    Captures mean reversion in returns (if any).
    """
    def __init__(self):
        super().__init__("AR(1)")
        self.phi = None
        self.intercept = None
        self.sigma = None
        self.fitted_model = None
    
    def fit(self, train_returns, **kwargs):
        try:
            # Fit AR(1) model
            model = AutoReg(train_returns, lags=1)
            self.fitted_model = model.fit()
            self.intercept = self.fitted_model.params[0]
            self.phi = self.fitted_model.params[1]
            self.sigma = np.std(self.fitted_model.resid)
            self.fitted = True
        except Exception as e:
            print(f"AR(1) fitting failed: {e}. Using fallback.")
            self.phi = 0.0
            self.intercept = np.mean(train_returns)
            self.sigma = np.std(train_returns)
            self.fitted = True  # Fallback is still usable
        
        return self
    
    def generate(self, history, num_paths, pred_length=64):
        """Generate paths using AR(1) dynamics."""
        paths = []
        
        # Use last value from history as starting point
        if len(history) > 0:
            r_prev = history[-1]
        else:
            r_prev = self.intercept
        
        for _ in range(num_paths):
            path = []
            r = r_prev
            
            for t in range(pred_length):
                # AR(1): r_t = intercept + phi * r_{t-1} + epsilon
                eps = np.random.normal(0, self.sigma)
                r = self.intercept + self.phi * r + eps
                path.append(r)
            
            paths.append(path)
        
        return np.array(paths)


def get_all_baselines():
    """Factory function to create all baseline models."""
    baselines = [
        HistoricalBootstrap(),
        RandomWalkModel(),
    ]
    
    if HAS_ARCH:
        baselines.append(GARCHModel())
    
    if HAS_STATSMODELS:
        baselines.append(AR1Model())
        
    return baselines


# ==========================================
# 6. EVALUATION METRICS
# ==========================================

def compute_crps(samples, observation):
    """
    Compute the Continuous Ranked Probability Score (CRPS).
    
    CRPS is a proper scoring rule that measures how well a probabilistic forecast
    matches the actual observation. Lower is better.
    
    For an empirical distribution with samples x_1, ..., x_n and observation y:
    CRPS = (1/n) * sum_i |x_i - y| - (1/2n^2) * sum_i sum_j |x_i - x_j|
    
    This is the energy form of CRPS which is computationally efficient.
    
    Args:
        samples: (n,) array of forecast samples
        observation: scalar, the actual observed value
    
    Returns:
        CRPS score (lower is better)
    """
    samples = np.asarray(samples).flatten()
    n = len(samples)
    
    if n == 0:
        return np.nan
    
    # Term 1: Mean absolute error between samples and observation
    term1 = np.mean(np.abs(samples - observation))
    
    # Term 2: Mean absolute difference between all pairs of samples
    # Efficient computation: E[|X - X'|] = 2 * E[X * F(X)] - 2 * E[X] * E[F(X)]
    # For empirical distribution, this simplifies to:
    sorted_samples = np.sort(samples)
    # Using the formula: sum_i sum_j |x_i - x_j| / n^2 = 2 * (sum_i (2i - n - 1) * x_i) / n^2
    indices = np.arange(1, n + 1)
    term2 = np.sum((2 * indices - n - 1) * sorted_samples) / (n * n)
    
    crps = term1 - term2
    return crps


def compute_pit(samples, observation):
    """
    Compute the Probability Integral Transform (PIT) value.
    
    PIT = F(y) where F is the empirical CDF of the forecast and y is the observation.
    If forecasts are well-calibrated, PIT values should be uniformly distributed.
    
    Args:
        samples: (n,) array of forecast samples
        observation: scalar, the actual observed value
    
    Returns:
        PIT value in [0, 1]
    """
    samples = np.asarray(samples).flatten()
    return np.mean(samples <= observation)


def evaluate_paths(generated_paths, actual_returns, model_name="Model"):
    """
    Compute comprehensive evaluation metrics for generated paths.
    
    Args:
        generated_paths: (num_paths, pred_length) array of generated returns
        actual_returns: (pred_length,) array of actual future returns
        model_name: Name for reporting
    
    Returns:
        Dictionary of metrics
    """
    num_paths, pred_length = generated_paths.shape
    
    # 1. Cumulative returns for coverage analysis
    actual_cum = np.cumsum(actual_returns)
    gen_cum = np.cumsum(generated_paths, axis=1)
    
    # Final cumulative return
    actual_final = actual_cum[-1]
    gen_finals = gen_cum[:, -1]
    
    # 2. Coverage metrics (is actual within prediction intervals?)
    p5 = np.percentile(gen_finals, 5)
    p95 = np.percentile(gen_finals, 95)
    p25 = np.percentile(gen_finals, 25)
    p75 = np.percentile(gen_finals, 75)
    
    in_90_ci = 1 if p5 <= actual_final <= p95 else 0
    in_50_ci = 1 if p25 <= actual_final <= p75 else 0
    
    # 3. Point prediction error
    mean_pred = np.mean(gen_finals)
    mae = np.abs(mean_pred - actual_final)
    
    # 4. Distributional metrics
    # Volatility comparison
    actual_vol = np.std(actual_returns)
    gen_vol = np.mean([np.std(generated_paths[i]) for i in range(num_paths)])
    vol_ratio = gen_vol / actual_vol if actual_vol > 0 else 1.0
    
    # 5. KS test on return distributions
    ks_stat, ks_pvalue = stats.ks_2samp(actual_returns, generated_paths.flatten())
    
    # 6. CRPS (Continuous Ranked Probability Score) - PROPER implementation
    # Using the energy score formulation for efficiency and correctness
    crps = compute_crps(gen_finals, actual_final)
    
    # 7. PIT for calibration assessment
    pit = compute_pit(gen_finals, actual_final)
    
    return {
        'model': model_name,
        'in_90_ci': in_90_ci,
        'in_50_ci': in_50_ci,
        'mae': mae,
        'mean_pred': mean_pred,
        'actual': actual_final,
        'vol_ratio': vol_ratio,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'crps': crps,
        'pit': pit,
    }


# ==========================================
# 7. MAIN EXECUTION
# ==========================================

def train_diffusion_model(train_dataset, conf, val_dataset=None, verbose=True):
    """
    Train the diffusion model on the given dataset.
    Returns the trained model and diffusion manager.
    
    Improvements:
    - Cosine annealing with warmup
    - Gradient clipping for stability
    - EMA for better sampling
    - Heteroscedastic loss (learns aleatoric uncertainty)
    - Multi-feature conditioning support
    - Validation-based early stopping
    """
    # Optimize DataLoader for GPU
    dataloader = DataLoader(
        train_dataset, 
        batch_size=conf.batch_size, 
        shuffle=True,
        pin_memory=conf.pin_memory if conf.device == "cuda" else False,
        num_workers=conf.num_workers if conf.device == "cuda" else 0,
        drop_last=True  # Avoid small final batches
    )
    
    # Get number of input features from dataset
    num_features = getattr(train_dataset, 'num_features', 1)
    if verbose:
        print(f"    Model input features: {num_features}")
    
    model = MarketDiffusion(conf, num_features=num_features).to(conf.device)
    
    # Compile model for PyTorch 2.0+ (can speed up 20-30%)
    if conf.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            if verbose:
                print("    ✓ Model compiled with torch.compile")
        except Exception as e:
            if verbose:
                print(f"    ⚠ torch.compile failed: {e}")
    
    diffuser = DiffusionManager(conf)
    
    # Optimizer with weight decay (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=0.01)
    
    # Initialize EMA
    ema = EMA(model, decay=0.995)
    
    # Cosine annealing with warmup
    warmup_epochs = max(5, conf.epochs // 10)
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (conf.epochs - warmup_epochs)
            # Minimum LR 10% of initial
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision scaler for GPU acceleration
    scaler = torch.cuda.amp.GradScaler() if conf.use_amp and conf.device == "cuda" else None
    
    # Use simple MSE loss - more stable than heteroscedastic for this task
    # The heteroscedastic loss was collapsing (variance going to infinity)
    loss_fn = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(conf.epochs):
        model.train()
        epoch_loss = 0
        
        for history, future in dataloader:
            history = history.to(conf.device, non_blocking=True)
            future = future.to(conf.device, non_blocking=True)
            
            # Uniform timestep sampling - standard for diffusion models
            t = torch.randint(0, conf.timesteps, (history.size(0),), device=conf.device).long()
            
            # Mixed precision training (faster on GPU)
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    noisy_future, noise = diffuser.add_noise(future, t)
                    # Use simple forward pass (ignore variance head during training)
                    noise_pred = model(noisy_future, t, history, return_variance=False)
                    loss = loss_fn(noise_pred, noise)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping for training stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                noisy_future, noise = diffuser.add_noise(future, t)
                noise_pred = model(noisy_future, t, history, return_variance=False)
                loss = loss_fn(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            # Update EMA
            ema.update(model)
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step()
        
        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch+1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch+1}/{conf.epochs} | Loss: {avg_loss:.6f} | LR: {current_lr:.2e}")
    
    # Return model with EMA weights
    ema.apply_shadow(model)
    return model, diffuser


def calibrate_variance_on_validation(model, diffuser, val_dataset, scaler, conf, target_coverage=0.90):
    """
    Calibrate the variance scaling factor on the validation set.
    
    This is critical for proper coverage: we find a multiplier that achieves
    the desired coverage (e.g., 90%) on held-out validation data.
    
    Uses binary search to find optimal scaling factor.
    
    Args:
        model: Trained diffusion model
        diffuser: DiffusionManager
        val_dataset: Validation dataset
        scaler: Data scaler (for inverse transform)
        conf: Config
        target_coverage: Target coverage level (default 0.90)
    
    Returns:
        Optimal variance scale factor
    """
    model.eval()
    
    # Sample validation windows
    num_windows = min(20, len(val_dataset))  # Use fewer windows for efficiency
    if num_windows < 5:
        return 1.0  # Not enough data for calibration
    
    np.random.seed(123)
    val_indices = np.random.choice(len(val_dataset), size=num_windows, replace=False)
    
    # Get scaler parameters
    if hasattr(scaler, 'center_'):
        scaler_mean = scaler.center_[0]
        scaler_scale = scaler.scale_[0]
    elif hasattr(scaler, 'mean_'):
        scaler_mean = scaler.mean_[0]
        scaler_scale = scaler.scale_[0]
    else:
        scaler_mean, scaler_scale = 0, 1
    
    # Collect generated paths and actuals
    all_gen_finals = []  # List of (num_paths,) arrays
    all_actuals = []     # List of scalars
    
    with torch.no_grad():
        for idx in val_indices:
            history_scaled, future_scaled = val_dataset[idx]
            history_scaled = history_scaled.unsqueeze(0).to(conf.device)
            
            # Get raw returns for this window
            raw_window = val_dataset.get_raw_window(idx)
            future_raw = raw_window[conf.cond_length:]
            actual_final = np.cumsum(future_raw)[-1]
            
            # Generate paths
            gen_paths_scaled = diffuser.sample_ddim(
                model,
                history_scaled.repeat(conf.num_paths, 1, 1),
                eta=getattr(conf, 'ddim_eta', 1.0),
                temperature=getattr(conf, 'sampling_temperature', 1.0),
                use_mc_dropout=True,
                mc_samples=3
            )
            gen_paths_scaled = gen_paths_scaled.squeeze(1).cpu().numpy()
            
            # Inverse transform
            gen_paths = gen_paths_scaled * scaler_scale + scaler_mean
            
            # Compute cumulative final returns
            gen_finals = np.cumsum(gen_paths, axis=1)[:, -1]
            
            all_gen_finals.append(gen_finals)
            all_actuals.append(actual_final)
    
    all_gen_finals = np.array(all_gen_finals)  # (num_windows, num_paths)
    all_actuals = np.array(all_actuals)        # (num_windows,)
    
    # Binary search for optimal scale factor
    def compute_coverage(scale_factor):
        coverages = []
        for i in range(len(all_actuals)):
            gen_finals = all_gen_finals[i]
            actual = all_actuals[i]
            
            # Scale around mean
            mean = gen_finals.mean()
            scaled_finals = mean + (gen_finals - mean) * scale_factor
            
            # Check 90% CI coverage
            p5 = np.percentile(scaled_finals, 5)
            p95 = np.percentile(scaled_finals, 95)
            coverages.append(1 if p5 <= actual <= p95 else 0)
        
        return np.mean(coverages)
    
    # Binary search
    low, high = 0.5, 4.0
    best_scale = 1.0
    best_diff = float('inf')
    
    for _ in range(15):  # 15 iterations gives ~0.01 precision
        mid = (low + high) / 2
        coverage = compute_coverage(mid)
        
        diff = abs(coverage - target_coverage)
        if diff < best_diff:
            best_diff = diff
            best_scale = mid
        
        if coverage < target_coverage:
            low = mid  # Need more variance
        else:
            high = mid  # Need less variance
    
    print(f"    Calibrated variance scale: {best_scale:.3f} (achieves {compute_coverage(best_scale)*100:.1f}% coverage on val)")
    
    return best_scale


def evaluate_on_test_windows(model, diffuser, test_dataset, baselines, train_returns, scaler, conf, calibrated_scale=None):
    """
    Evaluate model (diffusion or probabilistic) and baselines on test windows.
    
    Args:
        calibrated_scale: If provided, use this as the variance scale factor
                         (from validation set calibration)
    
    Returns a dictionary with results for each model.
    """
    model.eval()
    model_type = getattr(conf, 'model_type', 'diffusion')
    
    # Sample test windows
    num_windows = min(conf.num_test_windows, len(test_dataset))
    if num_windows == 0:
        return {}
    
    np.random.seed(42)
    test_indices = np.random.choice(len(test_dataset), size=num_windows, replace=False)
    
    # Results storage: model_name -> list of metric dicts
    all_results = {m.name: [] for m in baselines}
    all_results['Diffusion'] = []
    
    # Get sampling parameters
    sampling_method = getattr(conf, 'sampling_method', 'ddim')
    ddim_eta = getattr(conf, 'ddim_eta', 1.0)  # Default to fully stochastic
    
    for idx in test_indices:
        # Get test window (scaled for diffusion, raw for baselines)
        history_scaled, future_scaled = test_dataset[idx]
        history_scaled = history_scaled.unsqueeze(0).to(conf.device)
        
        # Get raw returns for this window
        raw_window = test_dataset.get_raw_window(idx)
        history_raw = raw_window[:conf.cond_length]
        future_raw = raw_window[conf.cond_length:]
        
        # --- Evaluate Model ---
        # Get sampling parameters
        temperature = getattr(conf, 'sampling_temperature', 1.0)
        diversity_noise = getattr(conf, 'diversity_noise_std', 0.0)
        
        with torch.no_grad():
            if model_type == 'probabilistic':
                # Probabilistic model: sample from predicted distribution
                gen_paths_scaled = model.sample(history_scaled, num_samples=conf.num_paths)
                gen_paths_scaled = gen_paths_scaled.squeeze(0)  # (num_paths, pred_length)
            else:
                # Diffusion model: iterative denoising
                if sampling_method == 'ddim':
                    gen_paths_scaled = diffuser.sample_ddim(
                        model, 
                        history_scaled.repeat(conf.num_paths, 1, 1),
                        eta=ddim_eta,
                        temperature=temperature
                    )
                else:
                    gen_paths_scaled = diffuser.sample_ddpm(
                        model,
                        history_scaled.repeat(conf.num_paths, 1, 1),
                        temperature=temperature
                    )
                gen_paths_scaled = gen_paths_scaled.squeeze(1)  # (num_paths, pred_length)
                
                # Add diversity noise to prevent mode collapse
                if diversity_noise > 0:
                    noise = torch.randn_like(gen_paths_scaled) * diversity_noise
                    gen_paths_scaled = gen_paths_scaled + noise
        
        # Unscale generated paths
        gen_paths = gen_paths_scaled.cpu().numpy()
        
        # Get scaler parameters (handle both StandardScaler and RobustScaler)
        if hasattr(scaler, 'center_'):  # RobustScaler
            scaler_mean = scaler.center_[0]
            scaler_scale = scaler.scale_[0]
        elif hasattr(scaler, 'mean_'):  # StandardScaler
            scaler_mean = scaler.mean_[0]
            scaler_scale = scaler.scale_[0]
        else:
            scaler_mean = 0
            scaler_scale = 1
        
        # Inverse transform to get raw returns
        gen_paths = gen_paths * scaler_scale + scaler_mean
        
        # ============================================================
        # NO POST-PROCESSING - trust the model
        # ============================================================
        
        gen_vol = np.std(gen_paths)
        recent_vol = np.std(history_raw[-20:]) if len(history_raw) >= 20 else np.std(history_raw)
            
        # Debug first window
        if len(all_results['Diffusion']) == 0:
            print(f"    DEBUG: Recent vol: {recent_vol:.4f}, Gen vol: {gen_vol:.4f}")
        
        metrics = evaluate_paths(gen_paths, future_raw, "Diffusion")
        all_results['Diffusion'].append(metrics)
        
        # --- Evaluate Baselines ---
        for baseline in baselines:
            baseline_paths = baseline.generate(history_raw, conf.num_paths, conf.pred_length)
            metrics = evaluate_paths(baseline_paths, future_raw, baseline.name)
            all_results[baseline.name].append(metrics)
    
    return all_results


def aggregate_results(results_dict):
    """Aggregate results across all test windows into summary statistics."""
    summary = {}
    
    for model_name, metrics_list in results_dict.items():
        if not metrics_list:
            continue
            
        summary[model_name] = {
            'coverage_90': np.mean([m['in_90_ci'] for m in metrics_list]) * 100,
            'coverage_50': np.mean([m['in_50_ci'] for m in metrics_list]) * 100,
            'mae': np.mean([m['mae'] for m in metrics_list]),
            'vol_ratio': np.mean([m['vol_ratio'] for m in metrics_list]),
            'crps': np.mean([m['crps'] for m in metrics_list]),
            'n_windows': len(metrics_list),
        }
    
    return summary


def print_comparison_table(aggregated_results, fold_name=""):
    """Print a formatted comparison table of all models."""
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON {fold_name}")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'90% Cov':>10} {'50% Cov':>10} {'MAE':>10} {'Vol Ratio':>12} {'CRPS':>10}")
    print(f"{'-'*80}")
    
    # Sort by CRPS (lower is better)
    sorted_models = sorted(aggregated_results.items(), key=lambda x: x[1]['crps'])
    
    for model_name, metrics in sorted_models:
        cov90 = f"{metrics['coverage_90']:.1f}%"
        cov50 = f"{metrics['coverage_50']:.1f}%"
        mae = f"{metrics['mae']:.4f}"
        vol = f"{metrics['vol_ratio']:.2f}x"
        crps = f"{metrics['crps']:.4f}"
        
        # Highlight diffusion model
        prefix = "→ " if model_name == "Diffusion" else "  "
        print(f"{prefix}{model_name:<18} {cov90:>10} {cov50:>10} {mae:>10} {vol:>12} {crps:>10}")
    
    print(f"{'='*80}")
    
    # Determine winner
    best_model = sorted_models[0][0]
    if best_model == "Diffusion":
        print("✅ Diffusion model has the BEST CRPS score!")
    else:
        diffusion_crps = aggregated_results.get('Diffusion', {}).get('crps', float('inf'))
        best_crps = sorted_models[0][1]['crps']
        diff_pct = ((diffusion_crps - best_crps) / best_crps) * 100 if best_crps > 0 else 0
        print(f"⚠️  {best_model} beats Diffusion by {diff_pct:.1f}% on CRPS")


# ==========================================
# 8. REPORT GENERATION
# ==========================================

def generate_research_report(all_fold_results, final_summary, conf, simulation_data=None, output_path="research_report.html"):
    """
    Generate a professional, comprehensive HTML research report.
    
    Features:
    - Executive summary with key metrics
    - Interactive charts (CRPS comparison, fold performance, radar chart)
    - Detailed fold-by-fold analysis
    - Model comparison across multiple metrics
    - Professional styling matching the forward prediction charts
    """
    
    # Prepare data
    model_names = list(final_summary.keys())
    sorted_by_crps = sorted(final_summary.items(), key=lambda x: x[1]['crps'])
    best_model = sorted_by_crps[0][0]
    
    # Diffusion metrics
    diff_metrics = final_summary.get('Diffusion', {})
    diff_crps = diff_metrics.get('crps', 0)
    diff_crps_std = diff_metrics.get('crps_std', 0)
    diff_coverage = diff_metrics.get('coverage_90', 0)
    diff_mae = diff_metrics.get('mae', 0)
    diff_vol = diff_metrics.get('vol_ratio', 0)
    diff_rank = next((i+1 for i, (name, _) in enumerate(sorted_by_crps) if name == 'Diffusion'), len(sorted_by_crps))
    
    # Win/loss analysis
    wins = sum(1 for f in all_fold_results 
               if f['results'].get('Diffusion', {}).get('crps', float('inf')) == 
               min(v['crps'] for v in f['results'].values()))
    total_folds = len(all_fold_results)
    win_rate = wins / total_folds * 100 if total_folds > 0 else 0
    
    # Best baseline
    best_baseline = sorted_by_crps[1] if len(sorted_by_crps) > 1 and sorted_by_crps[0][0] == 'Diffusion' else sorted_by_crps[0]
    best_baseline_name = best_baseline[0] if best_baseline[0] != 'Diffusion' else (sorted_by_crps[1][0] if len(sorted_by_crps) > 1 else 'N/A')
    
    # Improvement over best baseline
    if best_model == 'Diffusion' and len(sorted_by_crps) > 1:
        baseline_crps = sorted_by_crps[1][1]['crps']
        improvement = (baseline_crps - diff_crps) / baseline_crps * 100
        improvement_text = f"+{improvement:.1f}% better"
        status_class = "success"
        status_icon = "✓"
        status_text = "OUTPERFORMS BASELINES"
    else:
        baseline_crps = sorted_by_crps[0][1]['crps']
        improvement = (diff_crps - baseline_crps) / diff_crps * 100 if diff_crps > 0 else 0
        improvement_text = f"-{improvement:.1f}% vs best"
        status_class = "warning"
        status_icon = "⚠"
        status_text = "UNDERPERFORMS"
    
    generation_date = datetime.now().strftime("%B %d, %Y")
    generation_time = datetime.now().strftime("%H:%M:%S")
    
    # Prepare chart data as JSON
    comparison_data = json.dumps({
        'models': [m[0] for m in sorted_by_crps],
        'crps': [m[1]['crps'] for m in sorted_by_crps],
        'coverage': [m[1]['coverage_90'] for m in sorted_by_crps],
        'mae': [m[1]['mae'] for m in sorted_by_crps],
        'vol_ratio': [m[1]['vol_ratio'] for m in sorted_by_crps],
    })
    
    # Prepare forward simulation data if available
    forward_prediction_html = ""
    forward_prediction_js = ""
    if simulation_data is not None:
        future_dates = simulation_data['future_dates']
        mean_path = simulation_data['mean_path']
        median_path = simulation_data['median_path']
        p5 = simulation_data['p5']
        p95 = simulation_data['p95']
        last_price = simulation_data['last_price']
        last_date = simulation_data['last_date']
        prob_positive = simulation_data['prob_positive']
        var_95 = simulation_data['var_95']
        
        # Convert dates to strings for JSON
        date_strings = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        forward_data = json.dumps({
            'dates': date_strings,
            'mean': mean_path.tolist(),
            'median': median_path.tolist(),
            'p5': p5.tolist(),
            'p95': p95.tolist(),
            'last_price': float(last_price),
            'last_date': last_date.strftime('%Y-%m-%d'),
        })
        
        final_return = (median_path[-1] / last_price - 1) * 100
        sentiment = "BULLISH 📈" if prob_positive > 60 else "BEARISH 📉" if prob_positive < 40 else "NEUTRAL ➡️"
        sentiment_color = "var(--accent-green)" if prob_positive > 60 else "var(--accent-red)" if prob_positive < 40 else "var(--accent-orange)"
        
        forward_prediction_html = f'''
        <!-- Forward Prediction Section -->
        <div class="section">
            <h2 class="section-title"><span class="icon">🔮</span> Forward Market Prediction</h2>
            <div class="metric-cards">
                <div class="metric-card">
                    <div class="metric-value" style="color: {sentiment_color};">${median_path[-1]:.2f}</div>
                    <div class="metric-label">64-Day Target (Median)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{final_return:+.1f}%</div>
                    <div class="metric-label">Expected Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{prob_positive:.0f}%</div>
                    <div class="metric-label">P(Positive Return)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{var_95:+.1f}%</div>
                    <div class="metric-label">95% VaR</div>
                </div>
            </div>
            <div class="chart-container" style="margin-top: 1.5rem;">
                <div class="chart-title">📊 {conf.ticker} Price Forecast — Next 64 Trading Days</div>
                <div id="forwardChart" style="height: 400px;"></div>
            </div>
            <div style="text-align: center; margin-top: 1rem; padding: 1rem; background: var(--bg-card); border-radius: 8px;">
                <span style="font-size: 1.5rem; font-weight: 600; color: {sentiment_color};">
                    Model Sentiment: {sentiment}
                </span>
            </div>
        </div>
        '''
        
        forward_prediction_js = f'''
        // Forward Prediction Chart
        const forwardData = {forward_data};
        
        const forwardTrace1 = {{
            x: forwardData.dates,
            y: forwardData.p95,
            type: 'scatter',
            mode: 'lines',
            name: '95th Percentile',
            line: {{ color: 'rgba(0, 186, 124, 0.3)', width: 0 }},
            showlegend: false
        }};
        
        const forwardTrace2 = {{
            x: forwardData.dates,
            y: forwardData.p5,
            type: 'scatter',
            mode: 'lines',
            name: '90% Confidence',
            fill: 'tonexty',
            fillcolor: 'rgba(0, 186, 124, 0.2)',
            line: {{ color: 'rgba(0, 186, 124, 0.3)', width: 0 }},
        }};
        
        const forwardTrace3 = {{
            x: forwardData.dates,
            y: forwardData.mean,
            type: 'scatter',
            mode: 'lines',
            name: 'Mean Forecast',
            line: {{ color: '#1d9bf0', width: 2 }}
        }};
        
        const forwardTrace4 = {{
            x: forwardData.dates,
            y: forwardData.median,
            type: 'scatter',
            mode: 'lines',
            name: 'Median Forecast',
            line: {{ color: '#00ba7c', width: 3 }}
        }};
        
        const startLine = {{
            x: [forwardData.dates[0], forwardData.dates[0]],
            y: [Math.min(...forwardData.p5) * 0.98, Math.max(...forwardData.p95) * 1.02],
            type: 'scatter',
            mode: 'lines',
            name: 'Current Price: $' + forwardData.last_price.toFixed(2),
            line: {{ color: '#6e7681', width: 2, dash: 'dash' }}
        }};
        
        Plotly.newPlot('forwardChart', [forwardTrace1, forwardTrace2, forwardTrace3, forwardTrace4, startLine], {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {{ color: '#e7e9ea' }},
            margin: {{ t: 30, r: 30, b: 50, l: 60 }},
            xaxis: {{
                gridcolor: '#2f3542',
                tickangle: -30,
                title: 'Date'
            }},
            yaxis: {{
                gridcolor: '#2f3542',
                title: 'Price ($)',
                tickprefix: '$'
            }},
            legend: {{
                orientation: 'h',
                y: -0.15
            }},
            hovermode: 'x unified',
            hoverlabel: {{
                bgcolor: '#1a1f2e',
                bordercolor: '#2f3542',
                font: {{ color: '#e7e9ea', size: 13 }}
            }}
        }}, {{responsive: true}});
        '''
    
    fold_data = json.dumps([{
        'fold': f['fold'],
        'year': f['test_period'],
        'diffusion': f['results'].get('Diffusion', {}).get('crps', 0),
        'best_baseline': min([v['crps'] for k, v in f['results'].items() if k != 'Diffusion'] or [0]),
        'winner': 'Diffusion' if f['results'].get('Diffusion', {}).get('crps', float('inf')) <= 
                  min([v['crps'] for k, v in f['results'].items() if k != 'Diffusion'] or [float('inf')]) else 'Baseline'
    } for f in all_fold_results])
    
    # Generate model rows HTML
    model_rows = ""
    for rank, (model_name, metrics) in enumerate(sorted_by_crps, 1):
        is_diffusion = model_name == 'Diffusion'
        row_class = 'diffusion-row' if is_diffusion else ''
        badge = '<span class="winner-badge">👑 BEST</span>' if rank == 1 else ''
        
        model_rows += f'''
        <tr class="{row_class}">
            <td><span class="rank-badge">#{rank}</span></td>
            <td><strong>{model_name}</strong> {badge}</td>
                <td class="mono">{metrics['crps']:.4f}</td>
                <td class="mono">{metrics['coverage_90']:.1f}%</td>
                <td class="mono">{metrics['mae']:.4f}</td>
                <td class="mono">{metrics['vol_ratio']:.2f}x</td>
        </tr>'''
    
    # Generate fold rows HTML
    fold_rows = ""
    for f in all_fold_results:
        diff_crps_fold = f['results'].get('Diffusion', {}).get('crps', float('inf'))
        baseline_crps_fold = min([v['crps'] for k, v in f['results'].items() if k != 'Diffusion'] or [float('inf')])
        best_baseline_fold = min([(k, v['crps']) for k, v in f['results'].items() if k != 'Diffusion'], key=lambda x: x[1], default=('N/A', 0))
        
        delta = diff_crps_fold - baseline_crps_fold
        delta_class = 'positive' if delta < 0 else 'negative'
        winner_icon = '✓' if delta < 0 else '✗'
        
        fold_rows += f'''
        <tr>
            <td>Fold {f['fold']}</td>
            <td>{f['test_period']}</td>
            <td class="mono">{diff_crps_fold:.4f}</td>
            <td>{best_baseline_fold[0]}</td>
            <td class="mono">{baseline_crps_fold:.4f}</td>
            <td class="mono {delta_class}">{delta:+.4f} {winner_icon}</td>
        </tr>'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Diffusion Model — Research Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0f1419;
            --bg-card: #1a1f2e;
            --bg-card-hover: #242b3d;
            --text-primary: #e7e9ea;
            --text-secondary: #8b98a5;
            --text-muted: #6e7681;
            --accent-blue: #1d9bf0;
            --accent-green: #00ba7c;
            --accent-red: #f4212e;
            --accent-orange: #ff7a00;
            --accent-purple: #7856ff;
            --border-color: #2f3542;
            --gradient-blue: linear-gradient(135deg, #1d9bf0 0%, #7856ff 100%);
            --gradient-green: linear-gradient(135deg, #00ba7c 0%, #00d68f 100%);
            --shadow-lg: 0 10px 40px rgba(0,0,0,0.3);
        }}
        
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        /* Header */
        .header {{
            text-align: center;
            padding: 3rem 0;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 2rem;
        }}
        
        .header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--gradient-blue);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }}
        
        .header .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            margin-bottom: 1rem;
        }}
        
        .header .meta {{
            display: flex;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .badge-blue {{ background: rgba(29, 155, 240, 0.2); color: var(--accent-blue); }}
        .badge-green {{ background: rgba(0, 186, 124, 0.2); color: var(--accent-green); }}
        .badge-orange {{ background: rgba(255, 122, 0, 0.2); color: var(--accent-orange); }}
        
        /* Executive Summary Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .summary-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .summary-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }}
        
        .summary-card .value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }}
        
        .summary-card .label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        .summary-card.success .value {{ color: var(--accent-green); }}
        .summary-card.warning .value {{ color: var(--accent-orange); }}
        .summary-card.info .value {{ color: var(--accent-blue); }}
        
        /* Status Banner */
        .status-banner {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }}
        
        .status-banner.success {{ border-left: 4px solid var(--accent-green); }}
        .status-banner.warning {{ border-left: 4px solid var(--accent-orange); }}
        
        .status-icon {{
            font-size: 2rem;
        }}
        
        .status-content h3 {{
            font-size: 1.1rem;
            margin-bottom: 0.25rem;
        }}
        
        .status-content p {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        /* Section */
        .section {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .section-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .section-title .icon {{
            font-size: 1.2rem;
        }}
        
        /* Tables */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .data-table th {{
            background: var(--bg-dark);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .data-table td {{
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}
        
        .data-table tr:hover {{
            background: var(--bg-card-hover);
        }}
        
        .data-table .diffusion-row {{
            background: rgba(29, 155, 240, 0.1);
        }}
        
        .data-table .mono {{
            font-family: 'JetBrains Mono', monospace;
        }}
        
        .rank-badge {{
            display: inline-block;
            width: 28px;
            height: 28px;
            line-height: 28px;
            text-align: center;
            border-radius: 50%;
            background: var(--bg-dark);
            font-size: 0.8rem;
            font-weight: 600;
        }}
        
        .winner-badge {{
            background: var(--gradient-green);
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            margin-left: 0.5rem;
        }}
        
        .positive {{ color: var(--accent-green); }}
        .negative {{ color: var(--accent-red); }}
        
        /* Charts */
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        
        .chart-container {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1rem;
        }}
        
        .chart-title {{
            font-size: 0.9rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }}
        
        /* Footer */
        .footer {{
            text-align: center;
            padding: 2rem 0;
            border-top: 1px solid var(--border-color);
            margin-top: 2rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .container {{ padding: 1rem; }}
            .header h1 {{ font-size: 1.8rem; }}
            .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <h1>📊 Market Diffusion Model</h1>
            <p class="subtitle">Walk-Forward Cross-Validation Research Report</p>
            <div class="meta">
                <span><strong>Target:</strong> {conf.ticker}</span>
                <span><strong>Period:</strong> {conf.data_start} — 2025</span>
                <span><strong>Folds:</strong> {total_folds}</span>
                <span><strong>Generated:</strong> {generation_date}</span>
            </div>
        </header>
        
        <!-- Executive Summary -->
        <div class="summary-grid">
            <div class="summary-card {'success' if diff_rank == 1 else 'info'}">
                <div class="value">#{diff_rank}</div>
                <div class="label">Overall Rank</div>
            </div>
            <div class="summary-card info">
                <div class="value">{diff_crps:.4f}</div>
                <div class="label">Mean CRPS</div>
                </div>
            <div class="summary-card {'success' if diff_coverage > 80 else 'warning'}">
                <div class="value">{diff_coverage:.1f}%</div>
                <div class="label">90% Coverage</div>
                </div>
            <div class="summary-card {'success' if 0.8 <= diff_vol <= 1.2 else 'warning'}">
                <div class="value">{diff_vol:.2f}x</div>
                <div class="label">Vol Ratio</div>
                </div>
            <div class="summary-card {'success' if win_rate > 50 else 'warning'}">
                <div class="value">{wins}/{total_folds}</div>
                <div class="label">Folds Won</div>
            </div>
            <div class="summary-card info">
                <div class="value">{win_rate:.0f}%</div>
                <div class="label">Win Rate</div>
            </div>
        </div>
        
        <!-- Status Banner -->
        <div class="status-banner {status_class}">
            <div class="status-icon">{status_icon}</div>
            <div class="status-content">
                <h3>Diffusion Model {status_text}</h3>
                <p>{'The diffusion model achieves the best CRPS score, demonstrating superior probabilistic forecasting.' if best_model == 'Diffusion' else f'The {best_baseline_name} baseline currently outperforms. {improvement_text} on CRPS.'}</p>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="chart-grid">
            <div class="chart-container">
                <div class="chart-title">📈 CRPS by Model (Lower is Better)</div>
                <div id="crpsChart" style="height: 300px;"></div>
            </div>
            <div class="chart-container">
                <div class="chart-title">📊 Fold-by-Fold Performance</div>
                <div id="foldChart" style="height: 300px;"></div>
            </div>
        </div>
        
        <!-- Model Comparison Table -->
        <div class="section">
            <h2 class="section-title"><span class="icon">🏆</span> Model Rankings</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>CRPS ↓</th>
                        <th>90% Coverage</th>
                        <th>MAE</th>
                        <th>Vol Ratio</th>
                    </tr>
                </thead>
                <tbody>
                    {model_rows}
                </tbody>
            </table>
            </div>
            
        <!-- Fold Results Table -->
        <div class="section">
            <h2 class="section-title"><span class="icon">📋</span> Fold-by-Fold Results</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Fold</th>
                        <th>Test Year</th>
                        <th>Diffusion CRPS</th>
                        <th>Best Baseline</th>
                        <th>Baseline CRPS</th>
                        <th>Δ (Diff - Base)</th>
                    </tr>
                </thead>
                <tbody>
                    {fold_rows}
                </tbody>
            </table>
        </div>
        
        {forward_prediction_html}
        
        <!-- Key Insights -->
        <div class="section">
            <h2 class="section-title"><span class="icon">💡</span> Key Insights</h2>
            <ul style="list-style: none; padding: 0;">
                <li style="padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                    <strong>Performance:</strong> Diffusion model ranks <span class="badge badge-blue">#{diff_rank}</span> with CRPS of {diff_crps:.4f} ± {diff_crps_std:.4f}
                </li>
                <li style="padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                    <strong>Calibration:</strong> {'Well-calibrated' if 0.85 <= diff_vol <= 1.15 else 'Needs calibration'} — Vol ratio of {diff_vol:.2f}x {'(target: 1.0x)' if abs(diff_vol - 1.0) > 0.15 else '✓'}
                </li>
                <li style="padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                    <strong>Coverage:</strong> {diff_coverage:.1f}% of actuals fall within 90% CI {'(good)' if diff_coverage > 80 else '(needs improvement)'}
                </li>
                <li style="padding: 0.5rem 0;">
                    <strong>Consistency:</strong> Won {wins}/{total_folds} folds ({win_rate:.0f}% win rate)
                </li>
            </ul>
        </div>
        
        <!-- Footer -->
        <footer class="footer">
            <p>Generated by Market Diffusion Model • {generation_date} at {generation_time}</p>
            <p style="margin-top: 0.5rem; font-size: 0.8rem;">
                ⚠️ This is a statistical model for research purposes. Not financial advice.
            </p>
        </footer>
    </div>
    
    <script>
        const comparisonData = {comparison_data};
        const foldData = {fold_data};
        
        // CRPS Bar Chart
        const crpsColors = comparisonData.models.map(m => 
            m === 'Diffusion' ? '#1d9bf0' : '#6e7681'
        );
        
        Plotly.newPlot('crpsChart', [{{
            x: comparisonData.models,
            y: comparisonData.crps,
            type: 'bar',
            marker: {{ 
                color: crpsColors,
                line: {{ color: 'rgba(255,255,255,0.1)', width: 1 }}
            }},
            text: comparisonData.crps.map(v => v.toFixed(4)),
            textposition: 'outside',
            textfont: {{ color: '#e7e9ea', size: 11 }}
        }}], {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {{ color: '#8b98a5', family: 'Inter' }},
            margin: {{ t: 20, b: 60, l: 50, r: 20 }},
            xaxis: {{ 
                gridcolor: '#2f3542',
                tickfont: {{ size: 11 }}
            }},
            yaxis: {{ 
                gridcolor: '#2f3542',
                title: 'CRPS',
                tickfont: {{ size: 11 }}
            }},
            hoverlabel: {{
                bgcolor: '#1a1f2e',
                bordercolor: '#2f3542',
                font: {{ color: '#e7e9ea', size: 13 }}
            }}
        }}, {{ responsive: true, displayModeBar: false }});
        
        // Fold Performance Chart
        Plotly.newPlot('foldChart', [
            {{
                x: foldData.map(f => f.year),
                y: foldData.map(f => f.diffusion),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Diffusion',
                line: {{ color: '#1d9bf0', width: 2 }},
                marker: {{ size: 8 }}
            }},
            {{
                x: foldData.map(f => f.year),
                y: foldData.map(f => f.best_baseline),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Best Baseline',
                line: {{ color: '#ff7a00', width: 2, dash: 'dash' }},
                marker: {{ size: 8 }}
            }}
        ], {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            font: {{ color: '#8b98a5', family: 'Inter' }},
            margin: {{ t: 20, b: 60, l: 50, r: 20 }},
            xaxis: {{ 
                gridcolor: '#2f3542',
                title: 'Test Year',
                tickfont: {{ size: 11 }}
            }},
            yaxis: {{ 
                gridcolor: '#2f3542',
                title: 'CRPS',
                tickfont: {{ size: 11 }}
            }},
            legend: {{
                orientation: 'h',
                y: -0.2,
                x: 0.5,
                xanchor: 'center'
            }},
            showlegend: true,
            hoverlabel: {{
                bgcolor: '#1a1f2e',
                bordercolor: '#2f3542',
                font: {{ color: '#e7e9ea', size: 13 }}
            }}
        }}, {{ responsive: true, displayModeBar: false }});
        
        {forward_prediction_js}
    </script>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n📄 Research report saved to '{output_path}'")
    return output_path


def _generate_results_table_rows(sorted_models):
    """Legacy function - kept for compatibility."""
    return ""


def _generate_fold_table_rows(all_fold_results):
    """Legacy function - kept for compatibility."""
    return ""

def main():
    """
    Main execution with Walk-Forward Cross-Validation.
    
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                    WALK-FORWARD CROSS-VALIDATION                          │
    ├──────────────────────────────────────────────────────────────────────────┤
    │ This is the gold standard for time series model evaluation.               │
    │                                                                           │
    │ Key principles:                                                           │
    │ 1. NEVER use future data for training (prevents look-ahead bias)          │
    │ 2. Scaler is FIT ONLY on training data (prevents information leakage)     │
    │ 3. Expanding window: more data → better model (simulates real deployment) │
    │ 4. Multiple test periods: tests generalization across market regimes      │
    │                                                                           │
    │ Fold structure:                                                           │
    │   Fold 1: Train [2010-2016] → Val [2017] → Test [2018]                   │
    │   Fold 2: Train [2010-2017] → Val [2018] → Test [2019]                   │
    │   ...                                                                     │
    │   Fold 7: Train [2010-2022] → Val [2023] → Test [2024]                   │
    │                                                                           │
    │ The VALIDATION set is used for early stopping / hyperparameter tuning.    │
    │ The TEST set is NEVER touched during training - only for final eval.      │
    └──────────────────────────────────────────────────────────────────────────┘
    """
    conf = Config()
    
    # Auto-optimize for Colab
    if detect_colab() and conf.device == "cuda":
        # Increase batch size on Colab GPU if memory allows
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb > 10:  # >10GB (A100)
                conf.batch_size = min(128, conf.batch_size * 2)
                print(f"✓ Colab A100 detected - Optimized batch_size: {conf.batch_size}")
            elif gpu_memory_gb > 5:  # >5GB (T4)
                conf.batch_size = min(96, conf.batch_size)
                print(f"✓ Colab T4 detected - Using batch_size: {conf.batch_size}")
        except:
            pass
    
    print("="*80)
    print("MARKET DIFFUSION - Walk-Forward Cross-Validation")
    print("="*80)
    print(f"Ticker: {conf.ticker}")
    print(f"Device: {conf.device}")
    if conf.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Epochs per fold: {conf.epochs}")
    print(f"Batch size: {conf.batch_size}")
    print(f"Mixed Precision (AMP): {conf.use_amp}")
    print(f"Number of folds: {len(conf.walk_forward_folds)}")
    print("="*80)
    
    # ==========================================
    # 1. FETCH ALL DATA
    # ==========================================
    if conf.use_mock_data:
        dates = pd.date_range("2010-01-01", "2025-11-28", freq='B')
        returns = np.random.normal(0.0003, 0.01, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        df_full = pd.DataFrame({'price': prices, 'returns': returns}, index=dates)
    else:
        try:
            feature_tickers = getattr(conf, 'feature_tickers', None)
            df_full = fetch_data(
                conf.ticker, 
                start_date=conf.data_start, 
                end_date="2025-11-28",
                feature_tickers=feature_tickers
            )
        except Exception as e:
            print(f"Error fetching data: {e}. Switching to mock data.")
            conf.use_mock_data = True
            return main()
    
    print(f"\nTotal data: {len(df_full)} trading days ({df_full.index[0].date()} to {df_full.index[-1].date()})")
    
    # ==========================================
    # 2. WALK-FORWARD CROSS-VALIDATION LOOP
    # ==========================================
    all_fold_results = []
    
    for fold_idx, (train_end, val_end, test_end) in enumerate(conf.walk_forward_folds):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{len(conf.walk_forward_folds)}")
        print(f"  Train: {conf.data_start} → {train_end}")
        print(f"  Val:   {train_end} → {val_end}")
        print(f"  Test:  {val_end} → {test_end}")
        print(f"{'='*80}")
        
        # Split data by date (scaler fit on train only!)
        train_df, val_df, test_df, scaler = split_data_by_date(
            df_full, train_end, val_end, test_end, conf.data_start
        )
        
        print(f"  Train samples: {len(train_df)}")
        print(f"  Val samples:   {len(val_df)}")
        print(f"  Test samples:  {len(test_df)}")
        
        # Check if we have enough data
        min_samples = conf.seq_length + 10
        if len(train_df) < min_samples or len(test_df) < min_samples:
            print(f"  ⚠️  Skipping fold - insufficient data")
            continue
        
        # Create datasets
        train_dataset = FinancialDataset(train_df, conf.seq_length, conf.cond_length, use_scaled=True)
        test_dataset = FinancialDataset(test_df, conf.seq_length, conf.cond_length, use_scaled=True)
        
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"  ⚠️  Skipping fold - no valid windows")
            continue
        
        # ==========================================
        # 3. TRAIN MODEL
        # ==========================================
        model_type = getattr(conf, 'model_type', 'diffusion')
        
        if model_type == 'probabilistic':
            print(f"\n  Training Probabilistic Forecaster...")
            model = train_probabilistic_model(train_dataset, conf, verbose=True)
            diffuser = None  # Not needed for probabilistic model
        else:
            print(f"\n  Training Diffusion Model...")
            model, diffuser = train_diffusion_model(train_dataset, conf, verbose=True)
        
        # Use regime-aware scaling (fast, no calibration needed)
        calibrated_scale = None
        
        # ==========================================
        # 4. FIT BASELINE MODELS
        # ==========================================
        print(f"\n  Fitting Baseline Models...")
        train_returns = train_df['returns'].values
        
        baselines = get_all_baselines()
        for baseline in baselines:
            baseline.fit(train_returns, pred_length=conf.pred_length)
            print(f"    ✓ {baseline.name}")
        
        # ==========================================
        # 5. EVALUATE ALL MODELS ON TEST SET
        # ==========================================
        print(f"\n  Evaluating on {min(conf.num_test_windows, len(test_dataset))} test windows...")
        
        fold_results = evaluate_on_test_windows(
            model, diffuser, test_dataset, baselines, train_returns, scaler, conf,
            calibrated_scale=calibrated_scale
        )
        
        if fold_results:
            aggregated = aggregate_results(fold_results)
            all_fold_results.append({
                'fold': fold_idx + 1,
                'test_period': f"{val_end[:4]}",
                'results': aggregated
            })
            
            print_comparison_table(aggregated, f"(Fold {fold_idx + 1} - Test Year: {val_end[:4]})")
    
    # ==========================================
    # 6. AGGREGATE ACROSS ALL FOLDS
    # ==========================================
    print("\n" + "="*80)
    print("FINAL AGGREGATED RESULTS ACROSS ALL FOLDS")
    print("="*80)
    
    if not all_fold_results:
        print("No valid fold results to aggregate.")
        return
    
    # Combine results across folds
    combined_results = {}
    for fold_result in all_fold_results:
        for model_name, metrics in fold_result['results'].items():
            if model_name not in combined_results:
                combined_results[model_name] = {
                    'coverage_90': [],
                    'coverage_50': [],
                    'mae': [],
                    'vol_ratio': [],
                    'crps': [],
                }
            combined_results[model_name]['coverage_90'].append(metrics['coverage_90'])
            combined_results[model_name]['coverage_50'].append(metrics['coverage_50'])
            combined_results[model_name]['mae'].append(metrics['mae'])
            combined_results[model_name]['vol_ratio'].append(metrics['vol_ratio'])
            combined_results[model_name]['crps'].append(metrics['crps'])
    
    # Average across folds
    final_summary = {}
    for model_name, metrics in combined_results.items():
        final_summary[model_name] = {
            'coverage_90': np.mean(metrics['coverage_90']),
            'coverage_50': np.mean(metrics['coverage_50']),
            'mae': np.mean(metrics['mae']),
            'vol_ratio': np.mean(metrics['vol_ratio']),
            'crps': np.mean(metrics['crps']),
            'crps_std': np.std(metrics['crps']),
            'n_folds': len(metrics['crps']),
        }
    
    # Print final table
    print(f"\n{'Model':<20} {'90% Cov':>10} {'MAE':>10} {'Vol Ratio':>12} {'CRPS':>10} {'CRPS Std':>10}")
    print(f"{'-'*80}")
    
    sorted_models = sorted(final_summary.items(), key=lambda x: x[1]['crps'])
    
    for model_name, metrics in sorted_models:
        prefix = "→ " if model_name == "Diffusion" else "  "
        print(f"{prefix}{model_name:<18} {metrics['coverage_90']:>9.1f}% {metrics['mae']:>10.4f} "
              f"{metrics['vol_ratio']:>11.2f}x {metrics['crps']:>10.4f} {metrics['crps_std']:>10.4f}")
    
    print(f"\n{'='*80}")
    
    # Statistical significance test
    if 'Diffusion' in combined_results:
        diff_crps = combined_results['Diffusion']['crps']
        
        print("\nSTATISTICAL SIGNIFICANCE (paired t-test on CRPS):")
        print("-" * 50)
        
        for model_name, metrics in combined_results.items():
            if model_name == 'Diffusion':
                continue
            
            other_crps = metrics['crps']
            
            if len(diff_crps) == len(other_crps) and len(diff_crps) > 1:
                t_stat, p_value = stats.ttest_rel(diff_crps, other_crps)
                
                diff_mean = np.mean(diff_crps)
                other_mean = np.mean(other_crps)
                
                if diff_mean < other_mean:
                    winner = "Diffusion"
                    improvement = ((other_mean - diff_mean) / other_mean) * 100
                else:
                    winner = model_name
                    improvement = ((diff_mean - other_mean) / diff_mean) * 100
                
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  Diffusion vs {model_name:<15}: p={p_value:.4f} {sig:3} | Winner: {winner} (+{improvement:.1f}%)")
    
    # ==========================================
    # 7. GENERATE FORWARD PREDICTION (before report so we can include it)
    # ==========================================
    print("\n" + "="*80)
    print("GENERATING FORWARD MARKET PREDICTION")
    print("="*80)
    
    simulation_data = generate_forward_prediction(df_full, conf)
    
    # ==========================================
    # 8. GENERATE RESEARCH REPORT (with simulation data)
    # ==========================================
    generate_research_report(all_fold_results, final_summary, conf, simulation_data=simulation_data)
    
    print("\n" + "="*80)
    print("WALK-FORWARD CROSS-VALIDATION COMPLETE")
    print("="*80)


def generate_forward_prediction(df_full, conf, num_paths=1000):
    """
    Generate forward-looking predictions for what will happen next in the market.
    
    Creates professional, meaningful visualizations including:
    - Price forecast with confidence intervals
    - Risk metrics (VaR, Expected Shortfall)
    - Comparison to random walk baseline
    - Key milestone predictions (1-week, 1-month, etc.)
    """
    print(f"\nTraining final model on ALL available data...")
    
    # Use all data for training
    scaler = StandardScaler()
    all_returns = df_full['returns'].values.reshape(-1, 1)
    scaler.fit(all_returns)
    
    df_full_scaled = df_full.copy()
    df_full_scaled['returns_scaled'] = scaler.transform(all_returns)
    
    # Create dataset from all data
    full_dataset = FinancialDataset(df_full_scaled, conf.seq_length, conf.cond_length, use_scaled=True)
    
    print(f"  Training samples: {len(full_dataset)}")
    
    # Train model
    model, diffuser = train_diffusion_model(full_dataset, conf, verbose=True)
    
    # Get the most recent history for conditioning
    recent_returns = df_full['returns'].values[-conf.cond_length:]
    recent_returns_scaled = scaler.transform(recent_returns.reshape(-1, 1)).flatten()
    
    # Prepare conditioning tensor
    history = torch.tensor(recent_returns_scaled, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    history = history.to(conf.device)
    
    print(f"\n  Generating {num_paths} Monte Carlo paths for next {conf.pred_length} trading days...")
    
    # Generate paths
    model.eval()
    with torch.no_grad():
        gen_paths_scaled = diffuser.sample_ddim(
            model,
            history.repeat(num_paths, 1, 1),
            eta=conf.ddim_eta,
            temperature=conf.sampling_temperature
        )
        gen_paths_scaled = gen_paths_scaled.squeeze(1).cpu().numpy()
    
    # Inverse transform
    gen_paths = gen_paths_scaled * scaler.scale_[0] + scaler.mean_[0]
    
    # Apply coverage boost (same as evaluation)
    coverage_boost = getattr(conf, 'coverage_boost', 1.0)
    if coverage_boost != 1.0:
        path_mean = np.mean(gen_paths, axis=0, keepdims=True)
        gen_paths = path_mean + (gen_paths - path_mean) * coverage_boost
    
    # Also generate Random Walk baseline for comparison
    hist_mean = np.mean(df_full['returns'].values)
    hist_std = np.std(df_full['returns'].values)
    rw_paths = np.random.normal(hist_mean, hist_std, size=(num_paths, conf.pred_length))
    
    # Convert to cumulative returns and prices
    last_price = df_full['price'].iloc[-1]
    last_date = df_full.index[-1]
    
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=conf.pred_length)
    
    cum_returns = np.cumsum(gen_paths, axis=1)
    price_paths = last_price * np.exp(cum_returns)
    
    rw_cum_returns = np.cumsum(rw_paths, axis=1)
    rw_price_paths = last_price * np.exp(rw_cum_returns)
    
    # Calculate statistics
    mean_path = np.mean(price_paths, axis=0)
    median_path = np.median(price_paths, axis=0)
    p5 = np.percentile(price_paths, 5, axis=0)
    p10 = np.percentile(price_paths, 10, axis=0)
    p25 = np.percentile(price_paths, 25, axis=0)
    p75 = np.percentile(price_paths, 75, axis=0)
    p90 = np.percentile(price_paths, 90, axis=0)
    p95 = np.percentile(price_paths, 95, axis=0)
    
    # Random walk stats
    rw_median = np.median(rw_price_paths, axis=0)
    rw_p5 = np.percentile(rw_price_paths, 5, axis=0)
    rw_p95 = np.percentile(rw_price_paths, 95, axis=0)
    
    # Historical context
    hist_dates = df_full.index[-120:]
    hist_prices = df_full['price'].iloc[-120:].values
    
    # ============================================================
    # CREATE PROFESSIONAL VISUALIZATION
    # ============================================================
    
    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(16, 12))
    
    # Custom color palette
    BLUE = '#2E86AB'
    GREEN = '#28A745'
    RED = '#DC3545'
    ORANGE = '#FD7E14'
    PURPLE = '#6F42C1'
    GRAY = '#6C757D'
    
    # ============================================================
    # PLOT 1: Main Price Forecast (Large, Top)
    # ============================================================
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    
    # Historical prices
    ax1.plot(hist_dates, hist_prices, color=BLUE, linewidth=2, label='Historical Price')
    
    # Today marker
    ax1.axvline(x=last_date, color=GRAY, linestyle='--', linewidth=1.5, alpha=0.8)
    ax1.text(last_date, ax1.get_ylim()[1] * 0.98, ' Today', fontsize=9, color=GRAY, va='top')
    
    # Sample paths (very light)
    for i in range(min(100, num_paths)):
        ax1.plot(future_dates, price_paths[i], color=GREEN, alpha=0.02, linewidth=0.5)
    
    # Confidence intervals
    ax1.fill_between(future_dates, p5, p95, alpha=0.15, color=GREEN, label='90% Confidence')
    ax1.fill_between(future_dates, p25, p75, alpha=0.25, color=GREEN, label='50% Confidence')
    
    # Median prediction
    ax1.plot(future_dates, median_path, color=GREEN, linewidth=2.5, label='Diffusion Median')
    
    # Random walk comparison (dashed)
    ax1.plot(future_dates, rw_median, color=ORANGE, linewidth=1.5, linestyle='--', alpha=0.7, label='Random Walk')
    ax1.fill_between(future_dates, rw_p5, rw_p95, alpha=0.1, color=ORANGE)
    
    # Formatting
    ax1.set_title(f'{conf.ticker} Price Forecast — Next {conf.pred_length} Trading Days (~3 Months)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.tick_params(axis='x', rotation=30)
    
    # Add price annotations
    ax1.annotate(f'${last_price:.2f}', xy=(last_date, last_price), 
                 xytext=(-50, 10), textcoords='offset points',
                 fontsize=10, fontweight='bold', color=BLUE,
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1))
    
    ax1.annotate(f'${median_path[-1]:.2f}', xy=(future_dates[-1], median_path[-1]),
                 xytext=(10, 0), textcoords='offset points',
                 fontsize=10, fontweight='bold', color=GREEN)
    
    # ============================================================
    # PLOT 2: Key Milestones (Top Right)
    # ============================================================
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    
    # Define milestones
    milestones = [
        ('1 Week', 5),
        ('2 Weeks', 10),
        ('1 Month', 21),
        ('2 Months', 42),
        ('3 Months', 63),
    ]
    
    milestone_data = []
    for name, days in milestones:
        if days <= conf.pred_length:
            idx = min(days - 1, conf.pred_length - 1)
            ret = cum_returns[:, idx] * 100
            milestone_data.append({
                'name': name,
                'median': np.median(ret),
                'p25': np.percentile(ret, 25),
                'p75': np.percentile(ret, 75),
                'prob_pos': (ret > 0).mean() * 100
            })
    
    y_pos = np.arange(len(milestone_data))
    medians = [m['median'] for m in milestone_data]
    errors_low = [m['median'] - m['p25'] for m in milestone_data]
    errors_high = [m['p75'] - m['median'] for m in milestone_data]
    colors = [GREEN if m['median'] > 0 else RED for m in milestone_data]
    
    ax2.barh(y_pos, medians, xerr=[errors_low, errors_high], 
             color=colors, alpha=0.7, capsize=5, error_kw={'linewidth': 1.5})
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([m['name'] for m in milestone_data])
    ax2.set_xlabel('Expected Return (%)', fontsize=10)
    ax2.set_title('Return by Horizon', fontsize=12, fontweight='bold')
    
    # Add probability annotations
    for i, m in enumerate(milestone_data):
        ax2.text(ax2.get_xlim()[1] * 0.95, i, f'{m["prob_pos"]:.0f}%↑', 
                 va='center', ha='right', fontsize=9, color=GRAY)
    
    # ============================================================
    # PLOT 3: Risk Metrics (Bottom Left)
    # ============================================================
    ax3 = plt.subplot2grid((3, 3), (2, 0))
    
    final_returns = cum_returns[:, -1] * 100
    
    # Calculate risk metrics
    var_95 = np.percentile(final_returns, 5)  # 95% VaR (5th percentile of returns)
    var_99 = np.percentile(final_returns, 1)  # 99% VaR
    es_95 = final_returns[final_returns <= var_95].mean()  # Expected Shortfall
    
    # Histogram
    n, bins, patches = ax3.hist(final_returns, bins=50, density=True, alpha=0.7, color=BLUE, edgecolor='white')
    
    # Color the tail red
    for i, (patch, b) in enumerate(zip(patches, bins[:-1])):
        if b < var_95:
            patch.set_facecolor(RED)
    
    # VaR lines
    ax3.axvline(x=var_95, color=RED, linewidth=2, linestyle='--', label=f'95% VaR: {var_95:.1f}%')
    ax3.axvline(x=0, color='black', linewidth=1)
    ax3.axvline(x=np.median(final_returns), color=GREEN, linewidth=2, label=f'Median: {np.median(final_returns):.1f}%')
    
    ax3.set_xlabel(f'{conf.pred_length}-Day Return (%)', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('Return Distribution & Risk', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')
    
    # ============================================================
    # PLOT 4: Volatility Forecast (Bottom Middle)
    # ============================================================
    ax4 = plt.subplot2grid((3, 3), (2, 1))
    
    # Calculate rolling volatility of paths
    window = 5
    path_vols = []
    for d in range(window, conf.pred_length):
        daily_rets = gen_paths[:, d-window:d]
        vols = np.std(daily_rets, axis=1) * np.sqrt(252) * 100  # Annualized
        path_vols.append({
            'day': d,
            'median': np.median(vols),
            'p25': np.percentile(vols, 25),
            'p75': np.percentile(vols, 75)
        })
    
    days = [v['day'] for v in path_vols]
    med_vols = [v['median'] for v in path_vols]
    p25_vols = [v['p25'] for v in path_vols]
    p75_vols = [v['p75'] for v in path_vols]
    
    ax4.fill_between(days, p25_vols, p75_vols, alpha=0.3, color=PURPLE)
    ax4.plot(days, med_vols, color=PURPLE, linewidth=2, label='Expected Volatility')
    
    # Historical vol for reference
    hist_vol = np.std(df_full['returns'].values[-60:]) * np.sqrt(252) * 100
    ax4.axhline(y=hist_vol, color=GRAY, linestyle='--', label=f'Recent Vol: {hist_vol:.1f}%')
    
    ax4.set_xlabel('Days Ahead', fontsize=10)
    ax4.set_ylabel('Annualized Volatility (%)', fontsize=10)
    ax4.set_title('Volatility Forecast', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=8)
    
    # ============================================================
    # PLOT 5: Summary Statistics (Bottom Right)
    # ============================================================
    ax5 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax5.axis('off')
    
    # Calculate all stats
    final_price_median = median_path[-1]
    final_price_mean = mean_path[-1]
    pct_change_median = (final_price_median / last_price - 1) * 100
    pct_change_mean = (final_price_mean / last_price - 1) * 100
    
    prob_positive = (final_returns > 0).mean() * 100
    prob_5_up = (final_returns > 5).mean() * 100
    prob_10_up = (final_returns > 10).mean() * 100
    prob_5_down = (final_returns < -5).mean() * 100
    prob_10_down = (final_returns < -10).mean() * 100
    
    # Sentiment indicator
    if prob_positive > 60:
        sentiment = "BULLISH 📈"
        sent_color = GREEN
    elif prob_positive < 40:
        sentiment = "BEARISH 📉"
        sent_color = RED
    else:
        sentiment = "NEUTRAL ➡️"
        sent_color = GRAY
    
    summary_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       {conf.ticker} FORECAST SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 As of: {last_date.strftime('%B %d, %Y')}
💰 Current Price: ${last_price:.2f}
🎯 Horizon: {conf.pred_length} trading days
🎲 Simulations: {num_paths:,}

━━━━━━ PRICE TARGETS ━━━━━━

Median Target:  ${final_price_median:.2f}  ({pct_change_median:+.1f}%)
Mean Target:    ${final_price_mean:.2f}  ({pct_change_mean:+.1f}%)

90% Range: ${p5[-1]:.2f} — ${p95[-1]:.2f}
50% Range: ${p25[-1]:.2f} — ${p75[-1]:.2f}

━━━━━━ PROBABILITIES ━━━━━━

Positive Return:    {prob_positive:5.1f}%
Gain > 5%:          {prob_5_up:5.1f}%
Gain > 10%:         {prob_10_up:5.1f}%
Loss > 5%:          {prob_5_down:5.1f}%
Loss > 10%:         {prob_10_down:5.1f}%

━━━━━━ RISK METRICS ━━━━━━

95% VaR:           {var_95:+.1f}%
99% VaR:           {var_99:+.1f}%
Expected Shortfall: {es_95:+.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    
    ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9))
    
    # Add sentiment badge
    ax5.text(0.5, 0.02, sentiment, transform=ax5.transAxes, fontsize=16,
             ha='center', fontweight='bold', color=sent_color,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=sent_color, linewidth=2))
    
    # ============================================================
    # FINAL TOUCHES
    # ============================================================
    
    plt.suptitle(f'Market Diffusion Model — {conf.ticker} Forward Prediction', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Add disclaimer
    fig.text(0.5, -0.02, 
             '⚠️ DISCLAIMER: This is a statistical model, not financial advice. Past performance does not guarantee future results.',
             ha='center', fontsize=9, style='italic', color=GRAY)
    
    plt.tight_layout()
    plt.savefig('forward_prediction.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n  ✓ Forward prediction saved to 'forward_prediction.png'")
    
    # ============================================================
    # CONSOLE OUTPUT
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  📊 {conf.ticker} FORWARD PREDICTION — {last_date.strftime('%B %d, %Y')}")
    print(f"{'='*65}")
    print(f"  Current Price:     ${last_price:.2f}")
    print(f"  Forecast Horizon:  {conf.pred_length} trading days (~3 months)")
    print(f"  Monte Carlo Paths: {num_paths:,}")
    print(f"{'='*65}")
    print(f"  📈 PRICE TARGETS ({conf.pred_length}-Day):")
    print(f"     Median:  ${final_price_median:.2f}  ({pct_change_median:+.1f}%)")
    print(f"     Mean:    ${final_price_mean:.2f}  ({pct_change_mean:+.1f}%)")
    print(f"     90% CI:  ${p5[-1]:.2f} — ${p95[-1]:.2f}")
    print(f"{'='*65}")
    print(f"  🎲 PROBABILITIES:")
    print(f"     Positive return:  {prob_positive:.1f}%")
    print(f"     Gain > 5%:        {prob_5_up:.1f}%")
    print(f"     Loss > 5%:        {prob_5_down:.1f}%")
    print(f"{'='*65}")
    print(f"  ⚠️  RISK METRICS:")
    print(f"     95% VaR:          {var_95:+.1f}%  (5% chance of worse)")
    print(f"     Expected Shortfall: {es_95:+.1f}%  (avg loss in worst 5%)")
    print(f"{'='*65}")
    print(f"  🔮 MODEL SENTIMENT: {sentiment}")
    print(f"{'='*65}")
    print(f"\n  ⚠️  DISCLAIMER: This is a statistical model, not financial advice.")
    print(f"      Past performance does not guarantee future results.\n")
    
    return {
        'future_dates': future_dates,
        'price_paths': price_paths,
        'mean_path': mean_path,
        'median_path': median_path,
        'p5': p5,
        'p95': p95,
        'last_price': last_price,
        'last_date': last_date,
        'var_95': var_95,
        'expected_shortfall': es_95,
        'prob_positive': prob_positive,
    }


if __name__ == "__main__":
    main()

