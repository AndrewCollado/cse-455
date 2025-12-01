"""
Conditional Multi-Asset Diffusion Transformer (CMADT)
Complete implementation for tech stock forecasting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ==================== DATA COLLECTION & PREPROCESSING ====================

class StockDataCollector:
    """Collect and preprocess multi-asset stock data"""
    
    def __init__(self, 
                 tickers: List[str],
                 start_date: str = "2018-01-01",
                 end_date: str = "2023-12-31",
                 market_indices: List[str] = None):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.market_indices = market_indices or ['^GSPC', '^IXIC', '^DJI']
        
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """Download OHLCV data for all assets"""
        print("Downloading stock data...")
        data = {}
        
        # Download individual stocks
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if len(df) > 0:
                    data[ticker] = df
                    print(f"✓ {ticker}: {len(df)} days")
            except Exception as e:
                print(f"✗ {ticker}: Error - {e}")
        
        # Download market indices
        for index in self.market_indices:
            try:
                df = yf.download(index, start=self.start_date, end=self.end_date, progress=False)
                if len(df) > 0:
                    data[index] = df
                    print(f"✓ {index}: {len(df)} days")
            except Exception as e:
                print(f"✗ {index}: Error - {e}")
        
        return data
    
    def create_multivariate_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all assets into single aligned DataFrame"""
        # Get common dates
        all_dates = set.intersection(*[set(df.index) for df in data.values()])
        all_dates = sorted(list(all_dates))
        
        combined = pd.DataFrame(index=all_dates)
        
        for ticker, df in data.items():
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    combined[f"{ticker}_{col}"] = df.loc[all_dates, col]
        
        # Forward fill any missing values
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        
        print(f"\nCombined dataset: {combined.shape[0]} days, {combined.shape[1]} features")
        return combined
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for each stock"""
        result = df.copy()
        
        for ticker in self.tickers:
            close_col = f"{ticker}_Close"
            if close_col in df.columns:
                # Moving averages
                result[f"{ticker}_MA20"] = df[close_col].rolling(20).mean()
                result[f"{ticker}_MA50"] = df[close_col].rolling(50).mean()
                
                # Volatility (20-day)
                result[f"{ticker}_Volatility"] = df[close_col].rolling(20).std()
                
                # RSI (simplified)
                delta = df[close_col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-10)
                result[f"{ticker}_RSI"] = 100 - (100 / (1 + rs))
        
        result = result.fillna(method='bfill')
        return result


class StockTimeSeriesDataset(Dataset):
    """PyTorch Dataset for sliding window sequences"""
    
    def __init__(self, data: np.ndarray, window_size: int = 60, forecast_horizon: int = 1):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        
    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size:idx + self.window_size + self.forecast_horizon]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ==================== MODEL COMPONENTS ====================

class ConvolutionalEmbedding(nn.Module):
    """C - Convolutional Embedding: Extract local temporal features"""
    
    def __init__(self, input_dim: int, embed_dim: int = 128, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, embed_dim, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=kernel_size//2)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm1d(embed_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.norm(x)
        x = x.transpose(1, 2)  # [batch, seq_len, embed_dim]
        return x


class MultiAssetFusion(nn.Module):
    """M - Multi-Asset Fusion: Aggregate cross-asset information"""
    
    def __init__(self, embed_dim: int, num_assets: int):
        super().__init__()
        self.num_assets = num_assets
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        attended, _ = self.cross_attention(x, x, x)
        x = self.norm(x + attended)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DiffusionAugmentation(nn.Module):
    """D - Diffusion-based data augmentation (simplified)"""
    
    def __init__(self, embed_dim: int, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        self.noise_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def add_noise(self, x, t):
        """Add noise to data based on timestep t"""
        noise = torch.randn_like(x)
        alpha = 1 - t / self.num_steps
        return alpha * x + (1 - alpha) * noise, noise
    
    def forward(self, x, augment=False):
        if not augment or not self.training:
            return x
        
        # Simple diffusion augmentation during training
        t = torch.randint(0, self.num_steps, (1,)).item()
        noisy_x, noise = self.add_noise(x, t)
        predicted_noise = self.noise_predictor(noisy_x)
        
        # Return slightly augmented version
        return x + 0.1 * (noise - predicted_noise)


class TransformerEncoder(nn.Module):
    """T - Transformer Encoder: Capture long-range dependencies"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.transformer(x)
        return self.norm(x)


# ==================== COMPLETE CMADT MODEL ====================

class CMADT(nn.Module):
    """
    Conditional Multi-Asset Diffusion Transformer
    Complete architecture combining C-M-A-D-T components
    """
    
    def __init__(self, 
                 input_dim: int,
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 num_assets: int = 20,
                 dropout: float = 0.1,
                 forecast_horizon: int = 1):
        super().__init__()
        
        # C - Convolutional Embedding
        self.conv_embed = ConvolutionalEmbedding(input_dim, embed_dim)
        
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # M - Multi-Asset Fusion
        self.multi_asset_fusion = MultiAssetFusion(embed_dim, num_assets)
        
        # A - Attention (integrated in Transformer)
        # D - Diffusion Augmentation
        self.diffusion = DiffusionAugmentation(embed_dim)
        
        # T - Transformer Encoder
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers, dropout)
        
        # Regression head for prediction
        self.forecast_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, input_dim * forecast_horizon)
        )
        
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon
        
    def forward(self, x, augment=False):
        """
        Args:
            x: [batch, seq_len, input_dim]
            augment: whether to apply diffusion augmentation
        Returns:
            predictions: [batch, forecast_horizon, input_dim]
        """
        # C - Convolutional embedding
        x = self.conv_embed(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # M - Multi-asset fusion
        x = self.multi_asset_fusion(x)
        
        # D - Diffusion augmentation (during training)
        x = self.diffusion(x, augment=augment)
        
        # T - Transformer encoding (includes A - Attention)
        x = self.transformer(x)
        
        # Use last timestep for prediction
        x = x[:, -1, :]
        
        # Forecast head
        predictions = self.forecast_head(x)
        predictions = predictions.view(-1, self.forecast_horizon, self.input_dim)
        
        return predictions


# ==================== TRAINING & EVALUATION ====================

class CMADTTrainer:
    """Training and evaluation pipeline"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        self.scaler = StandardScaler()
        
    def prepare_data(self, df: pd.DataFrame, 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15,
                     window_size: int = 60):
        """Split data chronologically and create datasets"""
        
        # Normalize data
        data_array = df.values
        data_normalized = self.scaler.fit_transform(data_array)
        
        # Chronological split
        n = len(data_normalized)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_data = data_normalized[:train_end]
        val_data = data_normalized[train_end:val_end]
        test_data = data_normalized[val_end:]
        
        print(f"\nData splits:")
        print(f"  Train: {len(train_data)} days")
        print(f"  Val:   {len(val_data)} days")
        print(f"  Test:  {len(test_data)} days")
        
        # Create datasets
        train_dataset = StockTimeSeriesDataset(train_data, window_size)
        val_dataset = StockTimeSeriesDataset(val_data, window_size)
        test_dataset = StockTimeSeriesDataset(test_data, window_size)
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_loader, val_loader, epochs: int = 50, lr: float = 1e-4):
        """Train the CMADT model"""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward with augmentation
                predictions = self.model(batch_x, augment=True)
                
                # Loss on close prices (we'll predict all features but focus on close)
                loss = criterion(predictions, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    predictions = self.model(batch_x, augment=False)
                    loss = criterion(predictions, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_cmadt_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_cmadt_model.pt'))
        
        return train_losses, val_losses
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        predictions_list = []
        actuals_list = []
        
        criterion = nn.MSELoss()
        test_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x, augment=False)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
                
                predictions_list.append(predictions.cpu().numpy())
                actuals_list.append(batch_y.cpu().numpy())
        
        test_loss /= len(test_loader)
        predictions = np.concatenate(predictions_list, axis=0)
        actuals = np.concatenate(actuals_list, axis=0)
        
        # Compute metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
        # Directional accuracy (for close prices)
        # Assuming close price is one of the features
        pred_direction = np.sign(predictions[:, 0, :])
        actual_direction = np.sign(actuals[:, 0, :])
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        print(f"\n{'='*50}")
        print("TEST SET EVALUATION")
        print(f"{'='*50}")
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE:  {mae:.6f}")
        print(f"Directional Accuracy: {directional_accuracy:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'predictions': predictions,
            'actuals': actuals
        }


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution pipeline"""
    
    print("="*60)
    print("CMADT: Conditional Multi-Asset Diffusion Transformer")
    print("Tech Stock Forecasting Implementation")
    print("="*60)
    
    # Top 20 tech stocks
    TOP_20_TECH = [
        'MSFT', 'NVDA', 'AAPL', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO',
        'ORCL', 'ADBE', 'CRM', 'CSCO', 'INTC', 'AMD', 'IBM', 'QCOM',
        'TXN', 'NOW', 'INTU', 'AMAT'
    ]
    
    # For demo, use subset
    DEMO_TICKERS = TOP_20_TECH[:5]  # Use 5 stocks for faster execution
    
    # 1. Data Collection
    print("\n[1/5] DATA COLLECTION")
    collector = StockDataCollector(
        tickers=DEMO_TICKERS,
        start_date="2018-01-01",
        end_date="2023-12-31"
    )
    
    raw_data = collector.download_data()
    
    # 2. Data Preprocessing
    print("\n[2/5] DATA PREPROCESSING")
    df_combined = collector.create_multivariate_dataset(raw_data)
    df_with_indicators = collector.add_technical_indicators(df_combined)
    
    # 3. Model Initialization
    print("\n[3/5] MODEL INITIALIZATION")
    input_dim = df_with_indicators.shape[1]
    
    model = CMADT(
        input_dim=input_dim,
        embed_dim=128,
        num_heads=8,
        num_layers=6,
        num_assets=len(DEMO_TICKERS),
        dropout=0.1,
        forecast_horizon=1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    print(f"Input dimension: {input_dim} features")
    
    # 4. Training
    print("\n[4/5] TRAINING")
    trainer = CMADTTrainer(model)
    
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(
        df_with_indicators, 
        window_size=60
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    train_losses, val_losses = trainer.train(
        train_loader, 
        val_loader, 
        epochs=30,  # Reduced for demo
        lr=1e-4
    )
    
    # 5. Evaluation
    print("\n[5/5] EVALUATION")
    results = trainer.evaluate(test_loader)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot sample predictions vs actuals
    sample_idx = 0
    plt.plot(results['actuals'][sample_idx, 0, :10], 'o-', label='Actual', alpha=0.7)
    plt.plot(results['predictions'][sample_idx, 0, :10], 's-', label='Predicted', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Normalized Value')
    plt.title('Sample Prediction vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cmadt_results.png', dpi=150, bbox_inches='tight')
    print("\nResults plot saved as 'cmadt_results.png'")
    
    print("\n" + "="*60)
    print("CMADT Implementation Complete!")
    print("="*60)
    
    return model, trainer, results


if __name__ == "__main__":
    main()
