import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    use_mock_data = False     # Set to True if you don't have internet
    ticker = "SPY"            # Target ETF
    seq_length = 128          # Total window size (64 past + 64 future)
    cond_length = 64          # How much history the model sees
    pred_length = 64          # How much future the model generates
    
    # Model Hyperparameters
    timesteps = 500           # Diffusion steps
    batch_size = 64
    lr = 1e-3
    epochs = 100              # Increase to 500+ for real results
    hidden_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Monte Carlo Settings
    num_paths = 30            # Number of simulations for Monte Carlo

# ==========================================
# 2. DATA PIPELINE
# ==========================================
def fetch_data(ticker="SPY"):
    """
    Fetches OHLCV data for the ticker and VIX for sentiment.
    Constructs a proxy 'Fear & Greed' index using RSI and VIX.
    """
    print(f"Downloading data for {ticker}...")
    
    # 1. Get Asset Data (Suppress warnings by explicit auto_adjust)
    df = yf.download(ticker, start="2010-01-01", progress=False, auto_adjust=True)['Close']
    if isinstance(df, pd.DataFrame): df = df.iloc[:, 0] # Handle multi-index if needed
    
    # 2. Get VIX (Fear Proxy)
    vix = yf.download("^VIX", start="2010-01-01", progress=False, auto_adjust=True)['Close']
    if isinstance(vix, pd.DataFrame): vix = vix.iloc[:, 0]

    # Align dates
    data = pd.DataFrame({'price': df, 'vix': vix}).dropna()

    # 3. Calculate RSI (Greed Proxy)
    delta = data['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    data = data.dropna()

    # 4. Construct Composite Sentiment Score (-1 to 1)
    # VIX is Fear (High VIX = Low Sentiment). RSI is Momentum (High RSI = High Sentiment).
    # Normalize VIX (usually 10-60) and RSI (0-100) to 0-1
    scaler_vix = MinMaxScaler()
    scaler_rsi = MinMaxScaler()
    
    v_norm = scaler_vix.fit_transform(data[['vix']])
    r_norm = scaler_rsi.fit_transform(data[['rsi']])
    
    # Sentiment = (RSI_norm - VIX_norm). 
    # Result: +1 (High Greed: High RSI, Low VIX), -1 (High Fear: Low RSI, High VIX)
    data['sentiment'] = (r_norm - v_norm).flatten()
    
    # Clip to ensure range [-1, 1] just in case
    data['sentiment'] = data['sentiment'].clip(-1, 1)

    # 5. Log Returns for the Price
    data['returns'] = np.log(data['price'] / data['price'].shift(1))
    data = data.dropna()
    
    return data

class FinancialDataset(Dataset):
    def __init__(self, dataframe, seq_len, cond_len):
        self.seq_len = seq_len
        self.cond_len = cond_len
        self.data = dataframe
        
        # Convert to numpy
        self.returns = torch.tensor(dataframe['returns'].values, dtype=torch.float32).unsqueeze(-1) # (T, 1)
        self.sentiment = torch.tensor(dataframe['sentiment'].values, dtype=torch.float32).unsqueeze(-1) # (T, 1)

    def __len__(self):
        return len(self.returns) - self.seq_len

    def __getitem__(self, idx):
        # Full window: History + Future
        full_window = self.returns[idx : idx + self.seq_len]
        
        # Split
        x_history = full_window[:self.cond_len] # (32, 1)
        x_future = full_window[self.cond_len:]  # (32, 1)
        
        # Sentiment at the START of the prediction period (t=0 relative to generation)
        # We use the sentiment from the last day of history
        sentiment_val = self.sentiment[idx + self.cond_len - 1] 
        
        # Permute to (Channels, Length) for 1D Conv -> (1, 32)
        return x_history.permute(1, 0), x_future.permute(1, 0), sentiment_val

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================
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

class ResidualBlock1D(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        
        # Step/Sentiment Embedding Projection
        self.emb_proj = nn.Linear(emb_dim, channels)
        
        self.act = nn.SiLU()
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x, embedding):
        residual = x
        
        x = self.act(self.conv1(x))
        
        # Add Time/Sentiment Embedding (Global Conditioning)
        # Embed is (Batch, Dim), reshape to (Batch, Dim, 1) to add to sequence
        emb_out = self.emb_proj(embedding).unsqueeze(-1)
        x = x + emb_out 
        
        x = self.act(self.conv2(x))
        x = self.norm(x)
        
        return x + residual

class SentimentDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden = config.hidden_dim
        
        # Input: 1 channel (noise) + 1 channel (history condition) = 2
        self.input_proj = nn.Conv1d(2, self.hidden, 1)
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden),
            nn.Linear(self.hidden, self.hidden * 2),
            nn.SiLU(),
            nn.Linear(self.hidden * 2, self.hidden),
        )
        
        # Sentiment Embedding (Scalar -> Vector)
        self.sentiment_mlp = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden)
        )

        # Backbone (Simple ResNet for 1D)
        self.blocks = nn.ModuleList([
            ResidualBlock1D(self.hidden, self.hidden),
            ResidualBlock1D(self.hidden, self.hidden),
            ResidualBlock1D(self.hidden, self.hidden),
            ResidualBlock1D(self.hidden, self.hidden),
        ])
        
        self.dropout = nn.Dropout(0.1)
        self.final_conv = nn.Conv1d(self.hidden, 1, 1)

    def forward(self, x, t, history, sentiment):
        # 1. Embeddings
        t_emb = self.time_mlp(t)                 # (Batch, Hidden)
        s_emb = self.sentiment_mlp(sentiment)    # (Batch, Hidden)
        
        # Combine Time + Sentiment (Conditioning via Addition)
        global_cond = t_emb + s_emb
        
        # 2. Condition Input
        # Concatenate History and Noisy Future along channel dimension
        # x: (Batch, 1, Seq), history: (Batch, 1, Seq) -> (Batch, 2, Seq)
        net_input = torch.cat([x, history], dim=1)
        
        # 3. Backbone
        h = self.input_proj(net_input)
        
        for block in self.blocks:
            h = block(h, global_cond)
            h = self.dropout(h)
            
        return self.final_conv(h)

# ==========================================
# 4. DIFFUSION UTILITIES
# ==========================================
class DiffusionManager:
    def __init__(self, config):
        self.timesteps = config.timesteps
        self.device = config.device
        
        # Define Beta Schedule (Linear)
        self.betas = torch.linspace(1e-4, 0.02, self.timesteps).to(self.device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x_start, t):
        # Forward Process: q(x_t | x_0)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod[t])[:, None, None]
        
        noise = torch.randn_like(x_start)
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise, noise

    @torch.no_grad()
    def sample(self, model, history, sentiment):
        # Reverse Process: p(x_{t-1} | x_t)
        model.eval()
        batch_size = history.shape[0]
        seq_len = history.shape[2]
        
        # Start from pure noise
        img = torch.randn((batch_size, 1, seq_len)).to(self.device)
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = model(img, t, history, sentiment)
            
            # Algorithm 2 form DDPM paper
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)
            
            # Add temperature to noise to increase diversity
            noise = noise * 1.2
                
            img = (1 / torch.sqrt(alpha)) * (img - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
            
        return img

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    conf = Config()
    
    # 1. Prepare Data
    if conf.use_mock_data:
        # Generate sine waves if no internet
        dates = pd.date_range("2020-01-01", periods=1000)
        df = pd.DataFrame({
            'returns': np.sin(np.arange(1000) * 0.1) * 0.01 + np.random.normal(0, 0.005, 1000),
            'sentiment': np.random.uniform(-1, 1, 1000)
        }, index=dates)
    else:
        try:
            df = fetch_data(conf.ticker)
        except Exception as e:
            print(f"Error fetching data: {e}. Switching to mock data.")
            conf.use_mock_data = True
            main() # Retry with mock
            return

    # --- Scale the returns ---
    scaler = StandardScaler()
    # Reshape to (N, 1) for the scaler
    returns_array = df['returns'].values.reshape(-1, 1)
    df['returns'] = scaler.fit_transform(returns_array)
    # -------------------------

    print(f"Data Prepared: {df.shape[0]} samples.")
    dataset = FinancialDataset(df, conf.seq_length, conf.cond_length)
    dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True)
    
    # 2. Setup Model
    model = SentimentDiffusion(conf).to(conf.device)
    diffuser = DiffusionManager(conf)
    optimizer = optim.AdamW(model.parameters(), lr=conf.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    loss_fn = nn.MSELoss()
    
    # 3. Training Loop
    print("\nStarting Training...")
    loss_history = []
    
    for epoch in range(conf.epochs):
        model.train()
        epoch_loss = 0
        for history, future, sentiment in dataloader:
            history, future, sentiment = history.to(conf.device), future.to(conf.device), sentiment.to(conf.device)
            
            # Sample t
            t = torch.randint(0, conf.timesteps, (history.size(0),), device=conf.device).long()
            
            # Add noise
            noisy_future, noise = diffuser.add_noise(future, t)
            
            # Predict noise
            noise_pred = model(noisy_future, t, history, sentiment)
            
            loss = loss_fn(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        
        scheduler.step(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{conf.epochs} | Loss: {avg_loss:.6f}")

    # 4. Monte Carlo Simulation (Inference)
    print(f"\nRunning Monte Carlo Simulation ({conf.num_paths} paths per scenario)...")
    
    # Take a real sample from the end of the data to use as history
    last_idx = len(dataset) - 1
    sample_hist, _, _ = dataset[last_idx]
    sample_hist = sample_hist.unsqueeze(0).to(conf.device) # (1, 1, 32)
    
    # Generate multiple paths for "Extreme Fear" (-0.9)
    fear_cond = torch.tensor([[-0.9]], device=conf.device) 
    fear_paths = diffuser.sample(
        model, 
        sample_hist.repeat(conf.num_paths, 1, 1), 
        fear_cond.repeat(conf.num_paths, 1)
    )
    
    # Generate multiple paths for "Extreme Greed" (0.9)
    greed_cond = torch.tensor([[0.9]], device=conf.device)
    greed_paths = diffuser.sample(
        model, 
        sample_hist.repeat(conf.num_paths, 1, 1), 
        greed_cond.repeat(conf.num_paths, 1)
    )

    import matplotlib.ticker as mtick

    # 5. Visualization & Analysis
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # Adjust margins to remove space on left/right
    plt.margins(x=0)
    
    def process_paths(returns_tensor, start_price, scaler):
        # Input shape: (Batch, 1, Seq_Len)
        # Squeeze channel dim: (Batch, Seq_Len)
        r = returns_tensor.squeeze(1).cpu().numpy()
        
        # --- Inverse Transform Scaling ---
        r = r * scaler.scale_[0] + scaler.mean_[0]
        # ---------------------------------
        
        # Calculate Cumulative Returns -> Prices
        # We perform cumsum along the TIME axis (axis 1)
        # Formula: P_t = P_0 * exp(cumsum(r))
        cumulative_r = np.cumsum(r, axis=1)
        prices = start_price * np.exp(cumulative_r)
        # Prepend start_price to each path for visual continuity
        # Shape becomes (Batch, Seq_Len + 1)
        start_col = np.full((prices.shape[0], 1), start_price)
        prices = np.hstack([start_col, prices])
        
        return prices

    # Helper to plot confidence intervals
    def plot_confidence_intervals(ax, time_axis, prices, color, label):
        mean_path = np.mean(prices, axis=0)
        p05 = np.percentile(prices, 5, axis=0)
        p95 = np.percentile(prices, 95, axis=0)
        
        # Plot individual paths (thin, transparent)
        for i in range(min(len(prices), 50)): # Limit to 50 lines to avoid clutter
            ax.plot(time_axis, prices[i], color=color, alpha=0.1)
            
        # Plot Mean
        ax.plot(time_axis, mean_path, color=color, linewidth=2.5, linestyle='--', label=f'{label} Mean')
        
        # Plot Confidence Interval
        ax.fill_between(time_axis, p05, p95, color=color, alpha=0.2, label=f'{label} 90% CI')

    # Get start price for plotting continuity
    # History Processing
    hist_r = sample_hist.squeeze(1).cpu().numpy() # (1, 32)
    
    # --- Unscale History ---
    hist_r = hist_r * scaler.scale_[0] + scaler.mean_[0]
    # -----------------------
    
    # Assume arbitrary start of 100 for history, or normalize
    hist_prices = 100 * np.exp(np.cumsum(hist_r, axis=1)) 
    hist_prices = hist_prices[0] # Take first (only) batch item -> (32,)
    
    start_price = hist_prices[-1]
    
    # Process Futures
    fear_prices_arr = process_paths(fear_paths, start_price, scaler)
    greed_prices_arr = process_paths(greed_paths, start_price, scaler)

    # Time axes
    time_hist = np.arange(conf.cond_length)
    # Future time starts at the end of history (conf.cond_length - 1) to connect
    # Length is pred_length + 1 because we prepended the start price
    time_fut = np.arange(conf.cond_length - 1, conf.cond_length + conf.pred_length)

    # Plot History
    ax1.plot(time_hist, hist_prices, color='black', label='History (Real)', linewidth=2.5)
    
    # Plot Monte Carlo Fans
    plot_confidence_intervals(ax1, time_fut, fear_prices_arr, 'red', 'Extreme Fear (-0.9)')
    plot_confidence_intervals(ax1, time_fut, greed_prices_arr, 'green', 'Extreme Greed (+0.9)')

    ax1.set_title(f"Monte Carlo Diffusion: Sentiment-Conditioned Market Simulation\nTicker: {conf.ticker} | Paths: {conf.num_paths} | Epochs: {conf.epochs}")
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Price (Normalized)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Add Secondary Y-Axis for Percentage Change
    ax2 = ax1.twinx()
    
    # Calculate percentage bounds based on the primary axis view
    y1_min, y1_max = ax1.get_ylim()
    
    # Convert price limits to percentage change relative to start_price
    pct_min = ((y1_min - start_price) / start_price) * 100
    pct_max = ((y1_max - start_price) / start_price) * 100
    
    ax2.set_ylim(pct_min, pct_max)
    ax2.set_ylabel("Change from Current Price (%)")
    
    # Format the right y-axis with +/-% signs
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Force the x-axis to be tight
    ax1.set_xlim(time_hist[0], time_fut[-1])
    
    plt.tight_layout()
    plt.savefig("monte_carlo_diffusion.png")
    print("Monte Carlo Simulation complete. Results saved to 'monte_carlo_diffusion.png'.")

if __name__ == "__main__":
    main()

