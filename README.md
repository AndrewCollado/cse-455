# Market Diffusion

This project implements a Sentiment-Conditioned Market Simulation using a diffusion model. It fetches financial data (price, VIX), calculates technical indicators (RSI), constructs a sentiment score, and trains a 1D diffusion model to generate future price paths conditioned on market sentiment.

## Project Structure

- `market_diffusion.py`: Main script containing the data pipeline, model architecture (SentimentDiffusion), training loop, and Monte Carlo simulation.
- `requirements.txt`: List of Python dependencies.

## Setup and Usage

It is recommended to run this project in a virtual environment to manage dependencies cleanly.

### 1. Create a Virtual Environment

Run the following command to create a virtual environment named `venv` in the project directory:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

You should see `(venv)` appear at the start of your terminal prompt.

### 3. Install Dependencies

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Run the Simulation

Run the main script:

```bash
python market_diffusion.py
```

The script will:
1.  Fetch data for SPY (or use mock data if offline/configured).
2.  Train the diffusion model for the configured number of epochs.
3.  Run a Monte Carlo simulation for "Extreme Fear" and "Extreme Greed" scenarios.
4.  Save the resulting plot to `monte_carlo_diffusion.png`.

## Configuration

You can adjust hyperparameters in the `Config` class within `market_diffusion.py`:
- `ticker`: Target asset (default "SPY").
- `epochs`: Training epochs.
- `num_paths`: Number of Monte Carlo paths to simulate.
- `use_mock_data`: Set to `True` if you want to run without fetching data from Yahoo Finance.

