# ET1 Colab Notebooks

Per-model notebooks for running the ET1 pilot experiment on Google Colab or GPU clusters.

## Available Notebooks

| Notebook | Model | Parameters | Estimated Time |
|----------|-------|-----------|----------------|
| `pilot_smollm2_135m.ipynb` | SmolLM2-135M-Instruct | 135M | ~2-3 hrs (7 conditions × 3 seeds) |
| `pilot_smollm2_360m.ipynb` | SmolLM2-360M-Instruct | 360M | ~4-6 hrs |
| `pilot_qwen25_05b.ipynb` | Qwen2.5-0.5B-Instruct | 500M | ~6-8 hrs |

## Usage

### Google Colab
1. Upload the notebook to Colab
2. Set runtime to **GPU** (T4 minimum, A100 preferred)
3. Run all cells — checkpointing is built in, so Colab disconnects won't lose progress
4. Results are saved to Google Drive under `ET1_results/`

### GPU Cluster
```bash
# Clone the repo
git clone https://github.com/jemsbhai/jsonld-ex-experiments.git
cd jsonld-ex-experiments/ET1

# Install dependencies
pip install -r requirements.txt

# Run for a specific model
python -m src.run_pilot --config configs/training_config.yaml \
    --models smollm2-135m --seeds 42 137 2024

# Run all conditions for all models
python -m src.run_pilot --config configs/training_config.yaml
```

## Checkpointing

Each completed (model × condition × seed) run saves results to `results/{run_key}.json`.
If the script or notebook is restarted, completed runs are automatically skipped.
This means you can safely interrupt and resume at any point.

## Output

Results are saved as JSON files with per-fact predictions and aggregate metrics.
After all runs complete, a summary table is printed comparing conditions.
