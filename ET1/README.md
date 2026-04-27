# ET1: Semantic Annotation in Training Data

**Research Question:** Does jsonld-ex annotated training data affect model calibration and uncertainty awareness during fine-tuning?

See [PROTOCOL.md](PROTOCOL.md) for the full experimental protocol.

## Quick Start

```powershell
# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run full pipeline (after implementation)
python -m src.run_pilot
```

## Project Structure

```
ET1/
├── PROTOCOL.md              # Full experimental protocol (pre-registration)
├── configs/
│   └── training_config.yaml # All hyperparameters
├── src/
│   ├── knowledge_base.py    # KB generation
│   ├── data_formatter.py    # 7-condition formatter
│   ├── train.py             # Fine-tuning
│   ├── evaluate.py          # Evaluation suite
│   ├── parse_response.py    # Output parsing
│   ├── metrics.py           # ECE, Brier, AUROC
│   └── statistical_tests.py # Bootstrap, McNemar
├── tests/                   # TDD tests (written first)
├── data/                    # Generated data
├── results/                 # Experiment results
└── notebooks/               # Analysis
```
