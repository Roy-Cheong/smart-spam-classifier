# Smart Spam Classifier

A small Streamlit app that classifies SMS and Email text as Spam or Not Spam. It supports single‑message checks, batch CSV uploads, and a simple analytics pie chart.

## Features
- Single message classification (SMS or Email)
- Batch CSV upload with downloadable results
- Lightweight keyword highlighting and confidence display
- Uses a local model if available, otherwise downloads from Hugging Face Hub

## Supported Python Versions
- Recommended: Python 3.10 or 3.11 (64‑bit)
- Known issues: Python 3.12+ may lack some wheels; Python 3.13 commonly fails to install `numpy`/`pandas` on Windows without build tools. Use 3.10/3.11 for the smoothest setup.

## Quick Start (Windows 10/11)
1) Create and activate a virtual environment (Python 3.10/3.11):
```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V          # should show 3.11.x or 3.10.x
python -m pip -V   # path should point inside .venv
```

2) Install dependencies (CPU‑only; safest):
```
python -m pip install -U pip setuptools wheel
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.win-py310.txt
```

- If you have an NVIDIA GPU with CUDA 12.1, you can install a CUDA build of PyTorch instead:
```
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.win-py310.txt
```

3) Run the app:
```
python -m streamlit run app/app.py
```

## Quick Start (macOS/Linux)
1) Create and activate a virtual environment (Python 3.10/3.11):
```
python3 -m venv .venv
source .venv/bin/activate
python -V
```

2) Install dependencies:
```
pip install -U pip setuptools wheel
pip install -r requirements.min.txt
```
- If PyTorch fails to install from PyPI, consult https://pytorch.org/get-started/locally/ for a command matching your OS/Python, then re‑run `pip install -r requirements.txt`.

3) Run the app:
```
python -m streamlit run app/app.py
```

## Models
- SMS: `Roy-Cheong/smart-spam-sms`
- Email: `Roy-Cheong/smart-spam-email`

On startup, the app looks for a local model first:
- `model/transformer_sms`
- `model/transformer_email`

If the directory (with `config.json`) is present, it loads locally. Otherwise, it downloads from the Hugging Face Hub.

## Usage
- Choose SMS or Email in the sidebar.
- Paste your text and click “Check” to get a verdict and confidence.
- Batch tab: upload a CSV with a `message` column and download results.
- Analytics tab: shows a pie chart of Spam vs Not Spam from the latest batch.

## CSV Format (Batch Tab)
Example CSV (header + rows):
```
message
"Congrats! You won a prize, click here"
"Reminder: team meeting at 2pm"
```

## Troubleshooting
- Wrong Python active (Windows):
  - Ensure the venv is active: `.\.venv\Scripts\Activate.ps1`
  - Check versions:
    - `python -V` should be 3.10/3.11
    - `python -m pip -V` should point into `.venv`
  - If `py -0p` does not list 3.10/3.11, install Python 3.11 (64‑bit) from python.org.

- PyTorch install issues:
  - Use the official index for CPU: `--index-url https://download.pytorch.org/whl/cpu`
  - For NVIDIA CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`

- Streamlit not found:
  - Make sure you ran installs inside the venv and you launch with `python -m streamlit run app/app.py`.

- Offline/No Internet:
  - Place the model directories under `model/` as noted in Models, so the app loads locally.

## Development
- App entrypoint: `app/app.py`
- UI helpers: `app/layout.py`
- Inference helpers: `app/utils.py`
- Minimal deps: `requirements.txt`
