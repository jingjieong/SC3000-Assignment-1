# SC3000/CZ3005 Assignment 1

This project fine-tunes the provided character-level NanoGPT model with Direct Preference Optimization (DPO) so that it can answer simple arithmetic and algebra questions.

The implementation is under `NanoGPT-Math/`. The assignment notebook is `NanoGPT-Math/dpo/dpo.ipynb`.

## Setup

Use Python 3.10-3.14 and create a virtual environment from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sc3000 --display-name "Python (SC3000)"
```

The requirements file installs the standard PyPI PyTorch package. If you have an NVIDIA GPU and want CUDA acceleration, replace it with the CUDA build selected for your machine. For the RTX 4070 setup used by this project:

```powershell
python -m pip install --force-reinstall torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
```

Verify the installation:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Model files

The pretrained checkpoint is too large for normal Git hosting. Download `gpt.pt` from the link in `NanoGPT-Math/README.md` and place it at:

```text
NanoGPT-Math/sft/gpt.pt
```

The tokenizer file `NanoGPT-Math/sft/meta.pkl` is already part of the project.

## Run the notebook

Launch Jupyter from the notebook directory so its relative paths resolve correctly:

```powershell
cd NanoGPT-Math\dpo
python -m jupyter lab dpo.ipynb
```

Select the `Python (SC3000)` kernel. Run the cells from top to bottom. Steps 5-8 contain the assignment work: loading the preference data, defining the optimizer, training with DPO, and testing the fine-tuned model.

If using the included GPU configuration, ensure `CUDA_VISIBLE_DEVICES` is set to `0`. For CPU-only execution, set `device = 'cpu'`; training will be substantially slower.

## Team workflow

Commit source code, the notebook, `pos_neg_pairs.json`, documentation, and `requirements.txt`. Do not commit `.venv`, Jupyter checkpoints, experiment logs, or generated checkpoints. Keep the notebook outputs reproducible by running it from top to bottom before submission.

Record each member's contribution in the beginning of the notebook, as required by the assignment. Use small branches or focused commits for data generation, DPO training, testing, and documentation, then review changes before merging.

## Submission files

The final ZIP should contain:

```text
dpo/dpo.pt
dpo/dpo.ipynb
dpo/pos_neg_pairs.json
```

Do not include `.venv` or other local environment files.
