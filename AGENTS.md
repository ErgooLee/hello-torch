# Repository Guide (`hello-torch`)

## Python Environment
- **Conda Environment:** Do not use system Python (3.14). Use the `torch_env` environment:
  - Python path: `~/miniforge3/envs/torch_env/bin/python` (Python 3.10.19, PyTorch 2.11, torchvision, matplotlib, pandas, d2l).
  - Activation: `source ~/miniforge3/bin/activate torch_env`

## Execution & PYTHONPATH Quirks
Scripts across modules use conflicting relative and package-level imports:
- Always run scripts from the project root with `PYTHONPATH=.:mu`:
  ```bash
  PYTHONPATH=.:mu ~/miniforge3/envs/torch_env/bin/python <path/to/script.py>
  ```
- **`paoge/train.py`**: Uses `from paoge.LeNet import LeNet` and expects `./data` relative to project root. Run with:
  ```bash
  PYTHONPATH=. ~/miniforge3/envs/torch_env/bin/python -m paoge.train
  ```
- **`mu/softmax_three_level.py`**: Uses both `from mu import ...` and `from softmax_self import ...`. Requires `PYTHONPATH=.:mu`.
- **`mu/softmax_data.py`**: Hardcodes relative dataset path `"../data"`.

## Directory Overview
- `paoge/`: Deep learning experiments and LeNet CNN on Fashion-MNIST (`LeNet.py`, `train.py`, `hello_attention.py`, tensor autograd demos).
- `mu/`: Dive-into-DL implementations from scratch vs PyTorch modules (linear regression, softmax, MLP, custom `Accumulator` and `Animator`).
- `hello/`: Python syntax reference (`hello_python.py`).
- `data/`: Local FashionMNIST dataset storage.

## Verification
- No formal test runner (`pytest`) is configured.
- Syntax verification:
  ```bash
  ~/miniforge3/envs/torch_env/bin/python -m py_compile <path/to/file.py>
  ```
- Run individual scripts directly to verify runtime behavior.

## Gotchas & Operational Notes
- **Matplotlib / GUI:** Scripts frequently call `plt.show()` and `plt.ion()`. In headless environments, set `matplotlib.use('Agg')` or avoid blocking on GUI windows.
- **External APIs:** `paoge/gemini3.py` requires `GEMINI_API_KEY` and analyzes local `paoge/maliao.mp4`.
- **Plotly:** `mu/liner_reg-plotly.py` writes `plotly_3d_plot.html` and triggers `webbrowser.open_new_tab`.
