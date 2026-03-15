# GAMA-Deep

ML behavioural analysis module for the GAMA ecosystem.

**CenturiaLabs / ClickSafe UAE**

---

## Architecture

Three-channel multi-modal analysis:

| Channel | Engine | Input | Dimension |
|---------|--------|-------|-----------|
| Static features | Rust | URI scores, SDK bitmap, permissions, native libs | 128-dim |
| Smali embeddings | Rust/Candle | Opcode sequences, synthetic GAMA tokens | 256-dim |
| Network sequences | Rust/Candle | Zeek conn/dns/ssl logs, temporal ordering | 256 × 16 |

All three channels fuse into a single **anomaly score 0–100** plus channel contribution weights and a human-readable explanation.

---

## Build

```bash
# CPU (default)
cargo build --release

# GPU (CUDA)
cargo build --release --features cuda

# With Python bindings
cargo build --release --features python
```

Binary: `target/release/gama-deep`

Install to PATH:
```bash
sudo cp target/release/gama-deep /usr/local/bin/
```

---

## Usage

### Analyse a workspace

```bash
# Analyse a GAMA-Intel workspace
gama-deep analyse /path/to/workspace/20260315_125407_com-app/

# With explicit model
gama-deep analyse /path/to/workspace/ --model ~/.gama/models/gama_deep_v1.safetensors

# Lower threshold (include more findings)
gama-deep analyse /path/to/workspace/ --threshold 30
```

### Python bridge

```python
from gama_deep import GAMADeepBridge, run_deep_analysis

# Full analysis (gama-deep + community modules)
findings = run_deep_analysis(Path("workspace/20260315_..."))

# Just gama-deep
bridge = GAMADeepBridge()
findings = bridge.analyse(Path("workspace/20260315_..."))
```

### Training

```bash
# Train on your workspace dataset
gama-deep train /path/to/workspaces/ --epochs 50

# After labelling workspaces in GAMA Framework Phase 5:
python3 gama_deep/bridge.py label /path/to/workspace --class C
gama-deep train /path/to/workspaces/ --epochs 100
```

---

## Training workflow

GAMA-Deep uses semi-supervised training:

1. **Collect workspaces** — run `gama-intel` on a dataset of APKs (minimum 20 recommended)
2. **Label** — after Phase 5 classification in GAMA Framework, label each workspace:
   ```bash
   python3 gama_deep/bridge.py label workspace/[ws_id] --class A  # clean
   python3 gama_deep/bridge.py label workspace/[ws_id] --class D  # deceptive
   ```
3. **Train** — `gama-deep train workspace/ --epochs 50`
4. **Inference** — `gama-deep analyse workspace/new_analysis/`

Unlabelled workspaces contribute to the autoencoder pretraining (learns what "normal" looks like). Labelled workspaces provide the contrastive signal (Class C/D = anomalous, Class A = normal).

---

## Community Modules

Drop a Python file in `~/.gama/modules/` implementing the `GAMAModule` protocol:

```python
class MyModule:
    name       = "my-module"
    version    = "1.0.0"
    input_spec = ["static/uri_scan.json"]

    def analyse(self, workspace_path: Path) -> list:
        # Return list of finding dicts
        return []
```

See `gama_deep/module_template.py` for complete examples including a YARA scanner and a CT log checker.

---

## Integration with GAMA-Intel

GAMA-Deep integrates as an optional Phase 5 enrichment step. If the `gama-deep` binary is in PATH, GAMA-Intel runs it automatically after correlation. If not installed, analysis continues normally without ML scoring.

To enable in GAMA-Intel:
```bash
# Build and install
cargo build --release
sudo cp target/release/gama-deep /usr/local/bin/

# GAMA-Intel will detect and use it automatically
gama-intel analyse app.apk --skip-dynamic --skip-network
```

---

## Model file

The trained model is saved as a `.safetensors` file at `~/.gama/models/gama_deep_v1.safetensors`.

Without a trained model, GAMA-Deep runs with random weights — the anomaly score will not be meaningful. Train on at least 20 workspaces before using inference in production.

---

## Requirements

- Rust 1.75+
- Python 3.11+ (for bridge.py and community modules)
- ~200MB disk for Candle dependencies
- GPU optional (significant speedup for training on large datasets)
