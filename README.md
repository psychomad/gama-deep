# GAMA-Deep

**ML Behavioural Analysis Module for Android Greyware**

Optional ML scoring module for the [GAMA ecosystem](https://github.com/psychomad/gama-framework).

## Three-Channel Architecture

| Channel | Input | Dimension |
|---------|-------|-----------|
| Static features | URI scores, SDK bitmap, permissions | 128-dim |
| Smali embeddings | Opcode histogram + 16 GAMA tokens | 256-dim |
| Network sequences | Zeek logs, temporal ordering | 256-dim |

**Pure Rust. No CUDA. No cudarc. Compiles anywhere.**

## Quick Start
```bash
git clone https://github.com/psychomad/gama-deep
cd gama-deep
cargo build --release
sudo cp target/release/gama-deep /usr/local/bin/

gama-deep analyse path/to/gama-intel/workspace/
```

## Training
```bash
echo '{"class": "C"}' > workspace/20260315_.../deep/label.json
echo '{"class": "A"}' > workspace/clean_app/deep/label.json
gama-deep train path/to/workspaces/ --epochs 50
```

| Class | Target | Meaning |
|-------|--------|---------|
| A | 0.10 | Operational — clean |
| B | 0.35 | Disproportionate |
| C | 0.70 | Concealed — evasion |
| D | 0.95 | Deceptive |

## Community Modules
```python
# Drop in ~/.gama/modules/my_module.py
class MyModule:
    name = "my-module"
    version = "1.0.0"
    input_spec = ["static/uri_scan.json"]

    def analyse(self, workspace_path: Path) -> list:
        return []
```

## The GAMA Ecosystem

| Tool | Role |
|------|------|
| [GAMA Framework](https://github.com/psychomad/gama-framework) | Interactive analyst workspace |
| [GAMA-Intel](https://github.com/psychomad/gama-intel) | Automated static analysis pipeline |
| **GAMA-Deep** (this repo) | ML anomaly scoring |
| GAMA-Community *(coming soon)* | Shared finding knowledge base |

## Authors

**CenturiaLabs / ClickSafe UAE** · audit.centurialabs.pl

## License

MIT
