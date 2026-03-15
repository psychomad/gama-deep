"""
GAMA-Deep — Python Bridge
Integrates the Rust/Candle ML engine with GAMA-Intel.

This module:
1. Calls the gama-deep Rust binary as a subprocess
2. Parses the JSON output
3. Writes findings to the GAMA-Intel workspace

This is the only file that needs to exist on the Python side.
The community module interface is defined here.
"""

from __future__ import annotations
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol, runtime_checkable

log = logging.getLogger("gama_deep.bridge")


# ── Community module interface ────────────────────────────────────
# Any third-party module drops a Python file in ~/.gama/modules/
# implementing this protocol. GAMA-Intel loads them automatically.

@runtime_checkable
class GAMAModule(Protocol):
    """
    Interface contract for GAMA community modules.
    Implement this protocol to create a module that GAMA-Intel loads automatically.

    Drop your module file in:  ~/.gama/modules/your_module.py
    GAMA-Intel discovers and loads it at analysis time.

    Minimal implementation:
        class MyModule:
            name    = "my-module"
            version = "1.0.0"
            input_spec = ["static/uri_scan.json"]  # files from workspace you need

            def analyse(self, workspace_path: Path) -> list:
                # return list of dicts compatible with Finding.to_dict()
                return []

    The findings you return are appended to the workspace findings.jsonl
    with engine = "your-module-name".
    """
    name:       str
    version:    str
    input_spec: List[str]   # workspace-relative paths this module needs

    def analyse(self, workspace_path: Path) -> List[dict]:
        """
        Analyse a workspace and return a list of finding dicts.
        Each dict must have at minimum:
            gama_technique, classification, description,
            code_evidence, runtime_evidence, suspicion_score
        """
        ...


# ── Module loader ────────────────────────────────────────────────
class ModuleLoader:
    """
    Discovers and loads GAMA community modules from ~/.gama/modules/.
    Called by GAMA-Intel orchestrator during Phase 5 enrichment.
    """

    MODULES_DIR = Path.home() / ".gama" / "modules"

    @classmethod
    def discover(cls) -> List[GAMAModule]:
        modules = []
        if not cls.MODULES_DIR.exists():
            return modules

        import importlib.util
        for py_file in cls.MODULES_DIR.glob("*.py"):
            try:
                spec   = importlib.util.spec_from_file_location(py_file.stem, py_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find classes implementing GAMAModule protocol
                for name in dir(module):
                    obj = getattr(module, name)
                    if (isinstance(obj, type)
                            and obj is not GAMAModule
                            and hasattr(obj, 'name')
                            and hasattr(obj, 'analyse')):
                        try:
                            instance = obj()
                            modules.append(instance)
                            log.info(f"Loaded module: {instance.name} v{getattr(instance, 'version', '?')}")
                        except Exception as e:
                            log.warning(f"Could not instantiate {name} from {py_file}: {e}")

            except Exception as e:
                log.warning(f"Could not load module {py_file}: {e}")

        return modules

    @classmethod
    def run_all(cls, workspace_path: Path) -> List[dict]:
        """Run all discovered modules on a workspace and return all findings."""
        all_findings = []
        for module in cls.discover():
            try:
                findings = module.analyse(workspace_path)
                for f in findings:
                    f["engine"] = f.get("engine", module.name)
                all_findings.extend(findings)
                log.info(f"Module {module.name}: {len(findings)} findings")
            except Exception as e:
                log.error(f"Module {module.name} failed: {e}")
        return all_findings


# ── GAMA-Deep bridge ─────────────────────────────────────────────
class GAMADeepBridge:
    """
    Calls the gama-deep Rust binary and returns findings.
    Falls back gracefully if the binary is not installed.
    """

    BINARY_NAME  = "gama-deep"
    DEFAULT_MODEL = Path.home() / ".gama" / "models" / "gama_deep_v1.safetensors"

    def __init__(self, model_path: Optional[Path] = None,
                 threshold: float = 40.0):
        self.model_path = model_path or self.DEFAULT_MODEL
        self.threshold  = threshold
        self._binary    = shutil.which(self.BINARY_NAME)

    @property
    def available(self) -> bool:
        return self._binary is not None

    def analyse(self, workspace_path: Path) -> List[dict]:
        """
        Run gama-deep on a workspace and return findings as dicts.
        Returns empty list if binary not available (graceful fallback).
        """
        if not self.available:
            log.info(
                "gama-deep binary not found — ML analysis skipped. "
                "Build with: cd gama-deep && cargo build --release"
            )
            return []

        cmd = [
            self._binary, "analyse", str(workspace_path),
            "--model",    str(self.model_path),
            "--threshold", str(self.threshold),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                log.warning(f"gama-deep exited {result.returncode}: {result.stderr[:200]}")
                return []

            findings = json.loads(result.stdout)
            log.info(f"gama-deep: {len(findings)} findings (score >= {self.threshold})")
            return findings

        except subprocess.TimeoutExpired:
            log.warning("gama-deep timed out after 120s")
            return []
        except json.JSONDecodeError as e:
            log.warning(f"gama-deep output parse error: {e}")
            return []
        except Exception as e:
            log.error(f"gama-deep bridge error: {e}")
            return []

    def train(self, dataset_path: Path, output_path: Optional[Path] = None,
              epochs: int = 50) -> dict:
        """
        Train gama-deep model on a dataset of workspaces.
        """
        if not self.available:
            return {"error": "gama-deep binary not found"}

        out = output_path or self.DEFAULT_MODEL
        cmd = [
            self._binary, "train", str(dataset_path),
            "--output", str(out),
            "--epochs", str(epochs),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                return {"error": result.stderr[:400]}
            return json.loads(result.stdout)
        except Exception as e:
            return {"error": str(e)}

    @classmethod
    def write_label(cls, workspace_path: Path, label: str):
        """
        Write a training label for a workspace.
        label: "A" | "B" | "C" | "D"
        Call this after classifying findings in GAMA Framework Phase 5.
        """
        label_dir = workspace_path / "deep"
        label_dir.mkdir(exist_ok=True)
        (label_dir / "label.json").write_text(
            json.dumps({"class": label, "labelled_by": "analyst"})
        )
        log.info(f"Label {label} written to {workspace_path.name}")


# ── Integration with GAMA-Intel orchestrator ─────────────────────
def run_deep_analysis(workspace_path: Path,
                      model_path: Optional[Path] = None,
                      threshold: float = 40.0) -> List[dict]:
    """
    Entry point called by GAMA-Intel orchestrator.
    Runs gama-deep + all community modules.
    Returns combined findings list.
    """
    findings = []

    # Run gama-deep ML engine
    bridge = GAMADeepBridge(model_path=model_path, threshold=threshold)
    findings.extend(bridge.analyse(workspace_path))

    # Run community modules
    findings.extend(ModuleLoader.run_all(workspace_path))

    # Write to workspace findings.jsonl if GAMA-Intel is available
    if findings:
        _append_to_workspace(workspace_path, findings)

    return findings


def _append_to_workspace(workspace_path: Path, findings: List[dict]):
    """Append deep findings to the GAMA-Intel workspace JSONL log."""
    findings_log = workspace_path / "findings.jsonl"
    if not findings_log.exists():
        log.warning(f"findings.jsonl not found in {workspace_path}")
        return

    # Deduplication: skip if same engine+description already present
    existing_keys = set()
    if findings_log.exists():
        for line in findings_log.read_text(errors="ignore").splitlines():
            try:
                f = json.loads(line)
                key = f"{f.get('engine','')}::{f.get('description','')[:60]}"
                existing_keys.add(key)
            except Exception:
                pass

    written = 0
    with open(findings_log, "a") as f:
        for finding in findings:
            key = f"{finding.get('engine','')}::{finding.get('description','')[:60]}"
            if key not in existing_keys:
                f.write(json.dumps(finding, default=str) + "\n")
                existing_keys.add(key)
                written += 1

    log.info(f"gama-deep: {written} new findings written to {findings_log.name}")


# ── CLI for standalone use ────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="GAMA-Deep Python bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyse a workspace
  python3 bridge.py analyse /path/to/workspace

  # Train on dataset
  python3 bridge.py train /path/to/workspaces --epochs 50

  # Label a workspace (for training)
  python3 bridge.py label /path/to/workspace --class C

  # List available community modules
  python3 bridge.py modules
        """
    )
    sub = parser.add_subparsers(dest="command")

    p_analyse = sub.add_parser("analyse")
    p_analyse.add_argument("workspace")
    p_analyse.add_argument("--model",     default=None)
    p_analyse.add_argument("--threshold", type=float, default=40.0)

    p_train = sub.add_parser("train")
    p_train.add_argument("dataset")
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--output", default=None)

    p_label = sub.add_parser("label")
    p_label.add_argument("workspace")
    p_label.add_argument("--class", dest="label_class", required=True,
                          choices=["A","B","C","D"])

    sub.add_parser("modules")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")

    if args.command == "analyse":
        ws = Path(args.workspace)
        model = Path(args.model) if args.model else None
        findings = run_deep_analysis(ws, model, args.threshold)
        print(json.dumps(findings, indent=2, default=str))

    elif args.command == "train":
        bridge = GAMADeepBridge()
        report = bridge.train(Path(args.dataset),
                              Path(args.output) if args.output else None,
                              args.epochs)
        print(json.dumps(report, indent=2))

    elif args.command == "label":
        GAMADeepBridge.write_label(Path(args.workspace), args.label_class)
        print(f"Labelled {args.workspace} as Class-{args.label_class}")

    elif args.command == "modules":
        modules = ModuleLoader.discover()
        if not modules:
            print("No community modules found in ~/.gama/modules/")
        for m in modules:
            print(f"  {m.name} v{getattr(m, 'version', '?')}")

    else:
        parser.print_help()
