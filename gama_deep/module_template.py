"""
GAMA Community Module — Template
Copy this file to ~/.gama/modules/your_module.py and implement the analyse() method.
GAMA-Intel will discover and load it automatically.

This example implements a simple certificate transparency log checker
that looks up domains found in the workspace against CT logs.
"""

from pathlib import Path
import json


class CertTransparencyChecker:
    """
    Example community module: Certificate Transparency log checker.
    Checks domains observed during analysis against CT logs for suspicious certs.
    """

    name       = "cert-transparency"
    version    = "1.0.0"
    input_spec = [
        "network/dns_classification.json",
        "static/uri_scan.json",
    ]

    def analyse(self, workspace_path: Path) -> list:
        findings = []

        # Load DNS data
        dns_path = workspace_path / "network" / "dns_classification.json"
        if not dns_path.exists():
            return findings

        dns_data = json.loads(dns_path.read_text())

        # Collect tracking domains
        tracking_domains = [
            d["domain"]
            for d in dns_data.get("tracking", [])
        ]

        for domain in tracking_domains[:10]:   # cap to avoid rate limits
            # In a real module: query crt.sh or similar CT log API
            # result = requests.get(f"https://crt.sh/?q={domain}&output=json").json()
            # Here we just demonstrate the finding structure
            if self._is_suspicious_cert_pattern(domain):
                findings.append({
                    "gama_technique":   "GAMA-T000",
                    "classification":   "hypothesis",
                    "description":      f"CT log: suspicious certificate pattern for {domain}",
                    "code_evidence":    "N/A — network finding",
                    "runtime_evidence": f"network/dns_classification.json: {domain}",
                    "policy_gap":       "Pending analyst review",
                    "suspicion_score":  5,
                    "score_signals":    ["CT log pattern match"],
                })

        return findings

    def _is_suspicious_cert_pattern(self, domain: str) -> bool:
        # Placeholder — real implementation queries CT logs
        suspicious_patterns = ["cdn.", "edge.", "s3.", "storage."]
        return any(domain.startswith(p) for p in suspicious_patterns)


# ── Another example: YARA rule scanner ───────────────────────────

class YARAScanner:
    """
    Example: YARA rule scanner for APK assets.
    Requires: pip install yara-python
    Place your .yar rule files in ~/.gama/rules/yara/
    """

    name       = "yara-scanner"
    version    = "1.0.0"
    input_spec = ["static/apktool_out"]

    def analyse(self, workspace_path: Path) -> list:
        try:
            import yara
        except ImportError:
            return []   # graceful fallback — yara not installed

        rules_dir = Path.home() / ".gama" / "rules" / "yara"
        if not rules_dir.exists():
            return []

        findings = []
        apktool_out = workspace_path / "static" / "apktool_out"
        if not apktool_out.exists():
            return []

        # Compile rules
        rule_files = list(rules_dir.glob("*.yar"))
        if not rule_files:
            return []

        try:
            rules = yara.compile(filepaths={f.stem: str(f) for f in rule_files})
        except Exception:
            return []

        # Scan smali files
        for smali_file in list(apktool_out.rglob("*.smali"))[:100]:
            try:
                matches = rules.match(str(smali_file))
                for match in matches:
                    findings.append({
                        "gama_technique":   "GAMA-T000",
                        "classification":   "hypothesis",
                        "description":      f"YARA match: {match.rule} in {smali_file.name}",
                        "code_evidence":    str(smali_file.relative_to(workspace_path)),
                        "runtime_evidence": "N/A — static finding",
                        "policy_gap":       "Pending analyst review",
                        "suspicion_score":  7,
                        "score_signals":    [f"YARA rule: {match.rule}"],
                    })
            except Exception:
                continue

        return findings
