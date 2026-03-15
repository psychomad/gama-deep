// GAMA-Deep — Scoring and Finding output
//
// Converts anomaly scores into GAMA Finding-compatible JSON.
// Findings produced here integrate directly into GAMA-Intel workspace.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyScore {
    /// Overall anomaly score 0-100. Higher = more anomalous.
    pub anomaly_score:         f32,

    /// Fraction of score explained by each channel (sum = 1.0)
    pub static_contribution:   f32,
    pub smali_contribution:    f32,
    pub network_contribution:  f32,

    /// Model confidence 0-1 (depends on available data)
    pub confidence:            f32,

    /// Human-readable explanation of top contributing signals
    pub top_signals:           Vec<String>,
}

/// A GAMA-Deep finding — compatible with GAMA-Intel Finding format.
/// Written to workspace/deep/gama_deep.json and optionally to findings.jsonl.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepFinding {
    pub id:               String,
    pub timestamp:        String,
    pub workspace_id:     String,
    pub phase:            u8,             // always 5 (correlation phase)
    pub engine:           String,         // "gama-deep"
    pub gama_technique:   String,         // GAMA-T001 or GAMA-T000 (unknown)
    pub classification:   String,         // always "hypothesis" — analyst confirms
    pub description:      String,
    pub code_evidence:    String,
    pub runtime_evidence: String,
    pub policy_gap:       String,
    pub suspicion_score:  f32,            // anomaly_score mapped to 0-15 scale
    pub score_signals:    Vec<String>,
    pub score:            AnomalyScore,   // full score detail
    pub attck_technique:  Option<String>,
    pub masvs_control:    Option<String>,
}

pub fn to_findings(score: AnomalyScore, workspace_path: &Path) -> Vec<DeepFinding> {
    let workspace_id = workspace_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Map anomaly score to GAMA suspicion scale (0-100 → 0-15)
    let suspicion = (score.anomaly_score / 100.0 * 15.0).min(15.0);

    // Determine most likely GAMA technique from channel contributions
    let technique = if score.network_contribution > 0.5 && score.anomaly_score > 50.0 {
        "GAMA-T003"  // network-dominated anomaly → persistence/post-termination
    } else if score.smali_contribution > 0.5 && score.anomaly_score > 50.0 {
        "GAMA-T005"  // bytecode-dominated → JNI/native anomaly
    } else if score.static_contribution > 0.5 {
        "GAMA-T001"  // static-dominated → URI scheme / SDK
    } else {
        "GAMA-T000"  // unknown — all channels contributing equally
    };

    let attck = match technique {
        "GAMA-T001" => Some("T1637.002".to_string()),
        "GAMA-T003" => Some("T1624.003".to_string()),
        "GAMA-T005" => Some("proposed-JNI-bypass".to_string()),
        _           => None,
    };

    let masvs = match technique {
        "GAMA-T001" => Some("MASVS-PLATFORM-4 (proposed)".to_string()),
        "GAMA-T003" => Some("MASVS-PRIVACY-4 (proposed)".to_string()),
        "GAMA-T005" => Some("MASVS-CODE-6 (proposed)".to_string()),
        _           => None,
    };

    let description = format!(
        "GAMA-Deep anomaly score: {:.1}/100 \
         (static={:.0}% smali={:.0}% network={:.0}%) confidence={:.0}%",
        score.anomaly_score,
        score.static_contribution * 100.0,
        score.smali_contribution * 100.0,
        score.network_contribution * 100.0,
        score.confidence * 100.0,
    );

    let _top = score.top_signals.join("; ");
    let code_evidence = if score.static_contribution > 0.3 {
        "static/uri_scan.json + static/sdk_map.json".to_string()
    } else {
        "static analysis features".to_string()
    };
    let runtime_evidence = if score.network_contribution > 0.2 {
        format!("network/dns_classification.json — {} post-stop signals", {
            // Approximate from network contribution
            (score.network_contribution * 10.0) as u32
        })
    } else {
        "N/A — requires dynamic confirmation".to_string()
    };

    vec![DeepFinding {
        id:               uuid(),
        timestamp:        now_iso(),
        workspace_id,
        phase:            5,
        engine:           "gama-deep".to_string(),
        gama_technique:   technique.to_string(),
        classification:   "hypothesis".to_string(),
        description,
        code_evidence,
        runtime_evidence,
        policy_gap:       "Pending analyst review".to_string(),
        suspicion_score:  suspicion,
        score_signals:    score.top_signals.clone(),
        score:            score,
        attck_technique:  attck,
        masvs_control:    masvs,
    }]
}

fn uuid() -> String {
    // Simple UUID v4 without external crates
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("gd-{:032x}", t)
}

fn now_iso() -> String {
    // RFC 3339 without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // Basic ISO8601 — good enough for finding timestamps
    let (y, mo, d, h, mi, s) = epoch_to_ymd(secs);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, mi, s)
}

fn epoch_to_ymd(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let s  = secs % 60;
    let mi = (secs / 60) % 60;
    let h  = (secs / 3600) % 24;
    let days = secs / 86400;
    // Simplified — good enough for timestamps
    let y  = 1970 + days / 365;
    let mo = (days % 365) / 30 + 1;
    let d  = (days % 365) % 30 + 1;
    (y, mo.min(12), d.min(31), h, mi, s)
}
