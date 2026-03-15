// GAMA-Deep — Channel 3: Network Sequence Builder
//
// Reads Zeek logs (conn.log, dns.log, ssl.log) from workspace network/ directory
// and builds temporal sequences for LSTM anomaly detection.
//
// Sequence format: ordered events with features per timestep
// Features per event (16-dim):
//   [0]   event type (0=conn, 1=dns, 2=ssl)
//   [1]   relative timestamp (normalised 0-1 over session)
//   [2]   is_tracking_domain (0/1)
//   [3]   is_post_termination (0/1)
//   [4]   bytes_sent (log-normalised)
//   [5]   bytes_recv (log-normalised)
//   [6]   duration (log-normalised)
//   [7]   port_norm (dest port / 65535)
//   [8]   proto_tcp (0/1)
//   [9]   proto_udp (0/1)
//   [10]  tls_established (0/1)
//   [11]  sni_cdnlike (0/1) — SNI matches CDN pattern
//   [12]  dns_is_sdk (0/1) — query matches SDK domain pattern
//   [13]  dns_is_unknown (0/1) — unclassified domain
//   [14]  query_frequency_norm — how often this domain was queried
//   [15]  inter_arrival_norm — time since last event

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const NET_EVENT_DIM: usize = 16;
pub const MAX_EVENTS:    usize = 256;

/// SDK/tracking domain patterns for quick classification
const TRACKING_PATTERNS: &[&str] = &[
    "mbridge", "mintegral", "mobvista",
    "adjust.com", "adjust.io", "appsflyer", "onelink",
    "firebase", "app-measurement", "moloco", "ironsource",
    "applovin", "unity3d.com", "unityads", "bytedance", "pangle",
    "yandex", "appmetrica", "amplitude", "mixpanel",
    "branch.io", "bnc.lt", "onesignal", "tapjoy",
    "chartboost", "adcolony", "vungle", "inmobi",
    "doubleclick", "googlesyndication", "googleadservices",
    "facebook.net", "fbcdn", "crashlytics",
    "kochava", "singular.net",
];

const CDN_PATTERNS: &[&str] = &[
    "googleapis.com", "cloudfront.net", "fastly.net",
    "cloudflare.com", "akamaized.net", "akamai.net",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEvent {
    pub timestamp:    f64,
    pub event_type:   u8,    // 0=conn, 1=dns, 2=ssl
    pub domain:       String,
    pub features:     Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSequence {
    pub events:            Vec<NetworkEvent>,
    pub session_duration:  f64,
    pub force_stop_ts:     Option<f64>,
    pub post_stop_events:  usize,
    pub tracking_domains:  Vec<String>,
}

pub fn build_sequences(workspace_path: &Path) -> Result<NetworkSequence> {
    let network_dir = workspace_path.join("network");
    let dns_path    = network_dir.join("dns_classification.json");
    let zeek_dir    = network_dir.join("zeek");

    let mut events: Vec<NetworkEvent> = Vec::new();

    // ── DNS events from dns_classification.json ───────────────
    if dns_path.exists() {
        let dns: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&dns_path)?
        )?;

        let mut freq_map: std::collections::HashMap<String, u32> = Default::default();

        // Count all domain frequencies
        for category in &["tracking", "system", "unknown"] {
            if let Some(domains) = dns[category].as_array() {
                for d in domains {
                    let domain = d["domain"].as_str().unwrap_or("").to_string();
                    let count  = d["count"].as_u64().unwrap_or(1) as u32;
                    freq_map.insert(domain, count);
                }
            }
        }

        let max_freq = freq_map.values().copied().max().unwrap_or(1) as f32;

        for category in &["tracking", "unknown"] {
            if let Some(domains) = dns[category].as_array() {
                for d in domains {
                    let domain = d["domain"].as_str().unwrap_or("").to_string();
                    let count  = d["count"].as_u64().unwrap_or(1) as f32;
                    let is_tracking = *category == "tracking";

                    let mut features = vec![0.0f32; NET_EVENT_DIM];
                    features[0]  = 1.0;  // dns event type
                    features[2]  = if is_tracking { 1.0 } else { 0.0 };
                    features[12] = if is_tracking { 1.0 } else { 0.0 };
                    features[13] = if *category == "unknown" { 1.0 } else { 0.0 };
                    features[14] = count / max_freq;

                    events.push(NetworkEvent {
                        timestamp:  0.0,
                        event_type: 1,
                        domain:     domain.clone(),
                        features,
                    });
                }
            }
        }

        // Post-termination events — critical for GAMA-T003
        if let Some(post) = dns["post_termination"].as_array() {
            for p in post {
                let domain    = p["domain"].as_str().unwrap_or("").to_string();
                let delta     = p["delta_s"].as_f64().unwrap_or(0.0);
                let is_track  = is_tracking_domain(&domain);

                let mut features = vec![0.0f32; NET_EVENT_DIM];
                features[0] = 1.0;   // dns
                features[2] = if is_track { 1.0 } else { 0.0 };
                features[3] = 1.0;   // post-termination
                features[12] = if is_track { 1.0 } else { 0.0 };
                // Use delta_s as timestamp for post-stop events
                features[1] = (delta as f32 / 600.0).min(1.0);

                events.push(NetworkEvent {
                    timestamp:  delta,
                    event_type: 1,
                    domain:     domain.clone(),
                    features,
                });
            }
        }
    }

    // ── Zeek conn.log ────────────────────────────────────────
    let conn_log = zeek_dir.join("conn.log");
    if conn_log.exists() {
        let records = parse_zeek_log(&conn_log)?;
        for r in records {
            let ts     = r.get("ts").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            let dst_h  = r.get("id.resp_h").map(|s| s.as_str()).unwrap_or("");
            let proto  = r.get("proto").map(|s| s.as_str()).unwrap_or("tcp");
            let bytes_s = r.get("orig_bytes").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            let bytes_r = r.get("resp_bytes").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            let dur    = r.get("duration").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            let port   = r.get("id.resp_p").and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);

            let mut features = vec![0.0f32; NET_EVENT_DIM];
            features[0]  = 0.0;   // conn type
            features[4]  = log_norm(bytes_s as f32, 1_000_000.0);
            features[5]  = log_norm(bytes_r as f32, 1_000_000.0);
            features[6]  = log_norm(dur as f32, 600.0);
            features[7]  = port as f32 / 65535.0;
            features[8]  = if proto == "tcp" { 1.0 } else { 0.0 };
            features[9]  = if proto == "udp" { 1.0 } else { 0.0 };

            events.push(NetworkEvent {
                timestamp:  ts,
                event_type: 0,
                domain:     dst_h.to_string(),
                features,
            });
        }
    }

    // ── Zeek ssl.log ─────────────────────────────────────────
    let ssl_log = zeek_dir.join("ssl.log");
    if ssl_log.exists() {
        let records = parse_zeek_log(&ssl_log)?;
        for r in records {
            let ts          = r.get("ts").and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0);
            let sni         = r.get("server_name").map(|s| s.as_str()).unwrap_or("");
            let established = r.get("established").map(|v| v == "T").unwrap_or(false);

            let is_cdn = CDN_PATTERNS.iter().any(|&p| sni.contains(p));

            let mut features = vec![0.0f32; NET_EVENT_DIM];
            features[0]  = 2.0 / 3.0;   // ssl type normalised
            features[10] = if established { 1.0 } else { 0.0 };
            features[11] = if is_cdn { 1.0 } else { 0.0 };

            events.push(NetworkEvent {
                timestamp:  ts,
                event_type: 2,
                domain:     sni.to_string(),
                features,
            });
        }
    }

    // ── Sort by timestamp, normalise, truncate ────────────────
    events.sort_by(|a, b| a.timestamp.partial_cmp(&b.timestamp).unwrap());

    let min_ts = events.first().map(|e| e.timestamp).unwrap_or(0.0);
    let max_ts = events.last().map(|e| e.timestamp).unwrap_or(1.0);
    let session_duration = max_ts - min_ts;
    let range = (session_duration).max(1.0);

    // Compute inter-arrival times and normalise timestamps
    let mut prev_ts = min_ts;
    for event in &mut events {
        let rel_ts = (event.timestamp - min_ts) / range;
        let inter  = ((event.timestamp - prev_ts) / range) as f32;
        event.features[1]  = rel_ts as f32;
        event.features[15] = inter.min(1.0);
        prev_ts = event.timestamp;
    }

    events.truncate(MAX_EVENTS);

    // Extract metadata
    let post_stop_events = events.iter().filter(|e| e.features[3] > 0.5).count();
    let tracking_domains: Vec<String> = events.iter()
        .filter(|e| e.features[2] > 0.5)
        .map(|e| e.domain.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // Load force_stop_ts from intake if available
    let intake_path = workspace_path.join("intake.json");
    let force_stop_ts = if intake_path.exists() {
        serde_json::from_str::<serde_json::Value>(
            &std::fs::read_to_string(&intake_path)?
        ).ok()
        .and_then(|v| v["force_stop_ts"].as_f64())
    } else {
        None
    };

    Ok(NetworkSequence {
        events,
        session_duration,
        force_stop_ts,
        post_stop_events,
        tracking_domains,
    })
}

// ── Helpers ───────────────────────────────────────────────────────
fn is_tracking_domain(domain: &str) -> bool {
    TRACKING_PATTERNS.iter().any(|&p| domain.contains(p))
}

fn log_norm(val: f32, max: f32) -> f32 {
    if val <= 0.0 { return 0.0; }
    ((val + 1.0).ln() / (max + 1.0).ln()).min(1.0)
}

/// Parse a Zeek TSV log file into Vec<HashMap<field, value>>
fn parse_zeek_log(path: &Path) -> Result<Vec<std::collections::HashMap<String, String>>> {
    let content = std::fs::read_to_string(path)?;
    let mut fields: Vec<String> = Vec::new();
    let mut records = Vec::new();

    for line in content.lines() {
        if line.starts_with("#fields") {
            fields = line.split('\t').skip(1).map(String::from).collect();
        } else if line.starts_with('#') {
            continue;
        } else if !fields.is_empty() {
            let values: Vec<&str> = line.split('\t').collect();
            if values.len() == fields.len() {
                let record: std::collections::HashMap<String, String> =
                    fields.iter().cloned().zip(values.iter().map(|v| v.to_string())).collect();
                records.push(record);
            }
        }
    }

    Ok(records)
}
