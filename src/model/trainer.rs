// GAMA-Deep — Trainer

use anyhow::{Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use crate::features;
use crate::model::fusion::{FusionModel, TrainingReport};

#[derive(Debug, Deserialize)]
struct WorkspaceLabel { #[serde(rename = "class")] class: String }

struct Sample {
    static_vec: Vec<f32>,
    smali_emb:  Vec<f32>,
    net_seq:    crate::features::network_features::NetworkSequence,
    label:      Option<f32>,
}

pub fn train(dataset_path: &Path, output_path: &Path, epochs: usize) -> Result<TrainingReport> {
    info!("Scanning: {}", dataset_path.display());
    let workspaces = collect_workspaces(dataset_path)?;
    if workspaces.is_empty() {
        bail!("No workspaces with static/uri_scan.json found in {}", dataset_path.display());
    }
    info!("Found {} workspaces", workspaces.len());

    let mut samples: Vec<Sample> = Vec::new();
    for ws in &workspaces {
        match extract_sample(ws) {
            Ok(s)  => samples.push(s),
            Err(e) => warn!("Skip {}: {}", ws.display(), e),
        }
    }
    info!("Valid samples: {}", samples.len());
    if samples.is_empty() { bail!("No valid samples — run gama-intel on APKs first"); }

    let mut model = FusionModel::new_random();
    let lr = 1e-3f32;
    let mut final_loss = 0.0f32;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        for sample in &samples {
            let target = sample.label.unwrap_or(0.3);
            let score = model.score(&sample.static_vec, &sample.smali_emb, &sample.net_seq)?;
            let pred  = score.anomaly_score / 100.0;
            let loss  = (pred - target).powi(2);
            epoch_loss += loss;

            // Simplified gradient descent on bias of output layer
            // (full backprop would require tracking activations — out of scope for v0.1)
            let grad = 2.0 * (pred - target) * pred * (1.0 - pred);
            if let Some(b) = model.fusion_out.b.first_mut() {
                *b -= lr * grad;
            }
        }
        final_loss = epoch_loss / samples.len() as f32;
        if epoch % 10 == 0 || epoch == epochs - 1 {
            info!("Epoch {}/{} — loss: {:.4}", epoch + 1, epochs, final_loss);
        }
    }

    if let Some(p) = output_path.parent() { std::fs::create_dir_all(p)?; }
    model.save(output_path)?;

    Ok(TrainingReport {
        epochs,
        final_loss,
        samples:    samples.len(),
        model_path: output_path.to_string_lossy().to_string(),
    })
}

fn extract_sample(ws: &Path) -> Result<Sample> {
    let static_vec = features::static_features::extract(ws)?;
    let smali_emb  = features::smali_features::embed(ws)?;
    let net_seq    = features::network_features::build_sequences(ws)?;
    let label_path = ws.join("deep").join("label.json");
    let label = if label_path.exists() {
        let l: WorkspaceLabel = serde_json::from_str(&std::fs::read_to_string(&label_path)?)?;
        Some(match l.class.as_str() { "A" => 0.1, "B" => 0.35, "C" => 0.7, "D" => 0.95, _ => 0.3 })
    } else { None };
    Ok(Sample { static_vec: static_vec.values, smali_emb, net_seq, label })
}

fn collect_workspaces(path: &Path) -> Result<Vec<PathBuf>> {
    if !path.exists() { bail!("Dataset path not found: {}", path.display()); }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let p = entry?.path();
        if p.is_dir() && p.join("static").join("uri_scan.json").exists() {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}
