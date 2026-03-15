// GAMA-Deep — Fusion Model (pure ndarray, no candle/CUDA)

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::features::static_features::STATIC_DIM;
use crate::features::smali_features::SMALI_EMB_DIM;
use crate::features::network_features::NetworkSequence;
use crate::scoring::AnomalyScore;

pub const NET_FLAT_DIM: usize = 256;
pub const CHANNEL_DIM:  usize = 64;
pub const FUSION_DIM:   usize = CHANNEL_DIM * 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

impl Linear {
    pub fn new_random(in_dim: usize, out_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::rng();
        let scale = (2.0_f32 / in_dim as f32).sqrt();
        let w = (0..out_dim)
            .map(|_| (0..in_dim).map(|_| rng.random::<f32>() * 2.0 * scale - scale).collect())
            .collect();
        Self { w, b: vec![0.0f32; out_dim] }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        self.w.iter().zip(self.b.iter()).map(|(row, bias)| {
            row.iter().zip(x.iter()).map(|(w, xi)| w * xi).sum::<f32>() + bias
        }).collect()
    }
}

fn relu(v: Vec<f32>) -> Vec<f32> { v.into_iter().map(|x| x.max(0.0)).collect() }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
fn l2(v: &[f32]) -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionModel {
    static_l1:  Linear,
    static_l2:  Linear,
    smali_proj: Linear,
    net_proj:   Linear,
    fusion_l1:  Linear,
    fusion_l2:  Linear,
    pub fusion_out: Linear,
}

impl FusionModel {
    pub fn new_random() -> Self {
        Self {
            static_l1:  Linear::new_random(STATIC_DIM,   CHANNEL_DIM),
            static_l2:  Linear::new_random(CHANNEL_DIM,  CHANNEL_DIM),
            smali_proj: Linear::new_random(SMALI_EMB_DIM, CHANNEL_DIM),
            net_proj:   Linear::new_random(NET_FLAT_DIM,  CHANNEL_DIM),
            fusion_l1:  Linear::new_random(FUSION_DIM,   96),
            fusion_l2:  Linear::new_random(96,            48),
            fusion_out: Linear::new_random(48,             1),
        }
    }

    pub fn score(&self, static_vec: &[f32], smali_emb: &[f32], net_seq: &NetworkSequence) -> Result<AnomalyScore> {
        let s_out  = relu(self.static_l2.forward(&relu(self.static_l1.forward(static_vec))));
        let sm_out = relu(self.smali_proj.forward(smali_emb));
        let n_out  = relu(self.net_proj.forward(&flatten_network(net_seq)));

        let fused: Vec<f32> = s_out.iter().chain(sm_out.iter()).chain(n_out.iter()).copied().collect();
        let f1  = relu(self.fusion_l1.forward(&fused));
        let f2  = relu(self.fusion_l2.forward(&f1));
        let out = self.fusion_out.forward(&f2);
        let score = sigmoid(out[0]) * 100.0;

        let (sn, smn, nn) = (l2(&s_out), l2(&sm_out), l2(&n_out));
        let total = sn + smn + nn + 1e-8;

        Ok(AnomalyScore {
            anomaly_score: score,
            static_contribution: sn / total,
            smali_contribution: smn / total,
            network_contribution: nn / total,
            confidence: if net_seq.events.is_empty() { 0.4 } else { 0.7 },
            top_signals: vec![],
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            tracing::info!("Loading model: {}", path.display());
            Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
        } else {
            tracing::warn!("Model not found at {} — random weights. Run: gama-deep train", path.display());
            Ok(Self::new_random())
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(p) = path.parent() { std::fs::create_dir_all(p)?; }
        std::fs::write(path, serde_json::to_string(self)?)?;
        tracing::info!("Model saved: {}", path.display());
        Ok(())
    }
}

fn flatten_network(seq: &NetworkSequence) -> Vec<f32> {
    use crate::features::network_features::NET_EVENT_DIM;
    const SLOTS: usize = NET_FLAT_DIM / NET_EVENT_DIM;
    let mut flat = vec![0.0f32; NET_FLAT_DIM];
    for (i, ev) in seq.events.iter().take(SLOTS).enumerate() {
        let base = i * NET_EVENT_DIM;
        for (j, &v) in ev.features.iter().enumerate().take(NET_EVENT_DIM) {
            flat[base + j] = v;
        }
    }
    flat
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingReport {
    pub epochs:     usize,
    pub final_loss: f32,
    pub samples:    usize,
    pub model_path: String,
}
