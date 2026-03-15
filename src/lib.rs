// GAMA-Deep — ML analysis module
// Pure Rust MLP with ndarray — no CUDA/candle dependency.

pub mod features;
pub mod model;
pub mod scoring;

pub use features::static_features::StaticFeatureVector;
pub use features::smali_features::SmaliTokeniser;
pub use features::network_features::NetworkSequence;
pub use model::fusion::FusionModel;
pub use scoring::{AnomalyScore, DeepFinding};

use anyhow::Result;
use std::path::Path;

pub fn analyse_workspace(
    workspace_path: &Path,
    model_path:     &Path,
) -> Result<Vec<DeepFinding>> {
    let static_vec = features::static_features::extract(workspace_path)?;
    let smali_emb  = features::smali_features::embed(workspace_path)?;
    let net_seq    = features::network_features::build_sequences(workspace_path)?;
    let model      = FusionModel::load(model_path)?;
    let score      = model.score(&static_vec.values, &smali_emb, &net_seq)?;
    Ok(scoring::to_findings(score, workspace_path))
}

pub fn train(
    dataset_path: &Path,
    output_path:  &Path,
    epochs:       usize,
) -> Result<model::TrainingReport> {
    model::trainer::train(dataset_path, output_path, epochs)
}
