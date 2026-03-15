use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "gama-deep", about = "GAMA-Deep — ML behavioural analysis module", version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    Analyse {
        workspace: PathBuf,
        #[arg(long, default_value = "~/.gama/models/gama_deep_v1.json")]
        model: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value = "40")]
        threshold: f32,
    },
    Train {
        dataset: PathBuf,
        #[arg(long, default_value = "~/.gama/models/gama_deep_v1.json")]
        output: PathBuf,
        #[arg(long, default_value = "50")]
        epochs: usize,
    },
    Features {
        workspace: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(level))
        .with_target(false).init();

    match cli.command {
        Commands::Analyse { workspace, model, output, threshold } => {
            let ws = expand_tilde(&workspace);
            let mp = expand_tilde(&model);
            info!("Analysing: {}", ws.display());
            let findings = gama_deep_lib::analyse_workspace(&ws, &mp)?;
            let relevant: Vec<_> = findings.iter().filter(|f| f.score.anomaly_score >= threshold).collect();
            info!("{}/{} findings above threshold {}", relevant.len(), findings.len(), threshold);
            let out = output.unwrap_or_else(|| ws.join("deep").join("gama_deep.json"));
            std::fs::create_dir_all(out.parent().unwrap())?;
            std::fs::write(&out, serde_json::to_string_pretty(&findings)?)?;
            info!("Output: {}", out.display());
            println!("{}", serde_json::to_string(&findings)?);
        }
        Commands::Train { dataset, output, epochs } => {
            let ds  = expand_tilde(&dataset);
            let out = expand_tilde(&output);
            info!("Training on: {}", ds.display());
            let report = gama_deep_lib::train(&ds, &out, epochs)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::Features { workspace } => {
            let ws  = expand_tilde(&workspace);
            let vec = gama_deep_lib::features::static_features::extract(&ws)?;
            println!("{}", serde_json::to_string_pretty(&vec)?);
        }
    }
    Ok(())
}

fn expand_tilde(path: &PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with("~/") {
        if let Some(home) = dirs::home_dir() { return home.join(&s[2..]); }
    }
    path.clone()
}
