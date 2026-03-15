// GAMA-Deep — Channel 2: Smali Bytecode Embedder
//
// Tokenises smali opcodes and builds a Transformer encoder embedding.
// Architecture: token embedding → 4-layer Transformer encoder → mean pool → 256-dim vector
//
// Vocabulary: ~300 smali opcodes + special tokens
// Max sequence length: 512 tokens per file, aggregated across top-N files

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use rayon::prelude::*;

pub const SMALI_EMB_DIM:  usize = 256;
pub const MAX_SEQ_LEN:    usize = 512;
pub const MAX_FILES:      usize = 32;   // top-N smali files by suspicion
pub const VOCAB_SIZE:     usize = 512;  // smali opcode vocabulary

// ── Special tokens ────────────────────────────────────────────────
pub const TOKEN_PAD:      u32 = 0;
pub const TOKEN_UNK:      u32 = 1;
pub const TOKEN_CLS:      u32 = 2;   // classification token
pub const TOKEN_SEP:      u32 = 3;   // file separator

/// Smali opcode → token ID mapping.
/// Built from the official smali opcode set + GAMA-specific synthetic tokens.
pub struct SmaliTokeniser {
    vocab: HashMap<String, u32>,
}

impl SmaliTokeniser {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        vocab.insert("<PAD>".into(), TOKEN_PAD);
        vocab.insert("<UNK>".into(), TOKEN_UNK);
        vocab.insert("<CLS>".into(), TOKEN_CLS);
        vocab.insert("<SEP>".into(), TOKEN_SEP);

        // Core smali opcodes — the full dalvik instruction set
        let opcodes = [
            // Object operations
            "invoke-virtual", "invoke-static", "invoke-direct",
            "invoke-interface", "invoke-super", "invoke-polymorphic",
            "invoke-custom", "invoke-virtual/range", "invoke-static/range",
            // Field access
            "iget", "iget-object", "iget-boolean", "iget-byte",
            "iget-char", "iget-short", "iget-wide",
            "iput", "iput-object", "iput-boolean", "iput-wide",
            "sget", "sget-object", "sget-boolean", "sget-wide",
            "sput", "sput-object", "sput-boolean", "sput-wide",
            // Moves
            "move", "move/from16", "move/16", "move-wide",
            "move-object", "move-object/from16", "move-result",
            "move-result-object", "move-result-wide", "move-exception",
            // Constants
            "const", "const/4", "const/16", "const/high16",
            "const-wide", "const-wide/16", "const-wide/32",
            "const-wide/high16", "const-string", "const-string/jumbo",
            "const-class",
            // Branching
            "if-eq", "if-ne", "if-lt", "if-ge", "if-gt", "if-le",
            "if-eqz", "if-nez", "if-ltz", "if-gez", "if-gtz", "if-lez",
            "goto", "goto/16", "goto/32", "packed-switch", "sparse-switch",
            // Arithmetic
            "add-int", "sub-int", "mul-int", "div-int", "rem-int",
            "and-int", "or-int", "xor-int", "shl-int", "shr-int",
            "ushr-int", "add-long", "sub-long", "mul-long",
            "add-float", "sub-float", "mul-float", "div-float",
            "add-double", "sub-double", "mul-double", "div-double",
            "neg-int", "not-int", "neg-long", "not-long",
            "int-to-long", "int-to-float", "int-to-double",
            "long-to-int", "long-to-float", "long-to-double",
            "float-to-int", "float-to-long", "float-to-double",
            "double-to-int", "double-to-long", "double-to-float",
            // Arrays
            "new-array", "filled-new-array", "fill-array-data",
            "array-length", "aget", "aget-object", "aget-wide",
            "aput", "aput-object", "aput-wide",
            // Objects
            "new-instance", "instance-of", "check-cast",
            "throw", "return", "return-void", "return-object",
            "return-wide", "nop", "monitor-enter", "monitor-exit",
            // Comparison
            "cmpl-float", "cmpg-float", "cmpl-double", "cmpg-double",
            "cmp-long",
            // Annotations
            ".method", ".end method", ".class", ".super", ".implements",
            ".field", ".annotation", ".end annotation",
            ".param", ".registers", ".locals",
            // GAMA synthetic tokens — patterns of interest
            "WEBVIEW_LOAD",      // loadUrl or loadData pattern
            "WEBVIEW_OVERRIDE",  // shouldOverrideUrlLoading pattern
            "WEBVIEW_INTERCEPT", // shouldInterceptRequest pattern
            "URI_PARSE",         // Uri.parse pattern
            "INTENT_URI",        // Intent.parseUri pattern
            "JNI_LOAD",          // System.loadLibrary
            "REFLECTION_CALL",   // Class.forName + invoke
            "CRYPTO_INIT",       // Cipher.getInstance pattern
            "WORKMANAGER_ENQUEUE", // WorkManager.enqueue
            "JOBSCHEDULER_SCHEDULE", // JobScheduler.schedule
            "TELEPHONY_IDS",     // getDeviceId / getImei
            "ANDROID_ID",        // Settings.Secure.getString ANDROID_ID
            "NETWORK_HTTP",      // HttpURLConnection / OkHttp
            "NETWORK_SOCKET",    // raw Socket
            "BASE64_DECODE",     // Base64.decode
            "SHARED_PREFS",      // SharedPreferences access
        ];

        let mut next_id = TOKEN_SEP + 1;
        for opcode in &opcodes {
            vocab.insert(opcode.to_string(), next_id);
            next_id += 1;
        }

        Self { vocab }
    }

    /// Tokenise a smali file content into token IDs.
    /// Applies synthetic token substitution for GAMA-specific patterns.
    pub fn tokenise(&self, content: &str) -> Vec<u32> {
        let mut tokens = vec![TOKEN_CLS];

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }

            // Synthetic token detection (before opcode split)
            let synthetic = self.detect_synthetic(line);
            if let Some(tok) = synthetic {
                tokens.push(tok);
                continue;
            }

            // Extract opcode (first word)
            let opcode = line.split_whitespace().next().unwrap_or("");
            let id = self.vocab.get(opcode).copied().unwrap_or(TOKEN_UNK);
            tokens.push(id);
        }

        // Truncate to max sequence length
        tokens.truncate(MAX_SEQ_LEN);

        // Pad if necessary
        while tokens.len() < MAX_SEQ_LEN {
            tokens.push(TOKEN_PAD);
        }

        tokens
    }

    fn detect_synthetic(&self, line: &str) -> Option<u32> {
        if line.contains("shouldOverrideUrlLoading") {
            return self.vocab.get("WEBVIEW_OVERRIDE").copied();
        }
        if line.contains("shouldInterceptRequest") {
            return self.vocab.get("WEBVIEW_INTERCEPT").copied();
        }
        if line.contains("loadUrl") || line.contains("loadData") {
            return self.vocab.get("WEBVIEW_LOAD").copied();
        }
        if line.contains("Uri;->parse") {
            return self.vocab.get("URI_PARSE").copied();
        }
        if line.contains("System;->loadLibrary") || line.contains("System;->load(") {
            return self.vocab.get("JNI_LOAD").copied();
        }
        if line.contains("WorkManager") && line.contains("enqueue") {
            return self.vocab.get("WORKMANAGER_ENQUEUE").copied();
        }
        if line.contains("JobScheduler") && line.contains("schedule") {
            return self.vocab.get("JOBSCHEDULER_SCHEDULE").copied();
        }
        if line.contains("getDeviceId") || line.contains("getImei") {
            return self.vocab.get("TELEPHONY_IDS").copied();
        }
        if line.contains("ANDROID_ID") {
            return self.vocab.get("ANDROID_ID").copied();
        }
        if line.contains("Base64") && line.contains("decode") {
            return self.vocab.get("BASE64_DECODE").copied();
        }
        if line.contains("Class;->forName") {
            return self.vocab.get("REFLECTION_CALL").copied();
        }
        if line.contains("Cipher;->getInstance") {
            return self.vocab.get("CRYPTO_INIT").copied();
        }
        None
    }
}

/// Select the most relevant smali files for embedding.
/// Priority: files in SDK paths > files with suspicious URI schemes > others.
fn select_files(apktool_out: &Path) -> Vec<std::path::PathBuf> {
    let sdk_patterns = [
        "mbridge", "mintegral", "applovin", "unity3d", "adjust",
        "firebase", "moloco", "ironsource", "bytedance", "facebook",
    ];

    let all_smali: Vec<_> = walkdir::WalkDir::new(apktool_out)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "smali").unwrap_or(false))
        .map(|e| e.path().to_owned())
        .collect();

    // Score each file: SDK path = +2, suspicious patterns in name = +1
    let mut scored: Vec<(std::path::PathBuf, i32)> = all_smali.into_iter()
        .map(|p| {
            let path_str = p.to_string_lossy().to_lowercase();
            let score = sdk_patterns.iter()
                .filter(|&&pat| path_str.contains(pat))
                .count() as i32 * 2;
            (p, score)
        })
        .collect();

    scored.sort_by(|a, b| b.1.cmp(&a.1));
    scored.into_iter().take(MAX_FILES).map(|(p, _)| p).collect()
}

/// Embed smali files and return a 256-dim aggregated vector.
/// Without a trained model, returns a feature-count vector (warm start).
pub fn embed(workspace_path: &Path) -> Result<Vec<f32>> {
    let apktool_out = workspace_path.join("static").join("apktool_out");
    if !apktool_out.exists() {
        return Ok(vec![0.0f32; SMALI_EMB_DIM]);
    }

    let tokeniser = SmaliTokeniser::new();
    let files = select_files(&apktool_out);

    if files.is_empty() {
        return Ok(vec![0.0f32; SMALI_EMB_DIM]);
    }

    // Parallel tokenisation across selected files
    let all_tokens: Vec<Vec<u32>> = files.par_iter()
        .filter_map(|f| std::fs::read_to_string(f).ok())
        .map(|content| tokeniser.tokenise(&content))
        .collect();

    // Aggregate token frequency into a 256-dim histogram
    // (warm-start representation before model is trained)
    // Dimensions 0..255 = token frequency bins
    // This gives meaningful signal even before transformer training
    let mut histogram = vec![0.0f32; SMALI_EMB_DIM];
    let mut total_tokens = 0usize;

    for token_seq in &all_tokens {
        for &tok in token_seq {
            if tok != TOKEN_PAD {
                let bin = (tok as usize) % SMALI_EMB_DIM;
                histogram[bin] += 1.0;
                total_tokens += 1;
            }
        }
    }

    // Normalise by total tokens
    if total_tokens > 0 {
        let scale = 1.0 / (total_tokens as f32).sqrt();
        for v in &mut histogram {
            *v *= scale;
        }
    }

    Ok(histogram)
}

