// GAMA-Deep — Channel 1: Static Feature Extractor
//
// Reads GAMA-Intel workspace JSON files and builds a normalised
// dense feature vector for the fusion model.
//
// Feature dimensions:
//   [0..22]   URI scheme scores (top-22 suspicious schemes, normalised 0-1)
//   [23]      URI scheme count (normalised)
//   [24]      Encoded URI count (normalised)
//   [25..64]  SDK one-hot bitmap (40 known SDKs)
//   [65]      SDK high-risk count (normalised)
//   [66]      SDK advertising count (normalised)
//   [67..96]  Permission risk bitmap (30 dangerous permissions)
//   [97]      Dangerous permission count (normalised)
//   [98]      Native lib count (normalised)
//   [99]      Max native lib entropy (normalised 0-1)
//   [100]     APK size MB (log-normalised)
//   [101]     Size delta ratio (declared vs actual)
//   [102]     Exported component count (normalised)
//   [103]     Custom manifest scheme count (normalised)
//   [104..127] Reserved / padding
// Total: 128 dimensions

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const STATIC_DIM: usize = 128;

/// Ordered list of known SDKs — position determines bitmap index.
pub const SDK_CATALOGUE: &[&str] = &[
    "Mintegral / MBridge", "Unity Ads", "Adjust", "AppsFlyer",
    "Firebase / Google Analytics", "Moloco", "IronSource", "AppLovin",
    "TikTok / Pangle", "Chartboost", "AdColony", "Vungle / Liftoff",
    "InMobi", "Amplitude", "Mixpanel", "Branch",
    "OneSignal", "Tapjoy", "Crashlytics / Firebase Crashlytics", "Sentry",
    "Yandex Metrica", "MoPub", "Facebook Audience Network", "Twitter MoPub",
    "Fyber / Digital Turbine", "Digital Turbine", "Snap Audience Network",
    "Pinterest Ads", "Reddit Ads", "TradePlus",
    "Ogury", "Madvertise", "Smaato", "Rubicon",
    "OpenX", "Index Exchange", "PubMatic", "Criteo",
    "Amazon Publisher Services", "Google AdMob",
];

/// Dangerous permissions — ordered list for bitmap encoding
pub const DANGEROUS_PERMISSIONS: &[&str] = &[
    "READ_CONTACTS", "WRITE_CONTACTS", "READ_PHONE_STATE",
    "READ_PHONE_NUMBERS", "CALL_PHONE", "READ_CALL_LOG",
    "WRITE_CALL_LOG", "ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION",
    "ACCESS_BACKGROUND_LOCATION", "RECORD_AUDIO", "CAMERA",
    "READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE",
    "READ_MEDIA_IMAGES", "READ_MEDIA_VIDEO", "GET_ACCOUNTS",
    "USE_BIOMETRIC", "USE_FINGERPRINT", "QUERY_ALL_PACKAGES",
    "PACKAGE_USAGE_STATS", "SYSTEM_ALERT_WINDOW", "WRITE_SETTINGS",
    "REQUEST_INSTALL_PACKAGES", "BIND_ACCESSIBILITY_SERVICE",
    "MANAGE_EXTERNAL_STORAGE", "READ_PRECISE_READ_PHONE_STATE",
    "ACTIVITY_RECOGNITION", "BODY_SENSORS", "SEND_SMS",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticFeatureVector {
    pub values:   Vec<f32>,       // 128-dim normalised vector
    pub metadata: FeatureMeta,    // human-readable breakdown
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMeta {
    pub top_schemes:     Vec<(String, f32)>,  // scheme, normalised score
    pub sdks_detected:   Vec<String>,
    pub permissions_hit: Vec<String>,
    pub apk_size_mb:     f32,
    pub native_lib_count: usize,
}

pub fn extract(workspace_path: &Path) -> Result<StaticFeatureVector> {
    let static_dir = workspace_path.join("static");
    let mut v = vec![0.0f32; STATIC_DIM];
    let mut meta = FeatureMeta {
        top_schemes: vec![], sdks_detected: vec![],
        permissions_hit: vec![], apk_size_mb: 0.0,
        native_lib_count: 0,
    };

    // ── URI scan features [0..24] ─────────────────────────────
    let uri_path = static_dir.join("uri_scan.json");
    if uri_path.exists() {
        let uri: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&uri_path)?
        )?;

        if let Some(suspicious) = uri["suspicious"].as_object() {
            // Sort by score descending, take top 22
            let mut schemes: Vec<(String, f32)> = suspicious.iter()
                .filter_map(|(k, v)| {
                    v["score"].as_f64().map(|s| (k.clone(), s as f32))
                })
                .collect();
            schemes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (i, (scheme, score)) in schemes.iter().take(22).enumerate() {
                v[i] = (score / 15.0).min(1.0);  // normalise 0-15 → 0-1
                meta.top_schemes.push((scheme.clone(), *score));
            }

            let total_suspicious = suspicious.len() as f32;
            v[23] = (total_suspicious / 50.0).min(1.0);
        }

        // Encoded URI count
        if let Some(encoded) = uri["encoded_hits"].as_array() {
            v[24] = (encoded.len() as f32 / 20.0).min(1.0);
        }
    }

    // ── SDK bitmap [25..64] ──────────────────────────────────
    let sdk_path = static_dir.join("sdk_map.json");
    if sdk_path.exists() {
        let sdk: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&sdk_path)?
        )?;

        let mut high_risk = 0.0f32;
        let mut ad_count  = 0.0f32;

        if let Some(sdks) = sdk["sdks"].as_array() {
            for sdk_entry in sdks {
                let name = sdk_entry["name"].as_str().unwrap_or("");
                // Find position in catalogue
                if let Some(idx) = SDK_CATALOGUE.iter().position(|&s| s == name) {
                    v[25 + idx] = 1.0;
                    meta.sdks_detected.push(name.to_string());
                }
                if sdk_entry["risk"].as_str() == Some("high") { high_risk += 1.0; }
                if sdk_entry["category"].as_str() == Some("advertising") { ad_count += 1.0; }
            }
        }
        v[65] = (high_risk / 5.0).min(1.0);
        v[66] = (ad_count  / 10.0).min(1.0);
    }

    // ── Permission bitmap [67..96] ───────────────────────────
    let manifest_path = static_dir.join("manifest.json");
    if manifest_path.exists() {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&manifest_path)?
        )?;

        let mut dangerous_count = 0.0f32;
        if let Some(dangerous) = manifest["dangerous_permissions"].as_array() {
            for perm in dangerous {
                let name = perm.as_str().unwrap_or("")
                    .replace("android.permission.", "");
                if let Some(idx) = DANGEROUS_PERMISSIONS.iter()
                    .position(|&p| p == name)
                {
                    v[67 + idx] = 1.0;
                    meta.permissions_hit.push(name.clone());
                }
                dangerous_count += 1.0;
            }
        }
        v[97] = (dangerous_count / 15.0).min(1.0);

        // Exported components
        if let Some(exported) = manifest["exported_components"].as_array() {
            v[102] = (exported.len() as f32 / 10.0).min(1.0);
        }

        // Custom manifest schemes
        if let Some(schemes) = manifest["custom_schemes"].as_array() {
            v[103] = (schemes.len() as f32 / 10.0).min(1.0);
        }
    }

    // ── Native libs [98..101] ────────────────────────────────
    let native_path = static_dir.join("native_libs.json");
    if native_path.exists() {
        let native: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&native_path)?
        )?;

        let total = native["total"].as_f64().unwrap_or(0.0) as f32;
        v[98] = (total / 10.0).min(1.0);
        meta.native_lib_count = total as usize;

        // Max entropy across all libs
        if let Some(libs) = native["libs"].as_array() {
            let max_entropy = libs.iter()
                .filter_map(|l| l["entropy"].as_f64())
                .fold(0.0f64, f64::max);
            v[99] = (max_entropy / 8.0) as f32;  // max entropy is 8 bits
        }
    }

    // ── APK metadata [100..101] ──────────────────────────────
    let intake_path = workspace_path.join("intake.json");
    if intake_path.exists() {
        let intake: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&intake_path)?
        )?;

        if let Some(size) = intake["apk_size_mb"].as_f64() {
            // Log-normalise: log2(size)/10 gives ~0.7 for 100MB
            v[100] = ((size as f32).ln() / 10.0_f32.ln()).min(1.0).max(0.0);
            meta.apk_size_mb = size as f32;
        }
    }

    Ok(StaticFeatureVector { values: v, metadata: meta })
}
