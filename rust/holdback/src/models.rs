//! The two detectors' models: which one runs, where its files are, and the download from Hugging Face (or the
//! mirror in HF_ENDPOINT) at a pinned revision, each file checked against its SHA-256.
use crate::detect::{Detector, Filter};
use crate::student::Student;
use anyhow::{Context, Result, bail};
use aws_lc_rs::digest;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;

pub struct Model {
    /// The value of HOLDBACK_DETECTOR that picks it, also shown on the status page.
    pub name: &'static str,
    pub title: &'static str,
    repo: &'static str,
    revision: &'static str,
    files: &'static [(&'static str, &'static str)],
    /// The variable that names the user's own copy, which holdback reads and never writes to.
    pub dir_var: &'static str,
    /// Its download directory under ~/.cache/holdback.
    cache: &'static str,
    pub size: &'static str,
    loader: fn(&Path) -> Result<Detector>,
}

pub static DISTILLED: Model = Model {
    name: "distilled",
    title: "holdback's distilled privacy model",
    repo: "limbo23/holdback-student",
    revision: "8f2b162c59dd4835557bbdcb995fd1dd689c4a76",
    files: &[
        ("config.json", "1d8a7249b8e46ae58014ffee07e1d913cb33e3a5faa067aeee2e6330b56f15b2"),
        ("model.safetensors", "15c2bede412c9a6a19c808803d3d4afd8ba020270400c3a323bf0b30dc6115e3"),
        ("tokenizer.json", "d429409f3a2f493760bcc1c1398e1c4b32bbf3807b7fde2c6f2decca1a6ee2cc"),
    ],
    dir_var: "HOLDBACK_STUDENT_DIR",
    cache: "holdback-student",
    size: "about 115 MB",
    loader: |dir| Ok(Detector::Student(Box::new(Student::load(dir)?))),
};

pub static PRIVACY_FILTER: Model = Model {
    name: "privacy-filter",
    title: "OpenAI Privacy Filter (q4)",
    repo: "openai/privacy-filter",
    revision: "7ffa9a043d54d1be65afb281eddf0ffbe629385b",
    files: &[
        ("config.json", "b2b26a4a4a000639ad30b0c264adbefe365bdb567fbd7bb27303b8c438375bd1"),
        ("onnx/model_q4.onnx", "8f7dee8b46d096f052b359375dfba5d983cc4d18c44a783bf548615c472f8dea"),
        ("onnx/model_q4.onnx_data", "f30998e28c71c5374cc7e8b7de8f0f83e981592c0c2d652d2ad4928454dbb496"),
    ],
    dir_var: "HOLDBACK_MODEL_DIR",
    cache: "privacy-filter-q4",
    size: "about 0.9 GB",
    loader: |dir| Ok(Detector::Filter(Box::new(Filter::load(dir)?))),
};

/// The file in the cache that names the revision its files were downloaded or checked for.
const MARKER: &str = ".revision";

/// The model a value of HOLDBACK_DETECTOR picks; empty means the default.
pub fn named(value: &str) -> Result<&'static Model> {
    match value {
        "" | "distilled" => Ok(&DISTILLED),
        "privacy-filter" => Ok(&PRIVACY_FILTER),
        other => bail!("HOLDBACK_DETECTOR is {other:?}; use distilled or privacy-filter"),
    }
}

impl Model {
    fn own_dir(&self) -> Option<PathBuf> {
        std::env::var_os(self.dir_var).filter(|d| !d.is_empty()).map(PathBuf::from)
    }

    pub fn cache_dir(&self) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        Path::new(&home).join(".cache").join("holdback").join(self.cache)
    }

    /// The directory it runs from: the user's own copy when its variable is set, the cache otherwise.
    pub fn dir(&self) -> PathBuf {
        self.own_dir().unwrap_or_else(|| self.cache_dir())
    }

    /// Its files that `dir` lacks.
    pub fn absent(&self, dir: &Path) -> Vec<&'static str> {
        self.files.iter().map(|&(f, _)| f).filter(|f| !dir.join(f).exists()).collect()
    }

    /// Whether the cache holds every file of the pinned revision.
    pub fn cached(&self) -> bool {
        let dir = self.cache_dir();
        std::fs::read_to_string(dir.join(MARKER)).is_ok_and(|r| r == self.revision) && self.absent(&dir).is_empty()
    }

    /// Whether every file in `dir` has its pinned SHA-256.
    pub fn pinned(&self, dir: &Path) -> bool {
        self.files.iter().all(|&(f, want)| sha256(&dir.join(f)).is_ok_and(|h| h == want))
    }

    pub fn open(&self, dir: &Path) -> Result<Detector> {
        (self.loader)(dir).with_context(|| format!("could not load {} from {}", self.title, dir.display()))
    }

    /// Fills the cache with the pinned revision. A file already there stays when its SHA-256 matches, so a cache
    /// from an earlier version is checked once instead of downloaded again.
    async fn download(&self) -> Result<()> {
        let dir = self.cache_dir();
        // no marker while the files change, so that every version checks an interrupted download again
        let _ = std::fs::remove_file(dir.join(MARKER));
        remove_stale_parts(&dir, self.files);
        let host = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
        let client = reqwest::Client::builder().connect_timeout(Duration::from_secs(30)).read_timeout(Duration::from_secs(60)).build()?;
        let mut announced = false;
        for &(file, want) in self.files {
            let target = dir.join(file);
            if target.exists() && sha256(&target)? == want {
                continue;
            }
            if !announced {
                eprintln!("Downloading {} ({}) to {}", self.title, self.size, dir.display());
                announced = true;
            }
            std::fs::create_dir_all(target.parent().unwrap())?;
            let url = format!("{}/{}/resolve/{}/{file}", host.trim_end_matches('/'), self.repo, self.revision);
            // a part file of this process's own, so two starts at once cannot mix their bytes
            let part = target.with_file_name(format!("{}.{}.part", target.file_name().unwrap().to_string_lossy(), std::process::id()));
            if let Err(e) = fetch(&client, &url, &part, file, want).await {
                let _ = std::fs::remove_file(&part);
                return Err(e);
            }
            std::fs::rename(&part, &target)?;
        }
        std::fs::write(dir.join(MARKER), self.revision)?;
        Ok(())
    }
}

/// The detector HOLDBACK_DETECTOR picks, ready to run. The user's own copy must be complete; the cache is
/// downloaded or completed first.
pub async fn load() -> Result<Detector> {
    let model = named(&std::env::var("HOLDBACK_DETECTOR").unwrap_or_default())?;
    let dir = model.dir();
    if model.own_dir().is_some() {
        let absent = model.absent(&dir);
        if !absent.is_empty() {
            bail!("{} is {}, which has no {}", model.dir_var, dir.display(), absent.join(", "));
        }
    } else if !model.cached() {
        model.download().await?;
    }
    eprintln!("Loading {}...", model.title);
    let mut detector = model.open(&dir)?;
    detector.detect("Warm-up: Alice Example, alice@example.com")?;
    Ok(detector)
}

/// Removes the part files of downloads that stopped, such as one ended by Ctrl-C. A running download writes at least
/// once a minute, or its read times out, so a part file idle for ten minutes has no writer.
fn remove_stale_parts(dir: &Path, files: &[(&str, &str)]) {
    let mut folders: Vec<PathBuf> = files.iter().filter_map(|(f, _)| dir.join(f).parent().map(Path::to_path_buf)).collect();
    folders.dedup();
    for entry in folders.iter().filter_map(|d| std::fs::read_dir(d).ok()).flatten().flatten() {
        let idle = entry.metadata().and_then(|m| m.modified()).is_ok_and(|t| t.elapsed().is_ok_and(|e| e > Duration::from_secs(600)));
        if idle && entry.file_name().to_string_lossy().ends_with(".part") {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Streams `url` into `part` and fails unless the bytes have the SHA-256 `want`.
async fn fetch(client: &reqwest::Client, url: &str, part: &Path, file: &str, want: &str) -> Result<()> {
    let failed = || format!("downloading {url}");
    let mut resp = client.get(url).send().await.and_then(reqwest::Response::error_for_status).with_context(failed)?;
    let total = resp.content_length().unwrap_or(0);
    let mut out = tokio::fs::File::create(part).await?;
    let mut hash = digest::Context::new(&digest::SHA256);
    let (mut got, mut shown) = (0u64, 0u64);
    while let Some(chunk) = resp.chunk().await.with_context(failed)? {
        out.write_all(&chunk).await?;
        hash.update(&chunk);
        got += chunk.len() as u64;
        if total > 50_000_000 && got * 20 / total > shown {
            shown = got * 20 / total;
            eprintln!("  {file}: {} of {} MB", got / 1_000_000, total / 1_000_000);
        }
    }
    out.flush().await?;
    if hex(hash.finish()) != want {
        bail!("{url} does not match its pinned SHA-256 ({got} bytes); run again to retry");
    }
    Ok(())
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = digest::Context::new(&digest::SHA256);
    let mut buf = vec![0; 1 << 20];
    loop {
        match std::io::Read::read(&mut file, &mut buf)? {
            0 => return Ok(hex(hash.finish())),
            n => hash.update(&buf[..n]),
        }
    }
}

fn hex(digest: digest::Digest) -> String {
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}
