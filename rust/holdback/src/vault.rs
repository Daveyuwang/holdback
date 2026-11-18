//! Detection plus a process-wide placeholder map.
//!
//! A value maps to the same placeholder for the life of the process, so an agent that resends its whole
//! history every turn sends byte-identical redacted history (prompt caching keeps working), and a value the
//! detector missed in one context is still replaced once it has been seen elsewhere.
use crate::detect::{Detector, Span};
use crate::precision::refine;
use anyhow::Result;
use fancy_regex::Regex;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::SystemTime;
use std::sync::LazyLock;

pub static PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[[A-Z][A-Z_]*_\d+\]").unwrap());
static WHOLE_PLACEHOLDER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\[[A-Z][A-Z_]*_\d+\]$").unwrap());
static TAIL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[[A-Z0-9_]{0,48}$").unwrap());
const CACHE_LIMIT: usize = 4096;

/// What `holdback review` shows for one replaced value.
struct Seen {
    detected: usize, // times the detector or a rule found it
    matched: usize,  // times it was replaced as an already known value
    texts: usize,    // distinct texts in which the detector or a rule found it
    context: String, // the text around its first sighting
}

/// The user's corrections, read from the file that `holdback keep` and `holdback replace` write.
#[derive(Default)]
struct Lists {
    path: PathBuf,
    read_at: Option<SystemTime>, // the file's modification time when it was last read
    keep: HashSet<String>,       // never replaced
    replace: HashSet<String>,    // always replaced, whatever the detector says
}

/// Where a known value is replaced when the detector misses it.
#[derive(Clone, Copy, PartialEq)]
pub enum Spread {
    /// Everywhere, from the first time it is found.
    All,
    /// As `All`, except a single word taken for a person, which waits until a second text confirms it.
    /// A tool's name or a common word tagged once would otherwise be replaced everywhere.
    Twice,
}

pub struct Vault {
    detector: Detector,
    to_ph: HashMap<String, String>,
    to_val: HashMap<String, String>,
    by_length: Vec<String>, // known values, longest first
    counter: usize,
    cache: HashMap<String, Vec<Span>>,
    seen: HashMap<String, Seen>,
    store: Option<PathBuf>, // where the map is saved, once hook mode has been used
    stored: usize,          // values in the saved file
    lists: Lists,
    /// Saved values that today's rules would not replace. Their placeholders still restore; they no longer spread.
    retired: HashSet<String>,
    pub spread: Spread,
    pub stats: HashMap<String, usize>,
}

impl Vault {
    pub fn new(detector: Detector) -> Self {
        // Numbers start above 100, so a placeholder written as an example in a document, such as
        // [PRIVATE_EMAIL_2], is not taken for a real one and replaced with a value.
        let micros = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_micros()).unwrap_or(0);
        let counter = 100 + micros as usize % 800;
        Vault { detector, to_ph: HashMap::new(), to_val: HashMap::new(), by_length: vec![], counter, cache: HashMap::new(), seen: HashMap::new(), store: None, stored: 0, lists: Lists::default(), retired: HashSet::new(), spread: Spread::All, stats: HashMap::new() }
    }

    pub fn distinct_values(&self) -> usize {
        self.to_ph.len()
    }

    /// Saved values that are no longer replaced because today's rules would not replace them.
    pub fn retired(&self) -> usize {
        self.retired.len()
    }

    /// Every replaced value with its placeholder, counts and first context, most replaced first.
    /// These are the real values: only `holdback review` may show them.
    pub fn review(&self) -> Vec<Value> {
        let mut rows: Vec<_> = self.seen.iter().collect();
        rows.sort_by_key(|(v, s)| (std::cmp::Reverse(s.detected + s.matched), *v));
        rows.iter().map(|(v, s)| json!({"value": v, "placeholder": self.to_ph[*v], "detected": s.detected, "matched": s.matched, "context": s.context})).collect()
    }

    /// The refined spans of `text`, and whether this text is new to the cache.
    fn spans(&mut self, text: &str) -> Result<(Vec<Span>, bool)> {
        if let Some(hit) = self.cache.get(text) {
            return Ok((hit.clone(), false));
        }
        let spans = refine(text, &self.detector.detect(text)?);
        if self.cache.len() >= CACHE_LIMIT {
            self.cache.clear();
        }
        self.cache.insert(text.to_string(), spans.clone());
        Ok((spans, true))
    }

    fn placeholder(&mut self, value: &str, label: &str) -> String {
        if let Some(ph) = self.to_ph.get(value) {
            return ph.clone();
        }
        self.counter += 1;
        let ph = format!("[{}_{}]", label.to_uppercase(), self.counter);
        self.learn(value, &ph);
        ph
    }

    fn learn(&mut self, value: &str, ph: &str) {
        self.to_ph.insert(value.to_string(), ph.to_string());
        self.to_val.insert(ph.to_string(), value.to_string());
        let pos = self.by_length.partition_point(|v| v.chars().count() >= value.chars().count());
        self.by_length.insert(pos, value.to_string());
    }

    /// Keep the placeholder map in `path` from now on, loading what an earlier process saved there. Hook mode
    /// needs this: its placeholders stay in the conversation after this process ends. Without hook mode nothing
    /// is saved and the values stay in memory.
    pub fn keep_in(&mut self, path: PathBuf) -> Result<()> {
        if self.store.is_some() {
            return Ok(());
        }
        if let Ok(text) = std::fs::read_to_string(&path) {
            let saved: Value = serde_json::from_str(&text)?;
            for (ph, value) in saved["values"].as_object().into_iter().flatten() {
                if let Some(value) = value.as_str().filter(|v| !self.to_ph.contains_key(*v)) {
                    self.learn(value, ph);
                    // a rule added since the value was saved may say it was never private
                    let label = ph[1..ph.rfind('_').unwrap_or(1)].to_lowercase();
                    if refine(value, &[Span { start: 0, end: value.len(), label }]).is_empty() {
                        self.retired.insert(value.to_string());
                    }
                }
            }
            self.counter = self.counter.max(saved["counter"].as_u64().unwrap_or(0) as usize);
            self.stored = self.to_ph.len();
        }
        self.store = Some(path);
        Ok(())
    }

    /// In hook mode, append the shape of a new value and of the text around it to `sightings.log` beside the
    /// saved map. The contexts in memory are gone after a restart; this file is what remains for finding out
    /// why a value was replaced. It holds shapes, not values.
    fn log_sighting(&self, label: &str, value: &str, context: &str) {
        let Some(path) = self.store.as_ref().map(|p| p.with_file_name("sightings.log")) else { return };
        let _ = std::fs::create_dir_all(path.parent().unwrap());
        let mut options = std::fs::OpenOptions::new();
        options.append(true).create(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        if let Ok(mut file) = options.open(path) {
            let _ = std::io::Write::write_all(&mut file, format!("{}\n", json!({"label": label, "value": shape(value), "context": shape(context)})).as_bytes());
        }
    }

    /// Write the map to its file if it has grown. The file holds real values and only the user can read it.
    pub fn save(&mut self) -> Result<()> {
        let Some(path) = self.store.as_ref().filter(|_| self.stored != self.to_ph.len()) else { return Ok(()) };
        std::fs::create_dir_all(path.parent().unwrap())?;
        let part = path.with_extension("part");
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create(true).truncate(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        std::io::Write::write_all(&mut options.open(&part)?, json!({"counter": self.counter, "values": self.to_val}).to_string().as_bytes())?;
        std::fs::rename(&part, path)?;
        self.stored = self.to_ph.len();
        Ok(())
    }

    /// Take the user's corrections from `path`, now and whenever the file changes (see `refresh`).
    pub fn correct_from(&mut self, path: PathBuf) {
        self.lists.path = path;
        self.refresh();
    }

    /// Reread the corrections if their file changed. Called once per request, so a correction made while the
    /// process runs applies to the next request.
    pub fn refresh(&mut self) {
        let modified = std::fs::metadata(&self.lists.path).and_then(|m| m.modified()).ok();
        if modified == self.lists.read_at {
            return;
        }
        self.lists.read_at = modified;
        let file: Value = std::fs::read_to_string(&self.lists.path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null);
        let list = |key: &str| -> HashSet<String> { file[key].as_array().into_iter().flatten().filter_map(|v| v.as_str().map(String::from)).collect() };
        (self.lists.keep, self.lists.replace) = (list("keep"), list("replace"));
        for value in self.lists.replace.clone() {
            self.placeholder(&value, "private_value");
        }
    }

    /// How many private values `text` holds, without learning them. For text that cannot be redacted.
    pub fn count(&mut self, text: &str) -> Result<usize> {
        Ok(self.spans(text)?.0.iter().filter(|sp| !self.lists.keep.contains(&text[sp.start..sp.end])).count())
    }

    /// Whether a known value is replaced where the detector did not find it.
    fn spreads(&self, value: &str) -> bool {
        if self.lists.replace.contains(value) {
            return true;
        }
        if self.lists.keep.contains(value) || self.retired.contains(value) {
            return false;
        }
        let one_word_person = self.to_ph[value].starts_with("[PRIVATE_PERSON_") && !value.contains(char::is_whitespace);
        let confirmed = self.spread == Spread::All || !one_word_person || self.seen.get(value).is_some_and(|s| s.texts >= 2);
        value.chars().count() >= 4 && confirmed
    }

    /// The parts of `text` to replace, in order, each with its placeholder: what the detector and the rules
    /// found, then values known from elsewhere that the detector missed here. Placeholders already in the text
    /// are left alone.
    pub fn replacements(&mut self, text: &str) -> Result<Vec<(usize, usize, String)>> {
        let (spans, new_text) = self.spans(text)?;
        let mut out: Vec<(usize, usize, String)> = vec![];
        for sp in spans {
            let value = &text[sp.start..sp.end];
            if WHOLE_PLACEHOLDER.is_match(value.trim()).unwrap_or(false) || self.lists.keep.contains(value) {
                continue;
            }
            if !self.to_ph.contains_key(value) {
                self.log_sighting(&sp.label, value, &context(text, sp.start, sp.end));
            }
            let ph = self.placeholder(value, &sp.label);
            *self.stats.entry(sp.label.clone()).or_default() += 1;
            let seen = self.seen.entry(value.to_string()).or_insert_with(|| Seen { detected: 0, matched: 0, texts: 0, context: context(text, sp.start, sp.end) });
            seen.detected += 1;
            seen.texts += new_text as usize;
            out.push((sp.start, sp.end, ph));
        }
        for value in &self.by_length {
            if !text.contains(value.as_str()) || !self.spreads(value) {
                continue;
            }
            for at in whole_matches(text, value) {
                let end = at + value.len();
                if out.iter().all(|&(s, e, _)| e <= at || end <= s) {
                    out.push((at, end, self.to_ph[value].clone()));
                    if let Some(seen) = self.seen.get_mut(value) {
                        seen.matched += 1;
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }

    /// Replace private values with placeholders.
    pub fn redact(&mut self, text: &str) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(text.to_string());
        }
        let (mut out, mut last) = (String::with_capacity(text.len()), 0);
        for (start, end, ph) in self.replacements(text)? {
            out.push_str(&text[last..start]);
            out.push_str(&ph);
            last = end;
        }
        out.push_str(&text[last..]);
        Ok(out)
    }

    /// Put original values back. `json_escaped` when the text sits inside a JSON string literal.
    pub fn restore(&self, text: &str, json_escaped: bool) -> String {
        if !text.contains('[') {
            return text.to_string();
        }
        let (mut out, mut last) = (String::with_capacity(text.len()), 0);
        for m in PLACEHOLDER.find_iter(text).flatten() {
            out.push_str(&text[last..m.start()]);
            match self.to_val.get(m.as_str()) {
                Some(v) if json_escaped => {
                    let quoted = serde_json::to_string(v).unwrap();
                    out.push_str(&quoted[1..quoted.len() - 1]);
                }
                Some(v) => out.push_str(v),
                None => out.push_str(m.as_str()),
            }
            last = m.end();
        }
        out.push_str(&text[last..]);
        out
    }
}

/// The shape of a text: a for a lowercase letter, A for an uppercase one, 9 for a digit, 文 for any other
/// letter, everything else as it is. Safe to show and to save.
pub fn shape(text: &str) -> String {
    let one = |c: char| match c {
        'a'..='z' => 'a',
        'A'..='Z' => 'A',
        '0'..='9' => '9',
        _ if c.is_alphabetic() => '文',
        _ => c,
    };
    text.chars().map(one).collect()
}

/// The text around a span, on one line, with the value between « and ».
fn context(text: &str, start: usize, end: usize) -> String {
    const AROUND: usize = 40;
    let from = (start.saturating_sub(AROUND)..=start).find(|&i| text.is_char_boundary(i)).unwrap();
    let to = (end..=(end + AROUND).min(text.len())).rev().find(|&i| text.is_char_boundary(i)).unwrap();
    format!("{}«{}»{}", &text[from..start], &text[start..end], &text[end..to]).replace(char::is_control, " ")
}

/// Where `value` occurs in `text` without being part of a longer ASCII word, so "Khan" leaves "Khanate" alone.
fn whole_matches(text: &str, value: &str) -> Vec<usize> {
    let word = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric());
    let inside_word = |i: usize| {
        word(value.chars().next()) && word(text[..i].chars().next_back()) || word(value.chars().next_back()) && word(text[i + value.len()..].chars().next())
    };
    text.match_indices(value).map(|(i, _)| i).filter(|&i| !inside_word(i)).collect()
}

/// Restores placeholders in text that arrives in pieces, holding back a tail that may be an unfinished placeholder.
#[derive(Default)]
pub struct StreamRestorer {
    buf: String,
    json_escaped: bool,
}

impl StreamRestorer {
    pub fn new(json_escaped: bool) -> Self {
        StreamRestorer { buf: String::new(), json_escaped }
    }

    pub fn feed(&mut self, piece: &str, vault: &Vault) -> String {
        self.buf.push_str(piece);
        let cut = TAIL.find(&self.buf).ok().flatten().map(|m| m.start()).unwrap_or(self.buf.len());
        let ready: String = self.buf.drain(..cut).collect();
        vault.restore(&ready, self.json_escaped)
    }

    pub fn flush(&mut self, vault: &Vault) -> String {
        let ready = std::mem::take(&mut self.buf);
        vault.restore(&ready, self.json_escaped)
    }
}
