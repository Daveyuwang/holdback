//! OpenAI Privacy Filter: tiktoken o200k_base, token classification (crate::native), constrained Viterbi,
//! token spans to character spans. Matches opf's predict_text (checked span for span against Python).
use crate::native::{Model, Work};
use anyhow::{Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

const NEG_INF: f32 = -1e9;
/// Longest token run classified as one window; the model attends 128 tokens back, so a window sees all of itself.
const WINDOW: usize = 128;
/// Characters after which a window may end when no line break is near; shared with the student.
pub const SEPARATORS: &str = ",;{}[]()。，；！？";
/// Windows classified per model call. They share each expert's weights; working memory grows with this.
const GROUP: usize = 8;
/// Windows whose labels are remembered, about 0.6 KB each.
const REMEMBERED: usize = 8192;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Span {
    pub start: usize, // byte offset
    pub end: usize,
    pub label: String,
}

struct Labels {
    names: Vec<String>,
    token_span: Vec<usize>,
    tags: Vec<Option<char>>,
    background: usize,
}

impl Labels {
    fn from_config(cfg: &Value) -> Result<Self> {
        let id2label = cfg["id2label"].as_object().context("config.json has no id2label")?;
        let mut class_names = vec![String::new(); id2label.len()];
        for (k, v) in id2label {
            class_names[k.parse::<usize>()?] = v.as_str().context("label")?.to_string();
        }
        let mut names = vec!["O".to_string()];
        let (mut token_span, mut tags, mut background) = (vec![], vec![], 0);
        for (idx, name) in class_names.iter().enumerate() {
            if name == "O" {
                background = idx;
                token_span.push(0);
                tags.push(None);
                continue;
            }
            let (boundary, base) = name.split_once('-').context("label format")?;
            let span = names.iter().position(|n| n == base).unwrap_or_else(|| {
                names.push(base.to_string());
                names.len() - 1
            });
            token_span.push(span);
            tags.push(boundary.chars().next());
        }
        Ok(Labels { names, token_span, tags, background })
    }

    fn is_bg(&self, idx: usize) -> bool {
        idx == self.background || self.token_span[idx] == 0
    }

    /// Start, end and transition scores of the constrained decoder (the shipped calibration biases are zero).
    fn tables(&self) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
        let n = self.tags.len();
        let opens = |t: Option<char>| matches!(t, Some('B') | Some('S'));
        let (mut start, mut end, mut trans) = (vec![NEG_INF; n], vec![NEG_INF; n], vec![vec![NEG_INF; n]; n]);
        for p in 0..n {
            if opens(self.tags[p]) || p == self.background {
                start[p] = 0.0;
            }
            if matches!(self.tags[p], Some('E') | Some('S')) || p == self.background {
                end[p] = 0.0;
            }
            for q in 0..n {
                let ok = if self.is_bg(p) || matches!(self.tags[p], Some('E') | Some('S')) {
                    self.is_bg(q) || opens(self.tags[q])
                } else {
                    self.token_span[p] == self.token_span[q] && matches!(self.tags[q], Some('I') | Some('E'))
                };
                if ok {
                    trans[p][q] = 0.0;
                }
            }
        }
        (start, end, trans)
    }
}

fn viterbi(lp: &[Vec<f32>], start: &[f32], end: &[f32], trans: &[Vec<f32>]) -> Vec<usize> {
    let (t, n) = (lp.len(), start.len());
    let argmax = |v: &[f32]| (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    let mut score: Vec<f32> = (0..n).map(|j| lp[0][j] + start[j]).collect();
    let mut back = vec![vec![0usize; n]; t.saturating_sub(1)];
    for i in 1..t {
        let mut next = vec![0f32; n];
        for q in 0..n {
            let mut best = 0;
            for p in 1..n {
                if score[p] + trans[p][q] > score[best] + trans[best][q] {
                    best = p;
                }
            }
            next[q] = score[best] + trans[best][q] + lp[i][q];
            back[i - 1][q] = best;
        }
        score = next;
    }
    if !score.iter().any(|s| s.is_finite()) {
        return lp.iter().map(|row| argmax(row)).collect();
    }
    for j in 0..n {
        score[j] += end[j];
    }
    let mut path = vec![argmax(&score); t];
    for i in (0..t - 1).rev() {
        path[i] = back[i][path[i + 1]];
    }
    path
}

fn labels_to_spans(path: &[usize], l: &Labels) -> Vec<(usize, usize, usize)> {
    let (mut out, mut cur, mut start) = (vec![], None::<usize>, 0usize);
    let close = |cur: &mut Option<usize>, start: usize, end: usize, out: &mut Vec<_>| {
        if let Some(c) = cur.take() {
            out.push((c, start, end));
        }
    };
    for (i, &lab) in path.iter().enumerate() {
        let span = l.token_span[lab];
        if span == 0 {
            close(&mut cur, start, i, &mut out);
            continue;
        }
        match l.tags[lab] {
            Some('S') => {
                close(&mut cur, start, i, &mut out);
                out.push((span, i, i + 1));
            }
            Some('B') => {
                close(&mut cur, start, i, &mut out);
                cur = Some(span);
                start = i;
            }
            Some('I') if cur != Some(span) => {
                close(&mut cur, start, i, &mut out);
                cur = Some(span);
                start = i;
            }
            Some('I') => {}
            Some('E') if cur != Some(span) => {
                close(&mut cur, start, i, &mut out);
                out.push((span, i, i + 1));
            }
            Some('E') => {
                out.push((span, start, i + 1));
                cur = None;
            }
            _ => close(&mut cur, start, i, &mut out),
        }
    }
    close(&mut cur, start, path.len(), &mut out);
    out
}

/// The detector holdback uses: the distilled model by default, or the Privacy Filter (HOLDBACK_DETECTOR).
pub enum Detector {
    Filter(Box<Filter>),
    Student(Box<crate::student::Student>),
}

impl Detector {
    pub fn load(dir: &Path) -> Result<Self> {
        Ok(Detector::Filter(Box::new(Filter::load(dir)?)))
    }

    /// Private-value spans with byte offsets into `text`, non-overlapping and sorted.
    pub fn detect(&mut self, text: &str) -> Result<Vec<Span>> {
        match self {
            Detector::Filter(f) => f.detect(text),
            Detector::Student(s) => s.detect(text),
        }
    }

    /// The name the status page and `holdback doctor` show.
    pub fn name(&self) -> &'static str {
        match self {
            Detector::Filter(_) => "privacy-filter",
            Detector::Student(_) => "distilled",
        }
    }
}

pub struct Filter {
    model: Model,
    work: Work,
    bpe: tiktoken_rs::CoreBPE,
    labels: Labels,
    start: Vec<f32>,
    end: Vec<f32>,
    trans: Vec<Vec<f32>>,
    /// Labels of windows already classified, by their tokens. A window is labelled on its own, so this
    /// changes no output: a text that differs a little from an earlier one costs only its changed windows.
    labelled: HashMap<Vec<u32>, Vec<u8>>,
}

impl Filter {
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let labels = Labels::from_config(&cfg)?;
        let (start, end, trans) = labels.tables();
        Ok(Filter { model: Model::load(dir)?, work: Work::default(), bpe: tiktoken_rs::o200k_base()?, labels, start, end, trans, labelled: HashMap::new() })
    }

    /// Private-value spans with byte offsets into `text`, non-overlapping and sorted.
    pub fn detect(&mut self, text: &str) -> Result<Vec<Span>> {
        let ids = self.bpe.encode_with_special_tokens(text);
        let (mut windows, mut at) = (vec![], 0);
        while at < ids.len() {
            let end = pass_end(&self.bpe, &ids, at);
            windows.push(&ids[at..end]);
            at = end;
        }
        let new: Vec<&[u32]> = windows.iter().copied().filter(|w| !self.labelled.contains_key(*w)).collect();
        for group in new.chunks(GROUP) {
            let labels = self.label_tokens(group)?;
            if self.labelled.len() + group.len() > REMEMBERED {
                self.labelled.clear();
            }
            let mut at = 0;
            for w in group {
                self.labelled.insert(w.to_vec(), labels[at..at + w.len()].iter().map(|&l| l as u8).collect());
                at += w.len();
            }
        }
        let mut path = Vec::with_capacity(ids.len());
        for w in &windows {
            match self.labelled.get(*w) {
                Some(labels) => path.extend(labels.iter().map(|&l| l as usize)),
                None => path.extend(self.label_tokens(&[w])?), // dropped when the cache was cleared just above
            }
        }
        let token_spans = labels_to_spans(&path, &self.labels);

        // byte range of every token; tokens that split a character widen to whole characters, as opf does
        let (mut starts, mut ends, mut cursor) = (vec![], vec![], 0usize);
        for &id in &ids {
            let len = self.bpe.decode_bytes(&[id]).map(|b| b.len()).unwrap_or(0);
            starts.push(cursor);
            cursor += len;
            ends.push(cursor);
        }
        let floor = |mut i: usize| {
            while i > 0 && !text.is_char_boundary(i) {
                i -= 1;
            }
            i
        };
        let ceil = |mut i: usize| {
            while i < text.len() && !text.is_char_boundary(i) {
                i += 1;
            }
            i.min(text.len())
        };
        let mut spans = vec![];
        for (lab, a, b) in token_spans {
            let (s, e) = (floor(starts[a]), ceil(ends[b - 1]));
            let piece = &text[s..e];
            let lead = piece.len() - piece.trim_start().len();
            let trail = piece.len() - piece.trim_end().len();
            let (s, e) = (s + lead, e - trail);
            if e > s {
                spans.push(Span { start: s, end: e, label: self.labels.names[lab].clone() });
            }
        }
        spans.sort_by(|x, y| (x.start, std::cmp::Reverse(x.end - x.start), &x.label).cmp(&(y.start, std::cmp::Reverse(y.end - y.start), &y.label)));
        let mut kept: Vec<Span> = vec![];
        let mut last = 0;
        for sp in spans {
            if sp.start >= last {
                last = sp.end;
                kept.push(sp);
            }
        }
        Ok(kept)
    }

    /// Label ids for every token of `windows`, in order; each window is decoded on its own.
    fn label_tokens(&mut self, windows: &[&[u32]]) -> Result<Vec<usize>> {
        let logits = self.model.logits(windows, &mut self.work)?;
        let c = self.labels.tags.len();
        let lp: Vec<Vec<f32>> = logits
            .chunks(c)
            .map(|row| {
                let m = row.iter().cloned().fold(f32::MIN, f32::max);
                let lse = m + row.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
                row.iter().map(|x| x - lse).collect()
            })
            .collect();
        let (mut path, mut at) = (vec![], 0);
        for w in windows {
            path.extend(viterbi(&lp[at..at + w.len()], &self.start, &self.end, &self.trans));
            at += w.len();
        }
        Ok(path)
    }
}

/// Where the pass starting at `at` ends, so that a value is not cut in two: near the window's end, after a line
/// break, else after a separator such as a comma or bracket (long one-line JSON has no line breaks), else before
/// a space.
pub fn pass_end(bpe: &tiktoken_rs::CoreBPE, ids: &[u32], at: usize) -> usize {
    let hard = at + WINDOW;
    if hard >= ids.len() {
        return ids.len();
    }
    let piece = |i: usize| String::from_utf8_lossy(&bpe.decode_bytes(&[ids[i]]).unwrap_or_default()).into_owned();
    let after = |rule: &dyn Fn(usize) -> bool| (hard - WINDOW / 4..hard).rev().find(|&i| rule(i)).map(|i| i + 1);
    let separator = |i: usize| {
        let p = piece(i);
        p.contains(|c| SEPARATORS.contains(c)) && !p.contains(char::is_alphanumeric) // `","` yes, `,John` no
    };
    after(&|i| piece(i).contains('\n'))
        .or_else(|| after(&separator))
        .or_else(|| after(&|i| piece(i + 1).starts_with(char::is_whitespace)))
        .unwrap_or(hard)
}
