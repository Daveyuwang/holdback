//! The distilled detector: a 6-layer multilingual BERT encoder with a token-classification head, trained to label the
//! same eight categories as the Privacy Filter. Selected by HOLDBACK_STUDENT_DIR, a directory with config.json,
//! model.safetensors (f32) and tokenizer.json. Weights are read in place from the memory-mapped file, so they count
//! as file cache, not process memory.
use crate::detect::Span;
use crate::native::sgemm;
use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

/// Tokens per window, without the start and end token; the windows the model was trained on.
const BODY: usize = 254;
/// Windows run per model call.
const GROUP: usize = 16;
/// Windows whose labels are remembered.
const REMEMBERED: usize = 8192;
/// Tokens in a call from which the per-window and per-row work runs on several threads; below it, thread
/// handoffs cost more than they save.
const PARALLEL: usize = 512;

pub struct Student {
    model: Bert,
    tok: tokenizers::Tokenizer,
    work: Work,
    cls: u32,
    sep: u32,
    /// Label names by id: "O", "B-private_person", "I-private_person", ...
    names: Vec<String>,
    /// Labels of windows already classified, by their tokens. Each window is labelled on its own, so this changes
    /// no output.
    labelled: HashMap<Vec<u32>, Vec<u8>>,
}

impl Student {
    pub fn load(dir: &Path) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
        let model = Bert::load(dir)?;
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let names = (0..model.classes).map(|i| cfg["id2label"][i.to_string()].as_str().map(String::from)).collect::<Option<Vec<_>>>();
        let names = names.context("config.json has no label names (id2label)")?;
        let id = |t: &str| tok.token_to_id(t).with_context(|| format!("tokenizer has no {t}"));
        Ok(Student { cls: id("<s>")?, sep: id("</s>")?, model, tok, work: Work::default(), names, labelled: HashMap::new() })
    }

    /// Private-value spans with byte offsets into `text`, non-overlapping and sorted.
    pub fn detect(&mut self, text: &str) -> Result<Vec<Span>> {
        let enc = self.tok.encode(text, false).map_err(|e| anyhow!("{e}"))?;
        let (ids, offsets) = (enc.get_ids(), enc.get_offsets());
        let windows: Vec<&[u32]> = ids.chunks(BODY).collect();
        let new: Vec<&[u32]> = windows.iter().copied().filter(|w| !self.labelled.contains_key(*w)).collect();
        let mut fresh: HashMap<&[u32], Vec<u8>> = HashMap::new();
        for group in new.chunks(GROUP) {
            let logits = self.model.logits(group, self.cls, self.sep, &mut self.work)?;
            let mut rows = logits.chunks(self.model.classes);
            for w in group {
                let labels = rows.by_ref().take(w.len()).map(|r| r.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u8);
                fresh.insert(w, labels.collect());
            }
        }
        let path: Vec<u8> = windows.iter().flat_map(|w| self.labelled.get(*w).or_else(|| fresh.get(w)).unwrap().iter().copied()).collect();
        if self.labelled.len() + fresh.len() > REMEMBERED {
            self.labelled.clear();
        }
        self.labelled.extend(fresh.into_iter().map(|(w, l)| (w.to_vec(), l)));

        // a B- label opens a span; an I- label continues an open span of its category, or opens one
        let mut spans: Vec<Span> = vec![];
        let mut open = false;
        for (&(s, e), &l) in offsets.iter().zip(&path) {
            let name = &self.names[l as usize];
            let Some((tag, cat)) = name.split_once('-') else {
                open = false;
                continue;
            };
            match spans.last_mut() {
                Some(last) if open && tag == "I" && last.label == cat => last.end = e,
                _ => {
                    spans.push(Span { start: s, end: e, label: cat.to_string() });
                    open = true;
                }
            }
        }
        for sp in &mut spans {
            let piece = &text[sp.start..sp.end];
            sp.start += piece.len() - piece.trim_start().len();
            sp.end -= piece.len() - piece.trim_end().len();
        }
        spans.retain(|sp| sp.end > sp.start);
        Ok(spans)
    }
}

struct Bert {
    data: Mmap,
    start: usize,
    tensors: HashMap<String, (usize, usize)>,
    hidden: usize,
    heads: usize,
    layers: usize,
    inter: usize,
    classes: usize,
    eps: f32,
}

thread_local! {
    /// One window's attention scores, kept per thread so that windows on several threads do not allocate.
    static SCORES: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Buffers reused across calls.
#[derive(Default)]
struct Work {
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    ctx: Vec<f32>,
    tmp: Vec<f32>,
    ff: Vec<f32>,
}

impl Bert {
    fn load(dir: &Path) -> Result<Self> {
        let cfg: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let file = std::fs::File::open(dir.join("model.safetensors"))?;
        let data = unsafe { Mmap::map(&file)? };
        let len = u64::from_le_bytes(data[..8].try_into()?) as usize;
        let header: Value = serde_json::from_slice(&data[8..8 + len])?;
        let mut tensors = HashMap::new();
        for (name, t) in header.as_object().context("safetensors header")? {
            if name == "__metadata__" {
                continue;
            }
            if t["dtype"] != "F32" {
                bail!("tensor {name} is {}, only F32 is read", t["dtype"]);
            }
            let off = t["data_offsets"].as_array().context("tensor offsets")?;
            tensors.insert(name.clone(), (off[0].as_u64().unwrap_or(0) as usize, off[1].as_u64().unwrap_or(0) as usize));
        }
        let get = |k: &str| cfg[k].as_u64().with_context(|| format!("config.json has no {k}")).map(|v| v as usize);
        let mut bert = Bert {
            data,
            start: 8 + len,
            tensors,
            hidden: get("hidden_size")?,
            heads: get("num_attention_heads")?,
            layers: get("num_hidden_layers")?,
            inter: get("intermediate_size")?,
            classes: 0,
            eps: cfg["layer_norm_eps"].as_f64().unwrap_or(1e-12) as f32,
        };
        bert.classes = bert.f32s("classifier.bias")?.len();
        Ok(bert)
    }

    fn f32s(&self, name: &str) -> Result<&[f32]> {
        let (a, b) = *self.tensors.get(name).with_context(|| format!("model has no tensor {name}"))?;
        let (head, body, tail) = unsafe { self.data[self.start + a..self.start + b].align_to::<f32>() };
        if !head.is_empty() || !tail.is_empty() {
            bail!("tensor {name} is not 4-byte aligned");
        }
        Ok(body)
    }

    /// Class logits for every token of every window, `[tokens, classes]` in window order. Each window is wrapped in
    /// the start and end tokens and attends only to itself.
    fn logits(&self, windows: &[&[u32]], cls: u32, sep: u32, work: &mut Work) -> Result<Vec<f32>> {
        let (h, nh) = (self.hidden, self.heads);
        let d = h / nh;
        let rows: Vec<Vec<u32>> = windows.iter().map(|w| [&[cls][..], w, &[sep][..]].concat()).collect();
        let t: usize = rows.iter().map(Vec::len).sum();
        let Work { x, q, k, v, ctx, tmp, ff } = work;
        for buf in [&mut *x, &mut *q, &mut *k, &mut *v, &mut *ctx, &mut *tmp] {
            buf.resize(t * h, 0.0);
        }
        ff.resize(t * self.inter, 0.0);

        let word = self.f32s("bert.embeddings.word_embeddings.weight")?;
        let (pos, kind) = (self.f32s("bert.embeddings.position_embeddings.weight")?, self.f32s("bert.embeddings.token_type_embeddings.weight")?);
        let mut i = 0;
        for row in &rows {
            for (p, &id) in row.iter().enumerate() {
                let (w, ps) = (&word[id as usize * h..][..h], &pos[p * h..][..h]);
                for (j, out) in x[i * h..][..h].iter_mut().enumerate() {
                    *out = w[j] + ps[j] + kind[j];
                }
                i += 1;
            }
        }
        layer_norm(x, self.f32s("bert.embeddings.LayerNorm.weight")?, self.f32s("bert.embeddings.LayerNorm.bias")?, h, self.eps);

        let scale = 1.0 / (d as f32).sqrt();
        for l in 0..self.layers {
            let p = format!("bert.encoder.layer.{l}");
            self.linear(x, &format!("{p}.attention.self.query"), t, h, h, q)?;
            self.linear(x, &format!("{p}.attention.self.key"), t, h, h, k)?;
            self.linear(x, &format!("{p}.attention.self.value"), t, h, h, v)?;
            // each window attends only to itself, so windows run in parallel, each on its own rows of ctx
            let mut parts = vec![];
            let (mut rest, mut at) = (&mut ctx[..], 0);
            for row in &rows {
                let (mine, tail) = rest.split_at_mut(row.len() * h);
                parts.push((at, row.len(), mine));
                rest = tail;
                at += row.len();
            }
            let (q, k, v) = (&q[..], &k[..], &v[..]);
            let attend = |(at, n, out): (usize, usize, &mut [f32])| {
                SCORES.with_borrow_mut(|scores| {
                    scores.resize(n * n, 0.0);
                    for head in 0..nh {
                        let off = at * h + head * d;
                        sgemm(false, true, n, n, d, &q[off..], h, &k[off..], h, scores, n);
                        scores.chunks_mut(n).for_each(|r| softmax(r, scale));
                        sgemm(false, false, n, d, n, scores, n, &v[off..], h, &mut out[head * d..], h);
                    }
                })
            };
            if t < PARALLEL {
                parts.into_iter().for_each(attend);
            } else {
                parts.into_par_iter().for_each(attend);
            }
            self.linear(ctx, &format!("{p}.attention.output.dense"), t, h, h, tmp)?;
            x.iter_mut().zip(tmp.iter()).for_each(|(a, b)| *a += b);
            layer_norm(x, self.f32s(&format!("{p}.attention.output.LayerNorm.weight"))?, self.f32s(&format!("{p}.attention.output.LayerNorm.bias"))?, h, self.eps);
            self.linear(x, &format!("{p}.intermediate.dense"), t, h, self.inter, ff)?;
            rows_of(ff, self.inter, |row| row.iter_mut().for_each(|a| *a = 0.5 * *a * (1.0 + erf(*a * std::f32::consts::FRAC_1_SQRT_2))));
            self.linear(ff, &format!("{p}.output.dense"), t, self.inter, h, tmp)?;
            x.iter_mut().zip(tmp.iter()).for_each(|(a, b)| *a += b);
            layer_norm(x, self.f32s(&format!("{p}.output.LayerNorm.weight"))?, self.f32s(&format!("{p}.output.LayerNorm.bias"))?, h, self.eps);
        }
        let mut all = vec![0f32; t * self.classes];
        self.linear(x, "classifier", t, h, self.classes, &mut all)?;
        let mut out = Vec::with_capacity((t - 2 * rows.len()) * self.classes);
        let mut at = 0;
        for row in &rows {
            out.extend_from_slice(&all[(at + 1) * self.classes..(at + row.len() - 1) * self.classes]);
            at += row.len();
        }
        Ok(out)
    }

    /// y[t, out] = x[t, inp] * W[out, inp]^T + b
    fn linear(&self, x: &[f32], name: &str, t: usize, inp: usize, out: usize, y: &mut [f32]) -> Result<()> {
        let (w, b) = (self.f32s(&format!("{name}.weight"))?, self.f32s(&format!("{name}.bias"))?);
        sgemm(false, true, t, out, inp, x, inp, w, inp, y, out);
        for row in y[..t * out].chunks_mut(out) {
            row.iter_mut().zip(b).for_each(|(a, c)| *a += c);
        }
        Ok(())
    }
}

/// `f` on every row of `x`, in parallel when there are enough rows to pay for it.
fn rows_of(x: &mut [f32], width: usize, f: impl Fn(&mut [f32]) + Sync + Send) {
    if x.len() / width < PARALLEL {
        x.chunks_mut(width).for_each(f);
    } else {
        x.par_chunks_mut(width).for_each(f);
    }
}

fn layer_norm(x: &mut [f32], w: &[f32], b: &[f32], h: usize, eps: f32) {
    rows_of(x, h, |row| {
        let mean = row.iter().sum::<f32>() / h as f32;
        let var = row.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / h as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for (j, a) in row.iter_mut().enumerate() {
            *a = (*a - mean) * inv * w[j] + b[j];
        }
    });
}

/// In place: softmax(row * scale).
fn softmax(row: &mut [f32], scale: f32) {
    let m = row.iter().fold(f32::MIN, |a, &b| a.max(b)) * scale;
    row.iter_mut().for_each(|s| *s = exp(*s * scale - m));
    let sum: f32 = row.chunks(8).map(|c| c.iter().sum::<f32>()).sum();
    let inv = 1.0 / sum;
    row.iter_mut().for_each(|s| *s *= inv);
}

/// e^x in f32 without a libm call, so loops over it vectorize: x = n ln2 + r with |r| <= ln2/2, e^r by its Taylor
/// series to the 7th power (relative error below 2e-8 on that range), times 2^n from the exponent bits.
#[inline(always)]
fn exp(x: f32) -> f32 {
    let x = x.clamp(-87.0, 88.0);
    let n = (x * std::f32::consts::LOG2_E).round();
    let r = x - n * 0.693_145_75 - n * 1.428_606_8e-6;
    let p = 1.0 + r * (1.0 + r * (0.5 + r * (1.0 / 6.0 + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0 + r * (1.0 / 5040.0)))))));
    p * f32::from_bits(((n as i32 + 127) << 23) as u32)
}

/// The error function for the exact GELU BERT uses: Abramowitz and Stegun 7.1.26, largest error 1.5e-7.
#[inline(always)]
fn erf(x: f32) -> f32 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let y = 1.0 - ((((1.061_405_4 * t - 1.453_152) * t + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * exp(-a * a);
    y.copysign(x)
}

#[cfg(test)]
mod tests {
    /// Logits that PyTorch gives for a few windows (testdata/student_logits.json). Runs when HOLDBACK_STUDENT_DIR
    /// points at the model.
    #[test]
    fn matches_pytorch() {
        let Ok(dir) = std::env::var("HOLDBACK_STUDENT_DIR") else {
            eprintln!("skipped: HOLDBACK_STUDENT_DIR is not set");
            return;
        };
        let fixture: serde_json::Value = serde_json::from_str(include_str!("testdata/student_logits.json")).unwrap();
        let windows: Vec<Vec<u32>> = serde_json::from_value(fixture["windows"].clone()).unwrap();
        let want: Vec<f32> = serde_json::from_value(fixture["logits"].clone()).unwrap();
        let student = super::Student::load(std::path::Path::new(&dir)).unwrap();
        let refs: Vec<&[u32]> = windows.iter().map(Vec::as_slice).collect();
        let got = student.model.logits(&refs, student.cls, student.sep, &mut super::Work::default()).unwrap();
        assert_eq!(got.len(), want.len());
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(worst < 5e-3, "largest logit difference from PyTorch: {worst}");
    }
}
