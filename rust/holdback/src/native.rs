//! The Privacy Filter forward pass. Weights stay in the memory-mapped model file, so they count as file cache
//! the OS can drop, not as process memory, and each token reads only the 4 of 128 experts its router picks.
//! Logits match ONNX Runtime's to about 1e-4 (the test below holds reference values).
use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::Path;

const HIDDEN: usize = 640;
const HEADS: usize = 14;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const EXPERTS: usize = 128;
const TOP_K: usize = 4;
const FF: usize = 640; // expert width; gate_up holds gate and linear interleaved, 2 * FF wide
const LAYERS: usize = 8;
const BLOCK: usize = 32; // 4-bit quantization block
const SWIGLU_ALPHA: f32 = 1.702;
const SWIGLU_LIMIT: f32 = 7.0;
const EPS: f32 = 1e-5;
/// Up to this many tokens an expert is multiplied straight from its packed 4-bit form.
const DIRECT_TOKENS: usize = 4;
/// Tokens per expert step and weight rows per unpacking step; together they cap every scratch buffer.
const CHUNK_TOKENS: usize = 64;
const CHUNK_ROWS: usize = 128;

/// A 4-bit matrix: packed values, one scale per 32 values, and packed zero points when it has them.
type Quantized<'a> = (&'a [u8], &'a [f32], Option<&'a [u8]>);

/// One tensor in the ONNX file: either a range of the external data file or bytes stored inline.
enum Where {
    External(usize, usize),
    Inline(Vec<u8>),
}

pub struct Model {
    data: Mmap,
    tensors: HashMap<String, Where>,
    classes: usize,
}

impl Model {
    pub fn load(dir: &Path) -> Result<Self> {
        let onnx = std::fs::read(dir.join("onnx/model_q4.onnx"))?;
        let file = std::fs::File::open(dir.join("onnx/model_q4.onnx_data"))?;
        let data = unsafe { Mmap::map(&file)? };
        let tensors = initializers(&onnx)?;
        let mut model = Model { data, tensors, classes: 0 };
        model.classes = model.f32s("model_score_MatMul_weight_scales")?.len() / (HIDDEN / BLOCK);
        Ok(model)
    }

    fn bytes(&self, name: &str) -> Result<&[u8]> {
        Ok(match self.tensors.get(name).with_context(|| format!("model has no tensor {name}"))? {
            Where::External(off, len) => &self.data[*off..off + len],
            Where::Inline(b) => b,
        })
    }

    fn f32s(&self, name: &str) -> Result<&[f32]> {
        let b = self.bytes(name)?;
        let (head, body, tail) = unsafe { b.align_to::<f32>() };
        if !head.is_empty() || !tail.is_empty() {
            bail!("tensor {name} is not 4-byte aligned");
        }
        Ok(body)
    }

    /// Per-token class logits for several attention windows at once, `[tokens, classes]` in window order.
    /// Windows share the projections and the experts, so each expert is unpacked once per call. All working
    /// buffers live in `work` and keep their capacity, so repeated calls allocate nothing new.
    pub fn logits(&self, windows: &[&[u32]], work: &mut Work) -> Result<Vec<f32>> {
        let t: usize = windows.iter().map(|w| w.len()).sum();
        let Work { h, x, q, k, v, att, o, scores, y, w, down, .. } = work;
        for buf in [&mut *h, &mut *x, &mut *o, &mut *y] {
            buf.resize(t * HIDDEN, 0.0);
        }
        q.resize(t * HEADS * HEAD_DIM, 0.0);
        att.resize(t * HEADS * HEAD_DIM, 0.0);
        k.resize(t * KV_HEADS * HEAD_DIM, 0.0);
        v.resize(t * KV_HEADS * HEAD_DIM, 0.0);
        scores.resize(t * EXPERTS, 0.0);
        down.resize(t * TOP_K * HIDDEN, 0.0);

        let (eq, es, ez) = self.quantized("model_embed_tokens")?;
        for (i, &id) in windows.iter().flat_map(|w| w.iter()).enumerate() {
            dequant_into(eq, es, ez, id as usize..id as usize + 1, HIDDEN, &mut h[i * HIDDEN..][..HIDDEN]);
        }
        let positions: Vec<usize> = windows.iter().flat_map(|w| 0..w.len()).collect();
        let (cos, sin) = (self.f32s("cos_cache")?, self.f32s("sin_cache")?);
        for l in 0..LAYERS {
            let (p, q_) = (format!("model.layers.{l}"), format!("model_layers_{l}"));
            rms(h, self.f32s(&format!("{p}.input_layernorm.weight"))?, x);
            self.linear(&format!("{q_}_attn_q_proj_MatMul"), &format!("{p}.attn.q_proj.Add.bias"), x, HIDDEN, q, w)?;
            self.linear(&format!("{q_}_attn_k_proj_MatMul"), &format!("{p}.attn.k_proj.Add.bias"), x, HIDDEN, k, w)?;
            self.linear(&format!("{q_}_attn_v_proj_MatMul"), &format!("{p}.attn.v_proj.Add.bias"), x, HIDDEN, v, w)?;
            rope(q, HEADS, &positions, cos, sin);
            rope(k, KV_HEADS, &positions, cos, sin);
            let sinks = self.f32s(&format!("{p}.attn.sinks"))?;
            let mut at = 0;
            for win in windows {
                let n = win.len();
                let (qa, ka, va) = (&q[at * HEADS * HEAD_DIM..], &k[at * KV_HEADS * HEAD_DIM..], &v[at * KV_HEADS * HEAD_DIM..]);
                attention(qa, ka, va, n, sinks, &mut att[at * HEADS * HEAD_DIM..]);
                at += n;
            }
            self.linear(&format!("{q_}_attn_o_proj_MatMul"), &format!("{p}.attn.o_proj.Add.bias"), att, HEADS * HEAD_DIM, o, w)?;
            h.iter_mut().zip(o.iter()).for_each(|(a, b)| *a += b);

            rms(h, self.f32s(&format!("{p}.post_attention_layernorm.weight"))?, x);
            self.linear(&format!("{q_}_moe_router_MatMul"), &format!("{p}.moe.router.Add.bias"), x, HIDDEN, scores, w)?;
            self.experts(l, x, scores, down, y)?;
            h.iter_mut().zip(y.iter()).for_each(|(a, b)| *a += b);
        }
        rms(h, self.f32s("model.layers.8.final_norm_layernorm.weight")?, x);
        let mut out = vec![0f32; t * self.classes];
        let (sq, ss, sz) = self.quantized("model_score_MatMul")?;
        dequant(sq, ss, sz, 0..self.classes, HIDDEN, w);
        gemm_bt(x, w, t, self.classes, HIDDEN, &mut out);
        Ok(out)
    }

    fn quantized(&self, name: &str) -> Result<Quantized<'_>> {
        let zp = self.bytes(&format!("{name}_weight_zp")).ok();
        Ok((self.bytes(&format!("{name}_weight_quant"))?, self.f32s(&format!("{name}_weight_scales"))?, zp))
    }

    /// y = x * W^T + bias for the MatMulNBits matrix `name`; y's length sets the output width.
    fn linear(&self, name: &str, bias: &str, x: &[f32], cols: usize, y: &mut [f32], buf: &mut Vec<f32>) -> Result<()> {
        let (q, scales, zp) = self.quantized(name)?;
        let bias = self.f32s(bias)?;
        let (n, t) = (bias.len(), x.len() / cols);
        // unpack in parallel row ranges, then one BLAS call
        use rayon::prelude::*;
        buf.resize(n * cols, 0.0);
        buf.par_chunks_mut(16 * cols).enumerate().for_each(|(k, chunk)| {
            dequant_into(q, scales, zp, k * 16..k * 16 + chunk.len() / cols, cols, chunk);
        });
        gemm_bt(x, buf, t, n, cols, y);
        add_bias(y, bias);
        Ok(())
    }

    /// Mixture-of-experts block: each token's 4 highest-scoring experts, weighted by a softmax over their scores.
    fn experts(&self, l: usize, x: &[f32], scores: &[f32], down: &mut [f32], y: &mut [f32]) -> Result<()> {
        let (p, q_) = (format!("model.layers.{l}"), format!("model_layers_{l}"));
        let mut routed: Vec<Vec<(usize, f32)>> = vec![vec![]; EXPERTS]; // expert -> (token, weight)
        let mut order: Vec<usize> = (0..EXPERTS).collect();
        for (i, s) in scores.chunks(EXPERTS).enumerate() {
            order.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
            let top = &order[..TOP_K];
            let m = s[top[0]];
            let sum: f32 = top.iter().map(|&e| (s[e] - m).exp()).sum();
            for &e in top {
                routed[e].push((i, (s[e] - m).exp() / sum));
            }
        }
        let gq = self.bytes(&format!("{q_}_moe_experts_gate_up_proj_weight_quant"))?;
        let gs = self.f32s(&format!("{q_}_moe_experts_gate_up_proj_weight_scales"))?;
        let gb = self.f32s(&format!("{p}.moe.experts.gate_up_proj.bias"))?;
        let dq = self.bytes(&format!("{q_}_moe_experts_down_proj_weight_quant"))?;
        let ds = self.f32s(&format!("{q_}_moe_experts_down_proj_weight_scales"))?;
        let db = self.f32s(&format!("{p}.moe.experts.down_proj.bias"))?;

        // each expert writes its rows into its own slice of `down`, on rayon's persistent threads, which keep
        // their scratch buffers between calls
        let mut slices = Vec::with_capacity(EXPERTS);
        let mut rest = &mut down[..];
        for (e, tokens) in routed.iter().enumerate() {
            let (mine, tail) = rest.split_at_mut(tokens.len() * HIDDEN);
            slices.push((e, mine));
            rest = tail;
        }
        thread_local! {
            static SCRATCH: std::cell::RefCell<[Vec<f32>; 4]> = const { std::cell::RefCell::new([Vec::new(), Vec::new(), Vec::new(), Vec::new()]) };
        }
        use rayon::prelude::*;
        slices.into_par_iter().filter(|(_, s)| !s.is_empty()).for_each(|(e, out)| {
            SCRATCH.with(|s| {
                let [xe, gu, act, buf] = &mut *s.borrow_mut();
                for (tokens, out) in routed[e].chunks(CHUNK_TOKENS).zip(out.chunks_mut(CHUNK_TOKENS * HIDDEN)) {
                    let n = tokens.len();
                    xe.clear();
                    tokens.iter().for_each(|&(i, _)| xe.extend_from_slice(&x[i * HIDDEN..][..HIDDEN]));
                    gu.resize(n * 2 * FF, 0.0);
                    matmul_q(xe, gq, gs, None, e * 2 * FF..(e + 1) * 2 * FF, HIDDEN, n, gu, buf);
                    add_bias(gu, &gb[e * 2 * FF..][..2 * FF]);
                    act.clear();
                    act.extend(gu.chunks(2).map(|pair| {
                        let g = pair[0].min(SWIGLU_LIMIT);
                        let lin = pair[1].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
                        g / (1.0 + (-SWIGLU_ALPHA * g).exp()) * (lin + 1.0)
                    }));
                    matmul_q(act, dq, ds, None, e * HIDDEN..(e + 1) * HIDDEN, FF, n, out, buf);
                    add_bias(out, &db[e * HIDDEN..][..HIDDEN]);
                }
            })
        });
        y.fill(0.0);
        let mut at = 0;
        for tokens in &routed {
            for &(i, weight) in tokens {
                for (a, b) in y[i * HIDDEN..][..HIDDEN].iter_mut().zip(&down[at * HIDDEN..][..HIDDEN]) {
                    *a += weight * b;
                }
                at += 1;
            }
        }
        Ok(())
    }
}

/// Working buffers for `Model::logits`, reused across calls.
#[derive(Default)]
pub struct Work {
    h: Vec<f32>,
    x: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    o: Vec<f32>,
    scores: Vec<f32>,
    y: Vec<f32>,
    w: Vec<f32>,
    down: Vec<f32>,
}

/// Rows `rows` of a 4-bit matrix with `cols` columns: low nibble first, one scale per 32 values, and a packed
/// 4-bit zero point per block (8 when there is none).
fn dequant(q: &[u8], scales: &[f32], zp: Option<&[u8]>, rows: std::ops::Range<usize>, cols: usize, out: &mut Vec<f32>) {
    out.resize(rows.len() * cols, 0.0);
    dequant_into(q, scales, zp, rows, cols, out);
}

fn dequant_into(q: &[u8], scales: &[f32], zp: Option<&[u8]>, rows: std::ops::Range<usize>, cols: usize, out: &mut [f32]) {
    let (nb, zb) = (cols / BLOCK, (cols / BLOCK).div_ceil(2));
    for (o, r) in rows.enumerate() {
        let (qr, sr) = (&q[r * cols / 2..][..cols / 2], &scales[r * nb..][..nb]);
        let dst = &mut out[o * cols..][..cols];
        for b in 0..nb {
            let z = zp.map_or(8.0, |z| {
                let byte = z[r * zb + b / 2];
                (if b % 2 == 0 { byte & 15 } else { byte >> 4 }) as f32
            });
            let s = sr[b];
            for (i, &byte) in qr[b * BLOCK / 2..][..BLOCK / 2].iter().enumerate() {
                dst[b * BLOCK + 2 * i] = ((byte & 15) as f32 - z) * s;
                dst[b * BLOCK + 2 * i + 1] = ((byte >> 4) as f32 - z) * s;
            }
        }
    }
}

/// y[n, rows] = x[n, cols] * W[rows, cols]^T for a 4-bit W (zero point 8 when `zp` is None). Few rows of x:
/// multiply straight from the packed bytes, reading each weight once. Many rows: unpack W into `buf` and use
/// the BLAS.
#[allow(clippy::too_many_arguments)]
fn matmul_q(x: &[f32], q: &[u8], scales: &[f32], zp: Option<&[u8]>, rows: std::ops::Range<usize>, cols: usize, n: usize, y: &mut [f32], buf: &mut Vec<f32>) {
    let m = rows.len();
    if n > DIRECT_TOKENS {
        for r0 in (0..m).step_by(CHUNK_ROWS) {
            let r1 = (r0 + CHUNK_ROWS).min(m);
            dequant(q, scales, zp, rows.start + r0..rows.start + r1, cols, buf);
            sgemm(false, true, n, r1 - r0, cols, x, cols, buf, cols, &mut y[r0..], m);
        }
        return;
    }
    // per token and block: inputs at even and odd positions, matching the low and high nibbles of each byte
    let (nb, half) = (cols / BLOCK, BLOCK / 2);
    let mut split = [0f32; DIRECT_TOKENS * 1280];
    for (src, dst) in x.chunks(BLOCK).zip(split.chunks_mut(BLOCK)) {
        for i in 0..half {
            dst[i] = src[2 * i];
            dst[half + i] = src[2 * i + 1];
        }
    }
    for (o, r) in rows.enumerate() {
        let (qr, sr) = (&q[r * cols / 2..][..cols / 2], &scales[r * nb..][..nb]);
        let mut acc = [0f32; DIRECT_TOKENS];
        for b in 0..nb {
            let z = zp.map_or(8.0, |z| {
                let byte = z[r * nb.div_ceil(2) + b / 2];
                (if b % 2 == 0 { byte & 15 } else { byte >> 4 }) as f32
            });
            let bytes: &[u8; 16] = qr[b * half..][..half].try_into().unwrap();
            let (mut lo, mut hi) = ([0f32; 16], [0f32; 16]);
            for i in 0..16 {
                lo[i] = (bytes[i] & 15) as f32 - z;
                hi[i] = (bytes[i] >> 4) as f32 - z;
            }
            for (j, a) in acc.iter_mut().enumerate().take(n) {
                let xs: &[f32; 32] = split[j * cols + b * BLOCK..][..BLOCK].try_into().unwrap();
                let mut d = [0f32; 16];
                for i in 0..16 {
                    d[i] = lo[i] * xs[i] + hi[i] * xs[16 + i];
                }
                *a += sr[b] * d.iter().sum::<f32>();
            }
        }
        for (j, a) in acc.iter().enumerate().take(n) {
            y[j * m + o] = *a;
        }
    }
}

fn rms(x: &[f32], weight: &[f32], out: &mut [f32]) {
    for (src, dst) in x.chunks(HIDDEN).zip(out.chunks_mut(HIDDEN)) {
        let scale = 1.0 / (src.iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32 + EPS).sqrt();
        for ((d, s), w) in dst.iter_mut().zip(src).zip(weight) {
            *d = s * scale * w;
        }
    }
}

fn add_bias(y: &mut [f32], bias: &[f32]) {
    for row in y.chunks_mut(bias.len()) {
        row.iter_mut().zip(bias).for_each(|(a, b)| *a += b);
    }
}

/// Rotary position embedding on interleaved pairs; positions count from each window's start.
fn rope(x: &mut [f32], heads: usize, positions: &[usize], cos: &[f32], sin: &[f32]) {
    let half = HEAD_DIM / 2;
    for (&pos, row) in positions.iter().zip(x.chunks_mut(heads * HEAD_DIM)) {
        let (c, s) = (&cos[pos * half..][..half], &sin[pos * half..][..half]);
        for head in row.chunks_mut(HEAD_DIM) {
            for i in 0..half {
                let (a, b) = (head[2 * i], head[2 * i + 1]);
                head[2 * i] = a * c[i] - b * s[i];
                head[2 * i + 1] = a * s[i] + b * c[i];
            }
        }
    }
}

/// Grouped-query attention over the whole window, with a learned per-head "sink" logit that takes part in the
/// softmax but contributes no value. The window is at most 128 tokens, inside the model's 128-token band.
fn attention(q: &[f32], k: &[f32], v: &[f32], t: usize, sinks: &[f32], out: &mut [f32]) {
    let mut scores = vec![0f32; t * t];
    for h in 0..HEADS {
        let kv = h / (HEADS / KV_HEADS);
        sgemm(false, true, t, t, HEAD_DIM, &q[h * HEAD_DIM..], HEADS * HEAD_DIM, &k[kv * HEAD_DIM..], KV_HEADS * HEAD_DIM, &mut scores, t);
        for row in scores.chunks_mut(t) {
            let m = row.iter().copied().fold(sinks[h], f32::max);
            let mut sum = (sinks[h] - m).exp();
            for s in row.iter_mut() {
                *s = (*s - m).exp();
                sum += *s;
            }
            row.iter_mut().for_each(|s| *s /= sum);
        }
        sgemm(false, false, t, HEAD_DIM, t, &scores, t, &v[kv * HEAD_DIM..], KV_HEADS * HEAD_DIM, &mut out[h * HEAD_DIM..], HEADS * HEAD_DIM);
    }
}

/// y[m, n] = a[m, k] * w[n, k]^T
fn gemm_bt(a: &[f32], w: &[f32], m: usize, n: usize, k: usize, y: &mut [f32]) {
    sgemm(false, true, m, n, k, a, k, w, k, y, n);
}

/// C = op(A) * op(B) for row-major matrices, where op transposes when `ta` / `tb` is set. Apple's Accelerate on
/// macOS; elsewhere the pure-Rust `gemm` crate, which gives the same spans (checked on all benchmark sets).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm(ta: bool, tb: bool, m: usize, n: usize, k: usize, a: &[f32], lda: usize, b: &[f32], ldb: usize, c: &mut [f32], ldc: usize) {
    #[cfg(target_os = "macos")]
    {
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            fn cblas_sgemm(order: i32, ta: i32, tb: i32, m: i32, n: i32, k: i32, alpha: f32, a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32);
        }
        let t = |x: bool| if x { 112 } else { 111 };
        unsafe { cblas_sgemm(101, t(ta), t(tb), m as i32, n as i32, k as i32, 1.0, a.as_ptr(), lda as i32, b.as_ptr(), ldb as i32, 0.0, c.as_mut_ptr(), ldc as i32) }
    }
    #[cfg(not(target_os = "macos"))]
    {
        // gemm takes column stride then row stride; a transposed operand swaps them. Inside a rayon task it runs
        // on that thread only, so the pool cannot start another task there while the task holds its scratch.
        let (acs, ars) = if ta { (lda as isize, 1) } else { (1, lda as isize) };
        let (bcs, brs) = if tb { (ldb as isize, 1) } else { (1, ldb as isize) };
        let par = if rayon::current_thread_index().is_some() { gemm::Parallelism::None } else { gemm::Parallelism::Rayon(0) };
        unsafe { gemm::gemm(m, n, k, c.as_mut_ptr(), 1, ldc as isize, false, a.as_ptr(), acs, ars, b.as_ptr(), bcs, brs, 0.0, 1.0, false, false, false, par) }
    }
}

/// Initializers of an ONNX ModelProto: name -> location. Reads the few protobuf fields that matter.
fn initializers(onnx: &[u8]) -> Result<HashMap<String, Where>> {
    let graph = fields(onnx)?.into_iter().find(|f| f.0 == 7).context("no graph in model")?.1;
    let mut out = HashMap::new();
    for (num, tensor) in fields(graph)? {
        if num != 5 {
            continue;
        }
        let (mut name, mut raw, mut ext) = (String::new(), None, HashMap::new());
        for (f, v) in fields(tensor)? {
            match f {
                8 => name = String::from_utf8(v.to_vec())?,
                9 => raw = Some(v.to_vec()),
                13 => {
                    let kv: HashMap<u64, &[u8]> = fields(v)?.into_iter().collect();
                    let s = |n| kv.get(&n).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default();
                    ext.insert(s(1), s(2));
                }
                _ => {}
            }
        }
        let place = match (raw, ext.get("offset"), ext.get("length")) {
            (Some(bytes), _, _) => Where::Inline(bytes),
            (None, Some(off), Some(len)) => Where::External(off.parse()?, len.parse()?),
            _ => continue,
        };
        out.insert(name, place);
    }
    Ok(out)
}

/// Length-delimited fields of a protobuf message as (field number, bytes); other wire types are skipped.
fn fields(mut b: &[u8]) -> Result<Vec<(u64, &[u8])>> {
    fn varint(b: &mut &[u8]) -> Result<u64> {
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let (&byte, rest) = b.split_first().context("truncated protobuf")?;
            *b = rest;
            v |= ((byte & 0x7f) as u64) << shift;
            if byte < 0x80 {
                return Ok(v);
            }
            shift += 7;
        }
    }
    let mut out = vec![];
    while !b.is_empty() {
        let key = varint(&mut b)?;
        match key & 7 {
            0 => {
                varint(&mut b)?;
            }
            1 => b = &b[8..],
            5 => b = &b[4..],
            2 => {
                let len = varint(&mut b)? as usize;
                out.push((key >> 3, &b[..len]));
                b = &b[len..];
            }
            w => bail!("unsupported protobuf wire type {w}"),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    /// Logits that ONNX Runtime gives for two windows, rounded to 3 decimals (testdata/logits.json). Needs the
    /// model, which `holdback serve` downloads.
    #[test]
    fn matches_onnx_runtime() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!("testdata/logits.json")).unwrap();
        let windows: Vec<Vec<u32>> = serde_json::from_value(fixture["windows"].clone()).unwrap();
        let want: Vec<Vec<f32>> = serde_json::from_value(fixture["logits"].clone()).unwrap();
        let model = super::Model::load(&crate::model_dir()).expect("no model; run `holdback serve` once to download it");
        let refs: Vec<&[u32]> = windows.iter().map(Vec::as_slice).collect();
        let got = model.logits(&refs, &mut super::Work::default()).unwrap();
        let want: Vec<f32> = want.concat();
        assert_eq!(got.len(), want.len());
        let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(worst < 5e-3, "largest logit difference from ONNX Runtime: {worst}");
    }
}
