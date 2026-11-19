//! Structure-aware refinement of detector spans.
//!
//! The detector finds private values but replaces too much in code and agent traffic: whole connection
//! strings, URLs, dates, IDs. Here we (1) protect tokens an agent needs and that identify nobody, cutting
//! them out of detector spans, and (2) add precise spans for secrets the detector misses, replacing only the
//! secret part of a URL, connection string, config line or path. All offsets are byte offsets.
use crate::detect::Span;
use fancy_regex::Regex;
use std::sync::LazyLock;

fn re(p: &str) -> Regex {
    Regex::new(p).expect("valid pattern")
}

fn all(patterns: &[&str]) -> Vec<Regex> {
    patterns.iter().map(|p| re(p)).collect()
}

/// Tokens that are kept. Dates are kept unless they follow a birth keyword, and opaque IDs unless they are
/// assigned to something named like a secret.
static DATES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    all(&[
        r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?\b",
        r"\b\d{4}年\d{1,2}月\d{1,2}日",
        r"\b(?:(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),? )?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* {1,2}\d{1,2}(?:,? \d{2}:\d{2}(?::\d{2})?(?: [A-Z]{2,5})?)?(?:,? {1,2}\d{4})?\b",
        r"\b\d{2}/(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/\d{4}:\d{2}:\d{2}:\d{2}",
    ])
});
static OPAQUE_IDS: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    all(&[
        r"(?i)\b(?:[a-z]+_)?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", // also local_<uuid>
        // object ids of the Anthropic and OpenAI APIs; they look like tokens and name nothing private
        r"\b(?:toolu|srvtoolu|msg|req|call|fc|rs|resp|run|thread|asst|file|batch|msgbatch|sesn|cse|env)_[A-Za-z0-9]{8,}\b|\bchatcmpl-[A-Za-z0-9]{8,}\b",
        r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b",
    ])
});
static STRUCTURE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    all(&[
        r"\b(?!\d{3}\.\d{2}\.\d{4}\b)v?\d+\.\d+\.\d+(?:[-+][\w.]+)?\b", // a version; 123.45.6789 is a social security number
        r"\b(?:10|127)(?:\.\d{1,3}){3}\b|\b192\.168(?:\.\d{1,3}){2}\b|\b172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}\b",
        r"\b(?:us|eu|ap|sa|ca|me|af)-(?:north|south|east|west|central|northeast|southeast)-\d\b",
        r"#\d{2,10}\b",
        r"\b(?:usr|acct|cus|ord|req|evt|inv)_[A-Za-z0-9]{3,}\b",
        r#"(?m)(?<=:)\d{2,5}(?=[/"'\s]|$)"#,
        r"\b\d{2,5}:\d{2,5}\b",
        r#""(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS) [^"\s?]+"#,
        r#"(?:/Users/|/home/|C:\\Users\\)[^/\\\s"']+([/\\][^\s"']+)"#,
        r"(?m)^([dlcbps-][rwxsStT-]{9}[@+.]?\s+\d+)\s+\S+\s+\S+\s+(\d+)\s", // ls -l: mode and link count, size
        r"(?i)(?<=\bport )\d{2,5}\b",
        r"(?i)\b(?:no-?reply|do-?not-?reply)@[a-z0-9-]+(?:\.[a-z0-9-]+)+\b", // an address nobody reads
        r"\bClaude (?:Code|Opus|Sonnet|Haiku|Fable)\b(?: \d+(?:\.\d+)*)?", // the agent's own name, as in a commit trailer
        r"(?m)^\s*[-*] ([a-z][a-z0-9_-]*):(?= )", // the key of a definition list, as in a list of skills or commands
        // a checksum: hex after a word that says so, as in a lockfile, and the algorithm in front of it
        r"(?i)(?:checksum|sha(?:1|256|384|512)|md5|hash|digest|integrity)\W{1,12}([0-9a-f]{32,128})\b",
        r"(?i)\b(?:sha(?:1|224|256|384|512)|md5):[0-9a-f]{32,128}\b", // uv.lock hashes, image digests
        // a Subresource Integrity hash, as npm and yarn lockfiles and HTML integrity attributes write it
        r"\bsha(?:1|256|384|512)-[A-Za-z0-9+/]{27,}={0,2}",
        // a line of a Python wheel's RECORD file: path, hash and size
        r"\bsha(?:1|224|256|384|512)=[A-Za-z0-9_-]{27,}=*",
        r"(?m)^([^,\s][^,\n]*),sha(?:1|224|256|384|512)=[A-Za-z0-9_-]+=*,(\d*)$",
        // a package with its version after @, as Homebrew names formulae (openssl@3); an email has a domain there
        r"\b[a-z0-9][a-z0-9+_.-]*@\d+(?:\.\d+)*\b(?![.@\w])",
        // a path under a system directory; user names live under /Users, /home and FreeBSD's /usr/home
        r#"(?:^|[\s"'=(])(/(?:opt|usr(?!/home/)|etc|Library|Applications|System|nix|bin|sbin)(?:/[^\s"',;]+)+)"#,
        // seconds or milliseconds since 1970 under a key that names a time
        r#"(?i)"[\w-]*(?:time|timestamp|_at)":\s*(\d{9,13})\b"#,
        // a package name in a manifest or lockfile: `name = "x"`, or a line that is only a quoted name
        r#"(?m)^name = "([a-z0-9][a-z0-9_.-]*)""#,
        r#"(?m)^\s*"([a-z0-9][a-z0-9_.-]*)(?: [^"\n]*)?",?$"#,
        // ls -l: a file name without spaces, like the paths that are kept elsewhere
        r"(?m)^[dlcbps-][rwxsStT-]{9}[@+.]?\s+\d+\s+\S+\s+\S+\s+\d+\s+\w{3}\s+\d+\s+[\d:]+\s+(\S+)$",
        // a license named after the word license
        r"(?i)\blicen[sc]e\b[^\n]{0,16}?\b(MIT|ISC|0BSD|Zlib|Unlicense|Apache-2\.0|BSD-[23]-Clause|MPL-2\.0|L?GPL-[23]\.[01](?:-only|-or-later)?)\b",
        // owner/name@ref, as a workflow names an action; an email has a dot after the @ and no slash before it
        r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@(?:v?\d+(?:\.\d+)*|[A-Za-z][\w-]*|[0-9a-f]{7,40})\b(?!\.)",
    ])
});
static CONTEXT_ID: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?i)(?:订单|单号|运单|快递|工单|order|tracking|ticket|invoice|ref|#).{0,4}?([A-Z]{1,5}-?\d{2,14}(?:-\d{2,14})*)\b"));
/// An id that the text itself names as a task, job, session or request id.
static NAMED_ID: LazyLock<Regex> = LazyLock::new(|| {
    re(r#"(?i)\b(?:task|job|shell|run|session|request|trace|span|build|deploy)[ _-]?id\b["':= >]{1,4}([A-Za-z0-9_-]{6,40})|\bwith ID: ([a-z0-9]{6,16})\b"#)
});
static BIRTH: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)(?:dob|d\.o\.b|birth|born|出生|生日)[^\n]{0,16}$"));
static SECRET_NAME: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)(?:secret|token|passw|pwd|nonce|salt|credential|key)s?\W{1,8}$"));
const SURNAMES: &str = "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤";
static ZH_SPEAKER: LazyLock<Regex> = LazyLock::new(|| re(&format!(r"(?m)^([{SURNAMES}][\x{{4e00}}-\x{{9fff}}]{{1,2}})[：:]")));

static SECRET_RE: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    all(&[
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bASIA[0-9A-Z]{16}\b",
        r"\bghp_[A-Za-z0-9]{30,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{40,}\b",
        r"\bglpat-[A-Za-z0-9_-]{20,}\b",
        r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b",
        r"\b[rs]k_live_[A-Za-z0-9]{16,}\b",
        r"(?i)(?<=\bBearer )[A-Za-z0-9._~+/-]{16,}=*",
        r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
        r"\bAIza[0-9A-Za-z_-]{35}\b",
        r"\bnpm_[A-Za-z0-9]{30,}\b",
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----",
        r"(?<=hooks\.slack\.com/services/)[A-Za-z0-9/]+",
    ])
});
static URL_CRED: LazyLock<Regex> = LazyLock::new(|| re(r"(?<=://)([^/\s:@]+):([^@\s/]+)@"));
static URL_KEY_AT: LazyLock<Regex> = LazyLock::new(|| re(r"(?<=://)([0-9a-f]{16,})@"));
static QUERY_SECRET: LazyLock<Regex> = LazyLock::new(|| {
    re(r"(?i)[?&](?:code|token|access_token|refresh_token|id_token|key|api_key|apikey|secret|sig|signature|password|pwd|auth)=([^&\s#]+)")
});
const SECRET_KEY_NAME: &str = r"[A-Za-z0-9_.-]*(?:PASSWORD|PASSWD|PWD|SECRET|(?:(?<![A-Za-z])TOKEN|(?-i:(?<=[a-z])Token|(?<=[A-Z])TOKEN))(?![A-Za-z])|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|_KEY|AUTH(?![a-z]))[A-Za-z0-9_.-]*";
static CONFIG_SECRET: LazyLock<Regex> =
    LazyLock::new(|| re(&format!(r#"(?im)^\s*(?:export\s+)?(?:"?{SECRET_KEY_NAME}"?)\s*[:=]\s*["']?([^"'\s#,]{{6,}})"#)));
static JSON_SECRET: LazyLock<Regex> = LazyLock::new(|| re(&format!(r#"(?i)"{SECRET_KEY_NAME}"\s*:\s*"([^"]{{6,}})""#)));
static PATH_USER: LazyLock<Regex> = LazyLock::new(|| re(r#"(?:/Users/|/home/|C:\\Users\\)([^/\\\s"']+)"#));
static URL: LazyLock<Regex> = LazyLock::new(|| re(r#"(?i)\b[a-z][a-z0-9+.-]*://[^\s"'<>]+"#));
static FILE_PATH: LazyLock<Regex> = LazyLock::new(|| re(r#"(?<![\w:/.])~?/(?!/)[^\s"'`<>:]+|\b[A-Za-z]:\\[^\s"'`<>]+"#));
/// Field names sit where values sit, so the detector sometimes tags them. They are never private values.
const FIELD_WORDS: &[&str] = &["name", "email", "phone", "address", "user", "username", "date", "city", "id", "port"];
/// Lowercase words joined by - _ or . with no spaces: a build target, a package, a model or a folder name.
static JOINED_WORDS: LazyLock<Regex> = LazyLock::new(|| re(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)+$"));
/// A word that announces a secret somewhere in the text just before a value ("the password is ...").
static SECRET_NEAR: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)secret|token|passw|passphrase|pwd|credential|\bkey\b|\bpin\b"));
static FILE_NAME: LazyLock<Regex> = LazyLock::new(|| {
    re(r"^[\w.-]+\.(?:toml|lock|rs|md|json|ya?ml|txt|py|js|jsx|ts|tsx|sh|cfg|ini|conf|html|css|go|java|rb|c|h|cpp|hpp|sql|xml|csv|log|env|orig)$")
});
static CONSTANT: LazyLock<Regex> = LazyLock::new(|| re(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$"));
/// True for a name such as `aarch64-apple-darwin` or `privacy-filter-q4`: it has a plain word in it and no
/// long part that mixes letters and digits, as a random token would.
fn is_name(value: &str) -> bool {
    let parts: Vec<&str> = value.split(['-', '_', '.']).collect();
    let word = |p: &&str| p.len() >= 3 && p.chars().all(|c| c.is_ascii_lowercase());
    let random = |p: &&str| p.len() >= 8 && p.chars().any(|c| c.is_ascii_digit()) && p.chars().any(|c| c.is_ascii_lowercase());
    JOINED_WORDS.is_match(value).unwrap_or(false) && parts.iter().any(word) && !parts.iter().any(random)
}

/// The agents themselves. Alone, the name is the assistant; "Claude Dubois" is still a person.
const AGENT_NAMES: &[&str] = &["claude", "codex"];
/// `plugin:skill`, `image:tag`, `package:target`: lowercase names joined by a colon, with a hyphen or underscore
/// somewhere. A person found inside one is an identifier, unless a field name comes before the colon (`user:`).
static NAMESPACED: LazyLock<Regex> =
    LazyLock::new(|| re(r"(?<![\w@./:-])(?=[a-z0-9:]*[-_])([a-z0-9]+(?:[-_][a-z0-9]+)*):[a-z0-9]+(?:[-_][a-z0-9]+)*(?![\w@/-])"));
static HAS_CJK: LazyLock<Regex> = LazyLock::new(|| re(r"[\x{4e00}-\x{9fff}]"));
static HAS_ALNUM: LazyLock<Regex> = LazyLock::new(|| re(r"[A-Za-z0-9]"));
const STRIP: &[char] = &[' ', '\t', '\n', ':', '@', '/', '=', ',', ';', '.', '\'', '"', '(', ')', '[', ']', '<', '>', '#', '-', '_'];

type Range = (usize, usize);

fn group_spans(rx: &Regex, text: &str, group: usize, label: &str, out: &mut Vec<Span>) {
    for c in rx.captures_iter(text).flatten() {
        if let Some(m) = c.get(group) {
            out.push(Span { start: m.start(), end: m.end(), label: label.to_string() });
        }
    }
}

fn rule_spans(text: &str) -> Vec<Span> {
    let mut out = vec![];
    for rx in SECRET_RE.iter() {
        group_spans(rx, text, 0, "secret", &mut out);
    }
    group_spans(&URL_CRED, text, 2, "secret", &mut out);
    group_spans(&URL_KEY_AT, text, 1, "secret", &mut out);
    for rx in [&*QUERY_SECRET, &*CONFIG_SECRET, &*JSON_SECRET] {
        group_spans(rx, text, 1, "secret", &mut out);
    }
    for c in PATH_USER.captures_iter(text).flatten() {
        let m = c.get(1).unwrap();
        if !["Shared", "runner", "user", "ubuntu", "root"].contains(&m.as_str()) {
            out.push(Span { start: m.start(), end: m.end(), label: "private_person".into() });
        }
    }
    group_spans(&ZH_SPEAKER, text, 1, "private_person", &mut out);
    out
}

/// The 24 characters before byte offset `at`.
fn chars_before(text: &str, at: usize, n: usize) -> &str {
    let head = &text[..at];
    let start = head.char_indices().rev().nth(n - 1).map(|(i, _)| i).unwrap_or(0);
    &head[start..]
}

fn protected(text: &str, rules: &[Span]) -> Vec<Range> {
    let mut out = vec![];
    for (patterns, unless_after) in [(&*DATES, Some(&*BIRTH)), (&*OPAQUE_IDS, Some(&*SECRET_NAME)), (&*STRUCTURE, None)] {
        for rx in patterns {
            let groups = if rx.captures_len() > 1 { 1..rx.captures_len() } else { 0..1 };
            for c in rx.captures_iter(text).flatten() {
                for m in groups.clone().filter_map(|g| c.get(g)) {
                    if !unless_after.is_some_and(|u| u.is_match(chars_before(text, m.start(), 24)).unwrap_or(false)) {
                        out.push((m.start(), m.end()));
                    }
                }
            }
        }
    }
    for c in CONTEXT_ID.captures_iter(text).flatten().chain(NAMED_ID.captures_iter(text).flatten()) {
        let m = c.get(1).or_else(|| c.get(2)).unwrap();
        out.push((m.start(), m.end()));
    }
    // the name of a config key is kept even when the detector takes `NAME=value` as one secret
    for c in [&*CONFIG_SECRET, &*JSON_SECRET].into_iter().flat_map(|rx| rx.captures_iter(text).flatten()) {
        out.push((c.get(0).unwrap().start(), c.get(1).unwrap().start()));
    }
    for m in URL.find_iter(text).flatten() {
        let (url, base) = (m.as_str(), m.start());
        let scheme_end = url.find("://").unwrap() + 3;
        let cred_end = URL_CRED.find(url).ok().flatten().or_else(|| URL_KEY_AT.find(url).ok().flatten()).map(|c| c.end());
        let host_start = cred_end.unwrap_or(scheme_end);
        let q = url.find('?').unwrap_or(url.len());
        out.push((base, base + scheme_end));
        out.push((base + host_start, base + q));
    }
    out.retain(|&(s, e)| !rules.iter().any(|r| r.start < e && s < r.end));
    out
}

fn subtract(span: &Span, holes: &[Range]) -> Vec<Range> {
    let mut pieces = vec![(span.start, span.end)];
    for &(hs, he) in holes {
        let mut next = vec![];
        for (ps, pe) in pieces {
            if he <= ps || hs >= pe {
                next.push((ps, pe));
                continue;
            }
            if hs > ps {
                next.push((ps, hs));
            }
            if he < pe {
                next.push((he, pe));
            }
        }
        pieces = next;
    }
    pieces
}

/// Detector spans -> refined spans, non-overlapping and sorted.
pub fn refine(text: &str, spans: &[Span]) -> Vec<Span> {
    let rules = rule_spans(text);
    let holes = protected(text, &rules);
    let paths: Vec<Range> = FILE_PATH.find_iter(text).flatten().map(|m| (m.start(), m.end())).collect();
    let namespaced: Vec<Range> = NAMESPACED
        .captures_iter(text)
        .flatten()
        .filter(|c| !FIELD_WORDS.contains(&&c[1]))
        .map(|c| (c.get(0).unwrap().start(), c.get(0).unwrap().end()))
        .collect();
    let mut kept = vec![];
    for sp in spans {
        // a file path is not a URL; replacing it leaves an agent unable to name the files next to it
        if sp.label == "private_url" && paths.iter().any(|&(s, e)| s < sp.end && sp.start < e) {
            continue;
        }
        if matches!(sp.label.as_str(), "private_person" | "private_url") && namespaced.iter().any(|&(s, e)| s <= sp.start && sp.end <= e) {
            continue;
        }
        // a build target or a folder name is not a secret or an address, unless the text calls it a secret
        let value = &text[sp.start..sp.end];
        let named_secret = SECRET_NEAR.is_match(chars_before(text, sp.start, 32)).unwrap_or(false);
        if matches!(sp.label.as_str(), "secret" | "private_address") && is_name(value) && !named_secret {
            continue;
        }
        // a constant in code or a file name is not a person: `SWIGLU_ALPHA`, `Cargo.toml`; a mention after @ is a user,
        // even when it looks like a constant (`<@U02LIZ_MARCHETTI>` in Slack)
        let mention = text[..sp.start].ends_with('@');
        if sp.label == "private_person" && !mention && (CONSTANT.is_match(value).unwrap_or(false) || FILE_NAME.is_match(value).unwrap_or(false)) {
            continue;
        }
        // the same name taken for a person, when it has a digit and three parts or more:
        // `aarch64-apple-darwin`, not `maria-gonzalez` or the user name `m.gonzalez88`
        if sp.label == "private_person" && is_name(value) && value.contains(|c: char| c.is_ascii_digit()) && value.split(['-', '_', '.']).count() >= 3 {
            continue;
        }
        let pieces = subtract(sp, &holes);
        let left: usize = pieces.iter().map(|(s, e)| e - s).sum();
        if sp.label == "private_date" && left * 2 < sp.end - sp.start {
            continue; // the date itself is kept, and what is left around it is not a date
        }
        for (s, e) in pieces {
            let frag = &text[s..e];
            let stripped = frag.trim_matches(STRIP);
            // hosts and paths are kept, so what is left of a URL has nothing to replace when it is only a host,
            // a path or a word: `db.internal.example.io`, `owner/name`, `orders`. A part with `user:password@`
            // does, and so does an address made of digits, which is a client's IP address.
            let plain = stripped.chars().all(|c| c.is_ascii_alphanumeric() || "._/-".contains(c)) && stripped.contains(|c: char| c.is_ascii_alphabetic());
            if sp.label == "private_url" && plain {
                continue;
            }
            let cjk = HAS_CJK.is_match(stripped).unwrap_or(false);
            if !cjk && (stripped.chars().count() < 2 || !HAS_ALNUM.is_match(stripped).unwrap_or(false)) {
                continue;
            }
            let word = stripped.to_ascii_lowercase();
            if FIELD_WORDS.contains(&word.as_str()) || AGENT_NAMES.contains(&word.as_str()) {
                continue;
            }            let off = frag.find(stripped).unwrap_or(0);
            kept.push(Span { start: s + off, end: s + off + stripped.len(), label: sp.label.clone() });
        }
    }
    kept.extend(rules);
    kept.sort();
    let mut merged: Vec<Span> = vec![];
    for sp in kept {
        match merged.last_mut() {
            Some(last) if sp.start < last.end => last.end = last.end.max(sp.end),
            _ => merged.push(sp),
        }
    }
    merged
}

#[cfg(test)]
mod tests {
    use crate::detect::Span;

    /// Spans the detector produced on real tool output that could not be reproduced outside it. The rules must
    /// drop them and keep the ones that are real.
    #[test]
    fn drops_names_tagged_on_real_output() {
        let replaced = |text: &str, value: &str, label: &str| {
            let at = text.find(value).unwrap();
            let span = Span { start: at, end: at + value.len(), label: label.into() };
            super::refine(text, &[span]).iter().any(|s| s.start < at + value.len() && at < s.end)
        };
        let jobs = "success  build (macos-latest, aarch64-apple-darwin)\nsuccess  build (ubuntu-22.04, x86_64-unknown-linux-musl)";
        assert!(!replaced(jobs, "aarch64-apple-darwin", "private_address"));
        assert!(!replaced(jobs, "x86_64-unknown-linux-musl", "secret"));
        assert!(!replaced("the privacy model is in /Users/alex/.cache/tool/privacy-filter-q4", "privacy-filter-q4", "secret"));
        assert!(!replaced("- uses: actions/cache@v4\n- uses: dtolnay/rust-toolchain@stable", "actions/cache@v4", "secret"));
        assert!(!replaced("- uses: actions/cache@v4\n- uses: dtolnay/rust-toolchain@stable", "dtolnay/rust-toolchain@stable", "private_email"));
        assert!(!replaced("added marketplace from acme-labs/widget", "acme-labs/widget", "private_url"));
        assert!(!replaced(jobs, "aarch64-apple-darwin", "private_person"));
        assert!(replaced("ask maria-gonzalez about it", "maria-gonzalez", "private_person"));
        assert!(replaced("user m.gonzalez88 logged in", "m.gonzalez88", "private_person"));
        let lock = "[[package]]\nname = \"bstr\"\nversion = \"1.12.0\"\nchecksum = \"234113d19d0d7d613b40e86fb654acf958910802bcceab913a4f9e7cda03b1a4\"\ndependencies = [\n \"bytemuck_derive\",\n \"regex-automata 0.4.9\",\n]";
        assert!(!replaced(lock, "bstr", "private_person"));
        assert!(!replaced(lock, "234113d19d0d7d613b40e86fb654acf958910802bcceab913a4f9e7cda03b1a4", "secret"));
        assert!(!replaced(lock, "bytemuck_derive", "private_person"));
        assert!(!replaced("-rwxr-xr-x  1 root  wheel  196320 Aug 12 21:51 bputil", "bputil", "private_person"));
        assert!(!replaced("const SWIGLU_ALPHA: f32 = 1.702;", "SWIGLU_ALPHA", "private_person"));
        assert!(replaced("good call <@U02LIZ_MARCHETTI>, thanks", "U02LIZ_MARCHETTI", "private_person"));
        assert!(super::refine("tiktoken-rs = \"0.12.1\"\nmax_tokens = 100000\ntokenizer = \"o200k_base\"", &[]).is_empty());
        assert_eq!(super::refine("auth_token = \"hunter2hunter2\"\n\"accessToken\": \"hunter2hunter2\"\nGITHUB_TOKEN=hunter2hunter2", &[]).len(), 3);
        assert!(replaced("signing key 234113d19d0d7d613b40e86fb654acf958910802bcceab913a4f9e7cda03b1a4", "234113d19d0d7d613b40e86fb654acf958910802bcceab913a4f9e7cda03b1a4", "secret"));
        let record = "requests/__init__.py,sha256=Vn7I4lQ9VtgNnYxWQqzKqFWlYG4K5HfdG7rqLKZG8eA,5139\nrequests-2.32.3.dist-info/INSTALLER,sha256=zuuue4knoyJ-UwPPXg8fezS7VCrXJQrAP7zeNuwvFQg,4";
        assert!(!replaced(record, "Vn7I4lQ9VtgNnYxWQqzKqFWlYG4K5HfdG7rqLKZG8eA", "secret"));
        assert!(!replaced(record, "requests-2.32.3.dist-info/INSTALLER", "private_url"));
        assert!(!replaced(record, "5139", "account_number"));
        assert!(!replaced("\"full_name\": \"openssl@3\",\n\"versioned\": [\"python@3.12\"]", "openssl@3", "private_email"));
        assert!(!replaced("\"full_name\": \"openssl@3\",\n\"versioned\": [\"python@3.12\"]", "python@3.12", "private_email"));
        assert!(replaced("write to jane.doe@example.com today", "jane.doe@example.com", "private_email"));
        assert!(!replaced("\"cellar\": \"/opt/homebrew/Cellar\",", "Cellar", "private_person"));
        assert!(replaced("cd /home/jsmith/src/app", "jsmith", "private_person"));
        assert!(replaced("mail is in /var/mail/jsmith", "jsmith", "private_person"));
        assert!(replaced("cd /usr/home/jsmith/src", "jsmith", "private_person"));
        assert!(!replaced("\"installed_time\": 1794938741,", "1794938741", "private_phone"));
        assert!(replaced("\"phone\": 13800138000,", "13800138000", "private_phone"));
        let sri = "\"integrity\": \"sha512-vq24Bq3ym5HEQm2NKCr3yXDwjc7vTsEThRDnkp2DK9p1uqLR+DHurm/NOTo0KG7HYHU7eppKZj3MyqYuMBf62g==\",";
        assert!(!replaced(sri, "vq24Bq3ym5HEQm2NKCr3yXDwjc7vTsEThRDnkp2DK9p1uqLR+DHurm/NOTo0KG7HYHU7eppKZj3MyqYuMBf62g", "secret"));
        assert!(!replaced("<script src=\"x.js\" integrity=\"sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8wC\">", "oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K", "secret"));
        assert!(replaced("name = \"Maria Gonzalez\"", "Maria Gonzalez", "private_person"));
        assert!(replaced("SSN on file: 123.45.6789", "123.45.6789", "account_number"));
        assert!(!replaced("upgrade to version 12.4.1 today", "12.4.1", "account_number"));
        assert!(!replaced("nc: connect to db.internal.example.io port 5433 failed", "db.internal.example.io", "private_url"));
        assert!(!replaced("database orders is ready", "orders", "private_url"));
        let digest = "hash = \"sha256:0f3c9a7e5b1d2c4a6e8f0b1d3c5a7e9f1b3d5c7a9e0f2b4d6c8a0e1f3b5d7c9a\"";
        assert!(super::refine(digest, &[Span { start: digest.find("sha").unwrap(), end: digest.len() - 1, label: "secret".into() }]).is_empty());
        let host = "psql: could not connect to db.internal.example.io:5433/orders";
        assert!(super::refine(host, &[Span { start: host.find("db.").unwrap(), end: host.len(), label: "private_url".into() }]).is_empty());
        assert!(!replaced("src/lib.rs\nCargo.toml\nREADME.md", "Cargo.toml", "private_person"));
        assert!(replaced("signed off by j.smith", "j.smith", "private_person"));
        assert!(replaced("203.0.113.42 - - [14/Nov/2026:10:12:01] \"GET /health\"", "203.0.113.42", "private_url"));
        assert!(replaced("connect with app_admin:S3cr3t-Pa55w0rd!@db.internal.example.io", "app_admin:S3cr3t-Pa55w0rd!@db.internal.example.io", "private_url"));
        assert!(!replaced("- acme-tools:code-review: Review a change.", "acme-tools:code-review", "private_url"));
        assert!(!replaced("license: MIT", "MIT", "private_person"));
        assert!(!replaced("license = \"MIT OR Apache-2.0\"", "MIT", "private_person"));
        assert!(replaced("ask MIT about it", "MIT", "private_person"));
        // real ones stay
        assert!(replaced("login with the code a8f3k2zq-9dk2l1xx-zz81", "a8f3k2zq-9dk2l1xx-zz81", "secret"));
        assert!(replaced("the password is correct-horse-battery", "correct-horse-battery", "secret"));
        assert!(replaced("write to maria-gonzalez@gmail.com", "maria-gonzalez@gmail.com", "private_email"));
        assert!(replaced("ship to 221b-baker-street", "221b-baker-street", "private_address"));
        assert!(replaced("see https://internal.example.com/reset?token=abc123def456", "https://internal.example.com/reset?token=abc123def456", "private_url"));
    }

    /// Cases from real agent traffic, run through the real detector: each value in `kept` must survive and each
    /// value in `replaced` must go.
    #[test]
    fn keeps_what_an_agent_needs() {
        let mut detector = crate::detect::Detector::load(&crate::model_dir()).expect("no model; run `holdback serve` once to download it");
        let cases: &[(&str, &[&str], &[&str])] = &[
            (
                "- acme-tools:code-review: Review a change.\n- acme-tools:oss-ideas: Find a project idea.\n- yu-skills:clear-writing: Write emails in Yu's style.",
                &["acme-tools:code-review", "acme-tools:oss-ideas", "yu-skills:clear-writing"],
                &[],
            ),
            ("Deploy my-app:latest. The owner is user:jean-luc and Maria Gonzalez reviews.", &["my-app:latest"], &["jean-luc", "Maria Gonzalez"]),
            (
                "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nCo-Authored-By: Daniel Okafor <dokafor88@outlook.com>",
                &["Claude Opus 5.5", "noreply@anthropic.com"],
                &["Daniel Okafor", "dokafor88@outlook.com"],
            ),
            ("Reply to 12345+mgonzalez@users.noreply.github.com. Claude Dubois signed off.", &[], &["12345+mgonzalez@users.noreply.github.com", "Claude Dubois"]),
            (
                "Name: Priya Raman, email priya.raman@proton.me, order ORD-2026-0412 on port 5432. Ticket TKT-88213, invoice INV-2026-0412.",
                &["ORD-2026-0412", "5432", "TKT-88213", "INV-2026-0412"],
                &["Priya Raman", "priya.raman@proton.me"],
            ),
            ("Passport AB-1234567, account 6222020200112345678.", &[], &["AB-1234567", "6222020200112345678"]),
            (
                // the made-up key is written in two parts so that secret scanners do not take it for a real one
                concat!("DATABASE_URL=postgres://app_admin:S3cr3t-Pa55w0rd!@db.internal.acme.io:5432/orders\nSTRIPE_SECRET_KEY=sk_", "live_51HxQ2bEXAMPLEq7Wm3Tn8Vy\n"),
                &["STRIPE_SECRET_KEY=", "DATABASE_URL=", "db.internal.acme.io:5432/orders"],
                &[concat!("sk_", "live_51HxQ2bEXAMPLEq7Wm3Tn8Vy"), "S3cr3t-Pa55w0rd!"],
            ),
            (
                "Summoned by a doc comment (turn headed `[Artifact comment sent to Claude]`): answer under that root. Ask Claude Dubois if unsure.",
                &["to Claude]"],
                &["Claude Dubois"],
            ),
            (
                r#"[{"sessionId": "local_1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d", "title": "New session"}] tool_use_id: toolu_01CbaUpeqjCd9Rn2LQ35hHQX completed"#,
                &["local_1a2b3c4d", "0e1f2a3b4c5d", "toolu_01CbaUpeqjCd9Rn2LQ35hHQX"],
                &[],
            ),
            (
                "Command running in background with ID: bc927kf5m. <task-id>bq814mz2x</task-id> api_token = msg_01CbaUpeqjCd9Rn2LQ35hHQX",
                &["bc927kf5m", "bq814mz2x"],
                &["msg_01CbaUpeqjCd9Rn2LQ35hHQX"],
            ),
            (
                "论文润色、实验组织 and 论文表格.\n- dataviz: Use this skill whenever you are about to create a chart.\n- loop: Run a prompt on a recurring interval.",
                &["dataviz", "loop"],
                &[],
            ),
        ];
        // the detector's window cache changes no output: a text seen after a similar one gets the same spans
        // as it gets from a detector that has seen nothing
        let base: String = (0..400).map(|i| format!("line {i}: Maria Gonzalez wrote to dokafor88@outlook.com about order {i}\n")).collect();
        let edited = base.replacen("line 200: Maria Gonzalez", "line 200: Daniel Okafor and Priya Raman", 1);
        let mut fresh = crate::detect::Detector::load(&crate::model_dir()).unwrap();
        detector.detect(&base).unwrap();
        assert_eq!(detector.detect(&edited).unwrap(), fresh.detect(&edited).unwrap());
        for (text, kept, replaced) in cases {
            let spans = super::refine(text, &detector.detect(text).unwrap());
            let hit = |v: &str| {
                let at = text.find(v).unwrap();
                spans.iter().any(|s| s.start < at + v.len() && at < s.end)
            };
            for v in *kept {
                assert!(!hit(v), "{v:?} was replaced in {text:?}");
            }
            for v in *replaced {
                assert!(hit(v), "{v:?} was kept in {text:?}");
            }
        }
    }

    /// A one-line JSON response has no line break for a window to end at. No name or email in it may be cut in two
    /// by a window edge and replaced only in part, by the Privacy Filter or, with HOLDBACK_STUDENT_DIR, the student.
    #[test]
    fn keeps_values_whole_at_window_edges() {
        let (first, last) = (["Maria", "Daniel", "Priya", "Hannah", "Oliver", "Sofia", "Lucas", "Aisha"], ["Gonzalez", "Okafor", "Raman", "Becker", "Clarke", "Rossi", "Novak", "Dubois"]);
        let mut values = vec![];
        let records: Vec<String> = (0..200)
            .map(|i| {
                let (f, l) = (first[i % 8], last[i / 8 % 8]);
                let (name, email) = (format!("{f} {l}"), format!("{}.{}{i}@example.com", f.to_lowercase(), l.to_lowercase()));
                let record = format!(r#"{{"id":{},"status":"active","name":"{name}","email":"{email}","region":"eu-west-1"}}"#, 10_000 + i);
                values.extend([name, email]);
                record
            })
            .collect();
        let text = format!(r#"{{"users":[{}],"next":null}}"#, records.join(","));
        let mut detectors = vec![crate::detect::Detector::load(&crate::model_dir()).unwrap()];
        if let Ok(dir) = std::env::var("HOLDBACK_STUDENT_DIR") {
            detectors.push(crate::detect::Detector::Student(Box::new(crate::student::Student::load(std::path::Path::new(&dir)).unwrap())));
        }
        for mut detector in detectors {
            let spans = detector.detect(&text).unwrap();
            let (mut at, mut cut) = (0, vec![]);
            for v in &values {
                let start = at + text[at..].find(v.as_str()).unwrap();
                at = start + v.len();
                let inside: Vec<usize> = (start..at).filter(|&i| !b" @.".contains(&text.as_bytes()[i])).collect();
                let hit = inside.iter().filter(|&&i| spans.iter().any(|s| s.start <= i && i < s.end)).count();
                if 0 < hit && hit < inside.len() {
                    cut.push(v);
                }
            }
            assert!(cut.is_empty(), "{} replaced only part of {cut:?}", detector.name());
        }
    }
}
