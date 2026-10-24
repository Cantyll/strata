//! Leitor de referência do formato `.strata` (docs/02-especificacao-formato.md).
//!
//! Implementado de novo a partir da especificação, sem traduzir o leitor em JS, para que as duas
//! leituras possam ser comparadas no fuzzing (tests/fuzz-leitor.mjs --rust). Trata todo arquivo
//! como hostil: valida o cabeçalho inteiro pelas regras de A.12 antes de tocar no corpo, só aloca
//! o que o cabeçalho declara (e o total até `limite_saida`), confere o SHA-256 de cada stream antes
//! de descomprimir e o de cada tensor depois (A.13). Toda recusa é um `Erro`; nenhum `unsafe`.

pub mod dist;
pub mod residual;

use base64::Engine;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

pub const MAGIC: [u8; 7] = *b"STRATA\0";
pub const LIMITE_CABECALHO: u64 = 100 * 1024 * 1024;
const PROF_JSON: usize = 64;
const MAX_SAFE: u64 = (1 << 53) - 1;

#[derive(Debug, Clone)]
pub struct Erro(pub String);
impl std::fmt::Display for Erro {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Erro {}
pub type R<T> = Result<T, Erro>;
fn falha<T>(m: impl Into<String>) -> R<T> {
    Err(Erro(m.into()))
}

#[derive(Default, Clone)]
pub struct Opcoes {
    pub exigir_integridade: bool,
    /// SHA-256 (hex) de MAGIC + HL + cabeçalho, publicado fora do arquivo
    pub id_esperado: Option<String>,
    /// teto de bytes de saída; padrão max(1 GiB, 64 × tamanho do arquivo)
    pub limite_saida: Option<u64>,
    /// threads de `decode` (0 = todos os núcleos)
    pub threads: usize,
}

#[derive(Clone, Copy)]
struct Layout {
    bits: u32,
    e: u32,
    m: u32,
    bytes: u64,
}
fn layout(dtype: &str) -> Option<Layout> {
    let (bits, e, m, bytes) = match dtype {
        "BF16" => (16, 8, 7, 2),
        "F16" => (16, 5, 10, 2),
        "F32" => (32, 8, 23, 4),
        "F8_E4M3" => (8, 4, 3, 1),
        "F8_E5M2" => (8, 5, 2, 1),
        _ => return None,
    };
    Some(Layout { bits, e, m, bytes })
}
fn dtype_bytes(dtype: &str) -> Option<u64> {
    Some(match dtype {
        "BF16" | "F16" | "I16" | "U16" => 2,
        "F32" | "I32" | "U32" => 4,
        "F64" | "I64" | "U64" => 8,
        "I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2" => 1,
        _ => return None,
    })
}
fn dtype_ok(d: &str) -> bool {
    let b = d.as_bytes();
    !b.is_empty() && b.len() <= 32 && b[0].is_ascii_uppercase() && b.iter().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == b'_')
}
fn hex64(v: &Value) -> Option<&str> {
    match v {
        Value::String(s) if s.len() == 64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) => Some(s),
        _ => None,
    }
}
pub fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

/* ---------------- JSON ---------------- */

fn json_estrito(bytes: &[u8], onde: &str) -> R<Value> {
    // serde_json já recusa UTF-8 inválido, BOM, surrogate solto e números fora do alcance de um f64
    let v: Value = serde_json::from_slice(bytes).or_else(|_| falha(format!("{onde}: JSON ou UTF-8 inválido")))?;
    fn anda(x: &Value, d: usize, onde: &str) -> R<()> {
        if d > PROF_JSON {
            return falha(format!("{onde}: JSON com mais de {PROF_JSON} níveis"));
        }
        match x {
            Value::Array(a) => a.iter().try_for_each(|y| anda(y, d + 1, onde)),
            Value::Object(o) => o.values().try_for_each(|y| anda(y, d + 1, onde)),
            _ => Ok(()),
        }
    }
    anda(&v, 0, onde)?;
    Ok(v)
}
/// Inteiro não negativo até 2^53 − 1 (o maior que um double guarda exato), escrito como inteiro ou
/// como número com parte fracionária zero: as duas leituras (JS e Rust) aceitam o mesmo conjunto.
fn inteiro(v: Option<&Value>, onde: &str) -> R<u64> {
    if let Some(Value::Number(n)) = v {
        if let Some(u) = n.as_u64() {
            if u <= MAX_SAFE {
                return Ok(u);
            }
        } else if let Some(f) = n.as_f64() {
            if f.fract() == 0.0 && f >= 0.0 && f <= MAX_SAFE as f64 {
                return Ok(f as u64);
            }
        }
    }
    falha(format!("{onde}: inteiro inválido"))
}
fn obj<'a>(v: Option<&'a Value>, onde: &str) -> R<&'a Map<String, Value>> {
    match v {
        Some(Value::Object(o)) => Ok(o),
        _ => falha(format!("{onde} inválido")),
    }
}
fn so_chaves(o: &Map<String, Value>, exigidas: &[&str], onde: &str) -> R<()> {
    for k in o.keys() {
        if !exigidas.contains(&k.as_str()) {
            return falha(format!("{onde}: chave desconhecida {k:?}"));
        }
    }
    for k in exigidas {
        if !o.contains_key(*k) {
            return falha(format!("{onde}: falta a chave {k}"));
        }
    }
    Ok(())
}
fn texto(v: Option<&Value>) -> Option<&str> {
    match v {
        Some(Value::String(s)) => Some(s),
        _ => None,
    }
}
fn shape(v: Option<&Value>, onde: &str) -> R<(Vec<u64>, u64)> {
    let a = match v {
        Some(Value::Array(a)) => a,
        _ => return falha(format!("{onde}: shape inválido")),
    };
    let (mut dims, mut el) = (Vec::with_capacity(a.len()), 1u64);
    for d in a {
        let d = inteiro(Some(d), &format!("{onde}.shape"))?;
        el = match el.checked_mul(d) {
            Some(x) if x <= MAX_SAFE => x,
            _ => return falha(format!("{onde}: shape grande demais")),
        };
        dims.push(d);
    }
    Ok((dims, el))
}

/* ---------------- safetensors ---------------- */

#[derive(Clone)]
pub struct EntradaSt {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub begin: u64,
    pub end: u64,
}
/// Cabeçalho de safetensors com as regras do próprio safetensors; `dados` = tamanho da área de
/// dados quando conhecido. Devolve as entradas em ordem de offset e o total.
pub fn valida_safetensors(hb: &[u8], dados: Option<u64>) -> R<(Vec<EntradaSt>, u64)> {
    if hb.len() as u64 > LIMITE_CABECALHO {
        return falha("safetensors: cabeçalho grande demais");
    }
    if hb.first() != Some(&b'{') {
        return falha("safetensors: cabeçalho não começa com {");
    }
    let h = json_estrito(hb, "safetensors")?;
    let h = obj(Some(&h), "safetensors: cabeçalho")?;
    let mut es = Vec::new();
    for (name, v) in h {
        let onde = format!("safetensors {name:?}");
        if name == "__metadata__" {
            match v {
                Value::Object(m) if m.values().all(|x| x.is_string()) => continue,
                _ => return falha("safetensors: __metadata__ inválido"),
            }
        }
        let o = obj(Some(v), &onde)?;
        so_chaves(o, &["dtype", "shape", "data_offsets"], &onde)?;
        let dtype = match texto(o.get("dtype")) {
            Some(d) if dtype_ok(d) => d.to_string(),
            _ => return falha(format!("{onde}: dtype inválido")),
        };
        let (dims, el) = shape(o.get("shape"), &onde)?;
        let off = match o.get("data_offsets") {
            Some(Value::Array(a)) if a.len() == 2 => a,
            _ => return falha(format!("{onde}: data_offsets inválido")),
        };
        let (begin, end) = (inteiro(off.first(), &onde)?, inteiro(off.get(1), &onde)?);
        if end < begin {
            return falha(format!("{onde}: data_offsets invertido"));
        }
        if let Some(sz) = dtype_bytes(&dtype) {
            if end - begin != el * sz {
                return falha(format!("{onde}: tamanho não bate com shape × dtype"));
            }
        }
        es.push(EntradaSt { name: name.clone(), dtype, shape: dims, begin, end });
    }
    es.sort_by_key(|e| (e.begin, e.end));
    let mut pos = 0;
    for e in &es {
        if e.begin != pos {
            return falha("safetensors: data_offsets com buraco ou sobreposição");
        }
        pos = e.end;
    }
    if dados.is_some_and(|d| d != pos) {
        return falha("safetensors: área de dados não bate com os data_offsets");
    }
    Ok((es, pos))
}

/* ---------------- cabeçalho .strata ---------------- */

#[derive(Clone)]
struct Stream {
    o: u64,
    l: u64,
    r: u64,
    c: Cod,
    h: Option<String>,
}
#[derive(Clone, Copy, PartialEq)]
enum Cod {
    Raw,
    Deflate,
    Rans,
    Fecho,
}
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
    n: u64,
    lay: Option<Layout>,
    sha256: Option<String>,
    raw: Option<Stream>,
    sign: Option<Stream>,
    exp: Option<Stream>,
    planos: Vec<Option<Stream>>,
    res: Option<Res>,
    /// tensor que não é do safetensors (só existe para remontar o GGUF, 0.3)
    pub extra: bool,
}
/// Tensor em modo residual (0.3, A.14): estrato 0 = k-quant, um plano por nível, fecho.
struct Res {
    g: residual::Grade,
    permuta: Option<usize>,
    q: Option<Stream>,
    niveis: Vec<Option<Stream>>,
    fecho: Option<Stream>,
}
#[derive(Clone, Copy, PartialEq)]
enum ComoGguf {
    Cabecalho,
    Q,
    F32,
    Bruto,
}
/// Como remontar o GGUF do estrato 0 (0.3): partes na ordem do arquivo.
struct SaidaGguf {
    alinhamento: u64,
    sha256: Option<String>,
    partes: Vec<(usize, ComoGguf)>,
}
pub struct Arquivo {
    pub versao: u8,
    pub base: u64,
    pub fim: u64,
    pub m: u32,
    pub integridade: bool,
    pub tensores: Vec<Tensor>,
    pub fonte: Option<Vec<u8>>,
    pub fonte_sha256: Option<String>,
    /// níveis do modo residual (0.3); 0 sem tensores residuais
    pub niveis: u32,
    gguf: Option<SaidaGguf>,
}

/// Valida o cabeçalho (A.12). Não decodifica nada do corpo.
pub fn abre(file: &[u8], op: &Opcoes) -> R<Arquivo> {
    if file.len() < 12 {
        return falha("arquivo curto demais");
    }
    if file[..7] != MAGIC {
        return falha("assinatura .strata inválida");
    }
    let versao = file[7];
    if !(1..=3).contains(&versao) {
        return falha(format!("versão .strata desconhecida: {versao}"));
    }
    let hl = u32::from_le_bytes(file[8..12].try_into().unwrap()) as u64;
    if !(2..=LIMITE_CABECALHO).contains(&hl) {
        return falha(format!("tamanho do cabeçalho inválido: {hl}"));
    }
    let base = 12 + hl;
    if base > file.len() as u64 {
        return falha("cabeçalho incompleto");
    }
    let hb = &file[..base as usize];
    if let Some(id) = &op.id_esperado {
        if &sha256_hex(hb) != id {
            return falha("id do cabeçalho não confere");
        }
    }
    let hv = json_estrito(&hb[12..], "cabeçalho")?;
    let h = obj(Some(&hv), "cabeçalho")?;
    for k in ["format", "ordem", "estratos", "tensors", "streams", "fonte", "pai"] {
        if !h.contains_key(k) {
            return falha(format!("cabeçalho: falta a chave {k}"));
        }
    }
    if texto(h.get("format")) != Some(["strata/0.1", "strata/0.2", "strata/0.3"][versao as usize - 1]) {
        return falha("format não bate com a versão do arquivo");
    }
    if texto(h.get("ordem")) != Some("estrato-maior") {
        return falha("ordem desconhecida");
    }
    let integ = match h.get("integridade") {
        None | Some(Value::Null) => false,
        Some(v) => {
            let o = obj(Some(v), "integridade")?;
            so_chaves(o, &["alg"], "integridade")?;
            if texto(o.get("alg")) != Some("sha256") {
                return falha("integridade: algoritmo desconhecido");
            }
            true
        }
    };
    if op.exigir_integridade && !integ {
        return falha("arquivo sem hashes de integridade");
    }
    if !matches!(h.get("pai"), Some(Value::Null) | Some(Value::Object(_))) {
        return falha("pai inválido");
    }
    let niveis = if versao >= 3 {
        match inteiro(h.get("niveis"), "niveis")? {
            k @ 0..=8 => k as u32,
            _ => return falha("niveis: no máximo 8"),
        }
    } else {
        0
    };
    let tv = match h.get("tensors") {
        Some(Value::Array(a)) => a,
        _ => return falha("tensors inválido"),
    };
    let limite = op.limite_saida.unwrap_or_else(|| (1u64 << 30).max(64 * file.len() as u64));
    let ch_base: &[&str] = if integ { &["name", "dtype", "shape", "n", "bytes", "sha256"] } else { &["name", "dtype", "shape", "n", "bytes"] };
    let (mut nomes, mut m_max, mut saida, mut tem_res) = (HashSet::new(), 0u32, 0u64, false);
    let mut ts = Vec::with_capacity(tv.len());
    for (i, t) in tv.iter().enumerate() {
        let onde = format!("tensor {i}");
        let o = obj(Some(t), &onde)?;
        let mut ch_t = ch_base.to_vec();
        if versao >= 3 {
            ch_t.extend(["residual", "extra"].into_iter().filter(|k| o.contains_key(*k)));
        }
        so_chaves(o, &ch_t, &onde)?;
        let name = match texto(o.get("name")) {
            Some(n) if nomes.insert(n.to_string()) => n.to_string(),
            _ => return falha(format!("{onde}: nome inválido ou repetido")),
        };
        let dtype = match texto(o.get("dtype")) {
            Some(d) if dtype_ok(d) => d.to_string(),
            _ => return falha(format!("{onde}: dtype inválido")),
        };
        let (dims, el) = shape(o.get("shape"), &onde)?;
        let bytes = inteiro(o.get("bytes"), &format!("{onde}.bytes"))?;
        if dtype_bytes(&dtype).is_some_and(|sz| bytes != el * sz) {
            return falha(format!("{onde}: bytes não bate com shape × dtype"));
        }
        // n = null num dtype de ponto flutuante: tensor guardado cru, sem estratos (A.12)
        let lay = if matches!(o.get("n"), Some(Value::Null)) { None } else { layout(&dtype) };
        let n_ok = match (lay, o.get("n")) {
            (Some(_), Some(v)) => inteiro(Some(v), "n").ok() == Some(el),
            (None, Some(Value::Null)) => true,
            _ => false,
        };
        if !n_ok {
            return falha(format!("{onde}: n inválido"));
        }
        let sha256 = if integ {
            match hex64(o.get("sha256").unwrap()) {
                Some(s) => Some(s.to_string()),
                None => return falha(format!("{onde}: sha256 inválido")),
            }
        } else {
            None
        };
        let res = match o.get("residual") {
            None => None,
            Some(rv) => {
                let r = obj(Some(rv), &format!("{onde}.residual"))?;
                so_chaves(r, &["grade", "permuta"], &format!("{onde}.residual"))?;
                let Some(g) = texto(r.get("grade")).and_then(residual::Grade::de_nome) else {
                    return falha(format!("{onde}: grade desconhecida"));
                };
                if dtype != "BF16" || lay.is_none() || dims.len() != 2 || el == 0 || dims[1] % 256 != 0 {
                    return falha(format!("{onde}: modo residual só para BF16 2-D com linhas múltiplas de 256"));
                }
                let permuta = match r.get("permuta") {
                    Some(Value::Null) => None,
                    v => match inteiro(v, &format!("{onde}.permuta"))? {
                        h if h >= 1 && dims[0] % (2 * h) == 0 => Some(h as usize),
                        _ => return falha(format!("{onde}: permuta inválida")),
                    },
                };
                tem_res = true;
                Some(Res { g, permuta, q: None, niveis: vec![None; niveis as usize], fecho: None })
            }
        };
        let extra = match o.get("extra") {
            None => false,
            Some(v) if v.as_str() == Some("gguf") && lay.is_none() => true,
            _ => return falha(format!("{onde}: extra inválido")),
        };
        if let (Some(l), None) = (lay, &res) {
            m_max = m_max.max(l.m);
        }
        saida += bytes;
        if saida > limite {
            return falha(format!("o cabeçalho declara mais de {limite} bytes de saída (limite_saida)"));
        }
        let planos = vec![None; if res.is_some() { 0 } else { lay.map_or(0, |l| l.m as usize) }];
        ts.push(Tensor { name, dtype, shape: dims, bytes, n: el, lay, sha256, raw: None, sign: None, exp: None, planos, res, extra });
    }
    let estratos = (m_max as u64 + 1).max(if tem_res { niveis as u64 + 2 } else { 0 });
    if inteiro(h.get("estratos"), "estratos").ok() != Some(estratos) {
        return falha("estratos não bate com os tipos dos tensores");
    }
    let sv = match h.get("streams") {
        Some(Value::Array(a)) => a,
        _ => return falha("streams inválido"),
    };
    let ch_s: &[&str] = if integ { &["t", "s", "p", "o", "l", "r", "c", "h"] } else { &["t", "s", "p", "o", "l", "r", "c"] };
    let esperado: u64 = ts.iter().map(|t| if t.res.is_some() { 2 + niveis as u64 } else { t.lay.map_or(1, |l| 2 + l.m as u64) }).sum();
    let (mut off, mut ant, nt) = (0u64, None::<u64>, ts.len() as u64);
    for (i, e) in sv.iter().enumerate() {
        let onde = format!("stream {i}");
        let o = obj(Some(e), &onde)?;
        so_chaves(o, ch_s, &onde)?;
        let ti = inteiro(o.get("t"), &format!("{onde}.t"))?;
        if ti >= nt {
            return falha(format!("{onde}: tensor fora da tabela"));
        }
        let s_decl = inteiro(o.get("s"), &format!("{onde}.s"))?;
        let t = &ts[ti as usize];
        let p = match texto(o.get("p")) {
            Some(p) => p,
            None => return falha(format!("{onde}: parte inválida")),
        };
        // (estrato, posição dentro do estrato, tamanho cru esperado); plano j = índice j
        let bits_bytes = t.n.div_ceil(8);
        let (s, ordem, r, plano) = match (t.lay, p) {
            _ if t.res.is_some() => {
                let g = t.res.as_ref().unwrap().g;
                match p {
                    "q" => (0, 0, t.n / 256 * g.bytes_bloco() as u64, None),
                    "f" => (niveis as u64 + 1, 0, 2 * t.n, None),
                    _ => match p.strip_prefix('n').and_then(|j| plano_j(&format!("m{j}"))) {
                        Some(j) if (1..=niveis as u64).contains(&j) => (j, 0, t.n / 8, Some(j as usize - 1)),
                        _ => return falha(format!("{onde}: parte inválida: {p:?}")),
                    },
                }
            }
            (None, "raw") => (0u64, 0u64, t.bytes, None),
            (None, _) => return falha(format!("{onde}: parte inválida para um tensor cru")),
            (Some(_), "sign") => (0, 0, bits_bytes, None),
            (Some(_), "exp") => (0, 1, t.n, None),
            (Some(l), _) => match plano_j(p) {
                Some(j) if j <= l.m as u64 => (j, 0, bits_bytes, Some(j as usize - 1)),
                _ => return falha(format!("{onde}: parte inválida: {p:?}")),
            },
        };
        if s_decl != s {
            return falha(format!("{onde}: estrato não bate com a parte"));
        }
        let chave = (s * nt + ti) * 2 + ordem;
        if ant.is_some_and(|a| chave <= a) {
            return falha(format!("{onde}: fora de ordem ou repetido"));
        }
        ant = Some(chave);
        if inteiro(o.get("o"), &format!("{onde}.o"))? != off {
            return falha(format!("{onde}: offset fora de sequência"));
        }
        let l = inteiro(o.get("l"), &format!("{onde}.l"))?;
        if inteiro(o.get("r"), &format!("{onde}.r"))? != r {
            return falha(format!("{onde}: tamanho cru não bate com o tensor"));
        }
        let c = match texto(o.get("c")) {
            Some("fecho") if p == "f" && t.res.is_some() => Cod::Fecho,
            _ if p == "f" && t.res.is_some() => return falha(format!("{onde}: o fecho só tem a codificação fecho")),
            Some("raw") if l == r => Cod::Raw,
            Some("raw") => return falha(format!("{onde}: stream cru com tamanho errado")),
            Some("rans") if p == "exp" && versao >= 2 => Cod::Rans,
            Some("rans") => return falha(format!("{onde}: rANS fora do expoente")),
            Some("deflate") => Cod::Deflate,
            _ => return falha(format!("{onde}: codificação desconhecida")),
        };
        let hh = if integ {
            match hex64(o.get("h").unwrap()) {
                Some(x) => Some(x.to_string()),
                None => return falha(format!("{onde}: hash inválido")),
            }
        } else {
            None
        };
        off += l;
        if off > MAX_SAFE {
            return falha("corpo grande demais");
        }
        let st = Some(Stream { o: off - l, l, r, c, h: hh });
        let t = &mut ts[ti as usize];
        if let Some(rs) = &mut t.res {
            match (p, plano) {
                ("q", _) => rs.q = st,
                ("f", _) => rs.fecho = st,
                (_, Some(j)) => rs.niveis[j] = st,
                _ => unreachable!(),
            }
            continue;
        }
        match (p, plano) {
            ("raw", _) => t.raw = st,
            ("sign", _) => t.sign = st,
            ("exp", _) => t.exp = st,
            (_, Some(j)) => t.planos[j] = st,
            _ => unreachable!(),
        }
    }
    if sv.len() as u64 != esperado {
        return falha("faltam streams na tabela");
    }
    if file.len() as u64 > base + off {
        return falha("dados depois do fim do corpo");
    }
    let (fonte, fonte_sha256) = valida_fonte(h.get("fonte").unwrap(), &ts)?;
    let gguf = match h.get("gguf") {
        Some(v) if versao >= 3 && !v.is_null() => Some(valida_gguf(v, &ts)?),
        _ => None,
    };
    if versao >= 3 && ts.iter().any(|t| t.extra) && gguf.is_none() {
        return falha("tensores extras sem a seção gguf");
    }
    Ok(Arquivo { versao, base, fim: base + off, m: m_max, integridade: integ, tensores: ts, fonte, fonte_sha256, niveis, gguf })
}
fn plano_j(p: &str) -> Option<u64> {
    let d = p.strip_prefix('m')?;
    if d.is_empty() || d.starts_with('0') || !d.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    // número gigante: não cabe em u64, logo é maior que qualquer mantissa
    Some(d.parse::<u64>().unwrap_or(u64::MAX))
}
fn valida_fonte(v: &Value, ts: &[Tensor]) -> R<(Option<Vec<u8>>, Option<String>)> {
    if v.is_null() {
        return Ok((None, None));
    }
    let f = obj(Some(v), "fonte")?;
    let com_sha = f.contains_key("sha256");
    so_chaves(f, if com_sha { &["tipo", "header_b64", "sha256"] } else { &["tipo", "header_b64"] }, "fonte")?;
    if texto(f.get("tipo")) != Some("safetensors") {
        return falha("fonte: tipo desconhecido");
    }
    let sha = if com_sha {
        match hex64(f.get("sha256").unwrap()) {
            Some(s) => Some(s.to_string()),
            None => return falha("fonte: sha256 inválido"),
        }
    } else {
        None
    };
    let b64 = texto(f.get("header_b64")).ok_or_else(|| Erro("fonte.header_b64: base64 inválido".into()))?;
    let eng = base64::engine::general_purpose::STANDARD;
    let hb = eng.decode(b64).or_else(|_| falha("fonte.header_b64: base64 inválido"))?;
    if eng.encode(&hb) != b64 {
        return falha("fonte.header_b64: base64 não canônico");
    }
    let (es, _) = valida_safetensors(&hb, None)?;
    if es.len() != ts.iter().filter(|t| !t.extra).count() {
        return falha("fonte: tensores não batem com o .strata");
    }
    let por_nome: HashMap<&str, &Tensor> = ts.iter().filter(|t| !t.extra).map(|t| (t.name.as_str(), t)).collect();
    for e in &es {
        match por_nome.get(e.name.as_str()) {
            Some(t) if t.dtype == e.dtype && t.shape == e.shape && e.end - e.begin == t.bytes => {}
            _ => return falha(format!("fonte: tensor {:?} não bate com o .strata", e.name)),
        }
    }
    Ok((Some(hb), sha))
}

/// Seção `gguf` (0.3): alinhamento, partes na ordem do arquivo e SHA-256 opcional do GGUF.
fn valida_gguf(v: &Value, ts: &[Tensor]) -> R<SaidaGguf> {
    let o = obj(Some(v), "gguf")?;
    let com_sha = o.contains_key("sha256");
    so_chaves(o, if com_sha { &["alinhamento", "partes", "sha256"] } else { &["alinhamento", "partes"] }, "gguf")?;
    let alinhamento = inteiro(o.get("alinhamento"), "gguf.alinhamento")?;
    if !alinhamento.is_power_of_two() || alinhamento > 1 << 16 {
        return falha("gguf: alinhamento inválido");
    }
    let sha256 = if com_sha { Some(hex64(o.get("sha256").unwrap()).ok_or_else(|| Erro("gguf: sha256 inválido".into()))?.to_string()) } else { None };
    let Some(Value::Array(pv)) = o.get("partes") else { return falha("gguf: partes inválido") };
    let mut partes = Vec::with_capacity(pv.len());
    let mut usados = HashSet::new();
    for (i, p) in pv.iter().enumerate() {
        let onde = format!("gguf.partes[{i}]");
        let po = obj(Some(p), &onde)?;
        so_chaves(po, &["t", "como"], &onde)?;
        let ti = inteiro(po.get("t"), &onde)? as usize;
        let Some(t) = ts.get(ti) else { return falha(format!("{onde}: tensor fora da tabela")) };
        let como = match texto(po.get("como")) {
            Some("cabecalho") if i == 0 && t.extra => ComoGguf::Cabecalho,
            Some("q") if i > 0 && t.res.is_some() => ComoGguf::Q,
            Some("f32") if i > 0 && !t.extra && t.lay.is_none() && t.dtype == "BF16" => ComoGguf::F32,
            Some("bruto") if i > 0 && t.extra => ComoGguf::Bruto,
            _ => return falha(format!("{onde}: parte inválida")),
        };
        if !usados.insert((ti, como == ComoGguf::F32)) {
            return falha(format!("{onde}: parte repetida"));
        }
        partes.push((ti, como));
    }
    if partes.first().map(|p| p.1) != Some(ComoGguf::Cabecalho) {
        return falha("gguf: falta o cabeçalho");
    }
    Ok(SaidaGguf { alinhamento, sha256, partes })
}

/* ---------------- decodificadores ---------------- */

/// Deflate cru (RFC 1951) com saída de tamanho exato e todos os bytes de entrada consumidos.
fn inflate(src: &[u8], n: u64) -> R<Vec<u8>> {
    use miniz_oxide::inflate::core::{decompress, inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF, DecompressorOxide};
    use miniz_oxide::inflate::TINFLStatus;
    let mut out = vec![0u8; n as usize];
    let mut d = Box::<DecompressorOxide>::default();
    let (st, lido, escrito) = decompress(&mut d, src, &mut out, 0, TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF);
    match st {
        TINFLStatus::Done if lido == src.len() && escrito as u64 == n => Ok(out),
        TINFLStatus::Done if lido != src.len() => falha("deflate: bytes depois do fim do stream"),
        TINFLStatus::Done => falha("deflate: saída menor que o tamanho declarado"),
        TINFLStatus::HasMoreOutput => falha("deflate: saída maior que o tamanho declarado"),
        TINFLStatus::NeedsMoreInput | TINFLStatus::FailedCannotMakeProgress => falha("deflate: stream truncado"),
        _ => falha("deflate: stream inválido"),
    }
}

const RANS_BITS: u32 = 15;
const RANS_M: u32 = 1 << RANS_BITS;
const RANS_L: u32 = 1 << 23;
/// rANS de ordem 0 (A.11): tabela em ordem crescente que soma 2^15, estado inicial em [2^23, 2^31),
/// e no fim o estado volta a 2^23 com todos os bytes gastos.
fn rans(src: &[u8], n: u64) -> R<Vec<u8>> {
    if src.len() < 2 {
        return falha("rANS: stream curto demais");
    }
    let k = u16::from_le_bytes([src[0], src[1]]) as usize;
    if k > 256 || (n > 0 && k == 0) || src.len() < 2 + 3 * k + 4 {
        return falha("rANS: tabela de frequências inválida");
    }
    let (mut freq, mut cum, mut slot) = ([0u32; 256], [0u32; 256], vec![0u8; RANS_M as usize]);
    let (mut o, mut tot, mut ant) = (2usize, 0u32, -1i32);
    for _ in 0..k {
        let s = src[o];
        let fs = u16::from_le_bytes([src[o + 1], src[o + 2]]) as u32;
        if (s as i32) <= ant || fs == 0 || tot + fs > RANS_M {
            return falha("rANS: tabela de frequências inválida");
        }
        ant = s as i32;
        freq[s as usize] = fs;
        cum[s as usize] = tot;
        slot[tot as usize..(tot + fs) as usize].fill(s);
        tot += fs;
        o += 3;
    }
    if k > 0 && tot != RANS_M {
        return falha("rANS: frequências não somam 2^15");
    }
    let mut x = u32::from_le_bytes(src[o..o + 4].try_into().unwrap());
    if !(RANS_L..0x8000_0000).contains(&x) {
        return falha("rANS: estado inicial inválido");
    }
    let mut p = o + 4;
    let mut out = vec![0u8; n as usize];
    for v in out.iter_mut() {
        let sl = x & (RANS_M - 1);
        let s = slot[sl as usize] as usize;
        *v = s as u8;
        x = freq[s] * (x >> RANS_BITS) + sl - cum[s];
        while x < RANS_L {
            let Some(b) = src.get(p) else { return falha("rANS: stream truncado") };
            x = (x << 8) | *b as u32;
            p += 1;
        }
    }
    if x != RANS_L || p != src.len() {
        return falha("rANS: stream corrompido");
    }
    Ok(out)
}

/// Um stream: dentro do prefixo, hash antes de decodificar, tamanho exato e forma canônica depois.
/// `file` começa no byte `desloc` do arquivo (0 = do início).
fn le_stream(file: &[u8], desloc: u64, a: &Arquivo, t: &Tensor, e: &Stream, eh_exp: bool) -> R<Vec<u8>> {
    let ini = a.base + e.o - desloc;
    if ini + e.l > file.len() as u64 {
        return falha("estrato fora do prefixo lido: peça profundidade menor");
    }
    let b = &file[ini as usize..(ini + e.l) as usize];
    if let Some(h) = &e.h {
        if &sha256_hex(b) != h {
            return falha(format!("hash não confere: {}", t.name));
        }
    }
    let out = match e.c {
        Cod::Raw | Cod::Fecho => b.to_vec(),
        Cod::Deflate => inflate(b, e.r)?,
        Cod::Rans => rans(b, e.r)?,
    };
    if let (Some(l), None) = (t.lay, &t.res) {
        if eh_exp {
            if l.e < 8 && out.iter().any(|&x| (x as u32) >= 1 << l.e) {
                return falha(format!("expoente fora do campo do dtype: {}", t.name));
            }
        } else if !t.n.is_multiple_of(8) && out.last().is_some_and(|&x| x & (0xff >> (t.n % 8)) != 0) {
            return falha(format!("bits de enchimento não nulos: {}", t.name));
        }
    }
    Ok(out)
}

/// Roda `f(0..n)` em até `nt` threads (fila comum) e devolve os resultados na ordem. Um erro faz
/// as threads pararem de pegar trabalho; vale sempre o erro do menor índice, então o resultado não
/// depende do número de threads.
fn em_paralelo<T: Send>(nt: usize, n: usize, f: impl Fn(usize) -> R<T> + Sync) -> R<Vec<T>> {
    em_paralelo_idx(nt, n, f).map_err(|(_, e)| e)
}
/// Como `em_paralelo`, mas o erro diz o índice que falhou.
fn em_paralelo_idx<T: Send>(nt: usize, n: usize, f: impl Fn(usize) -> R<T> + Sync) -> Result<Vec<T>, (usize, Erro)> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
    if nt <= 1 || n <= 1 {
        return (0..n).map(|i| f(i).map_err(|e| (i, e))).collect();
    }
    let (prox, parar) = (AtomicUsize::new(0), AtomicBool::new(false));
    let feitos: Vec<Vec<(usize, R<T>)>> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..nt.min(n))
            .map(|_| {
                s.spawn(|| {
                    let mut meus = Vec::new();
                    while !parar.load(Relaxed) {
                        let i = prox.fetch_add(1, Relaxed);
                        if i >= n {
                            break;
                        }
                        let r = f(i);
                        if r.is_err() {
                            parar.store(true, Relaxed);
                        }
                        meus.push((i, r));
                    }
                    meus
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p))).collect()
    });
    let mut res: Vec<Option<R<T>>> = (0..n).map(|_| None).collect();
    for (i, r) in feitos.into_iter().flatten() {
        res[i] = Some(r);
    }
    // o que ficou sem fazer (as threads pararam por um erro) é refeito em série, para que o erro
    // devolvido seja o do primeiro índice que falha
    res.into_iter().enumerate().map(|(i, r)| r.unwrap_or_else(|| f(i)).map_err(|e| (i, e))).collect()
}

/// Monta um tensor com `prof` planos de mantissa (ponto médio quando faltam planos, A.7), lendo os
/// streams e juntando os pesos em `nt` threads.
fn monta(file: &[u8], a: &Arquivo, t: &Tensor, prof: u32, nt: usize) -> R<Vec<u8>> {
    if let Some(rs) = &t.res {
        let d = prof.min(a.niveis + 1);
        let mut sts = vec![rs.q.as_ref().unwrap()];
        sts.extend(rs.niveis[..d.min(a.niveis) as usize].iter().map(|s| s.as_ref().unwrap()));
        if d > a.niveis {
            sts.push(rs.fecho.as_ref().unwrap());
        }
        let lidos = em_paralelo(nt, sts.len(), |k| le_stream(file, 0, a, t, sts[k], false))?;
        let planos: Vec<&[u8]> = lidos[1..1 + d.min(a.niveis) as usize].iter().map(Vec::as_slice).collect();
        let fecho = (d > a.niveis).then(|| lidos.last().unwrap().as_slice());
        return junta_res(t, rs, d, &lidos[0], &planos, fecho, nt);
    }
    let Some(l) = t.lay else {
        let b = le_stream(file, 0, a, t, t.raw.as_ref().unwrap(), false)?;
        confere_tensor(t, &b)?;
        return Ok(b);
    };
    let d = prof.min(l.m);
    let mut sts: Vec<(&Stream, bool)> = vec![(t.sign.as_ref().unwrap(), false), (t.exp.as_ref().unwrap(), true)];
    sts.extend(t.planos[..d as usize].iter().map(|p| (p.as_ref().unwrap(), false)));
    let lidos = em_paralelo(nt, sts.len(), |k| le_stream(file, 0, a, t, sts[k].0, sts[k].1))?;
    let planos: Vec<&[u8]> = lidos[2..].iter().map(Vec::as_slice).collect();
    junta_tensor(t, l, d, &lidos[0], &lidos[1], &planos, nt)
}
/// Junta os streams já lidos de um tensor de ponto flutuante nos pesos (em `nt` threads) e, com
/// todos os planos, confere o SHA-256 do tensor.
#[allow(clippy::too_many_arguments)]
fn junta_tensor(t: &Tensor, l: Layout, d: u32, sign: &[u8], exp: &[u8], planos: &[&[u8]], nt: usize) -> R<Vec<u8>> {
    let (emax, drop) = ((1u32 << l.e) - 1, l.m - d);
    let meio = if drop > 0 { 1u32 << (drop - 1) } else { 0 };
    let n = t.n as usize;
    let mut out = vec![0u8; n * l.bytes as usize];
    // fatias de grupos de 8 pesos, uma por thread
    let w = l.bytes as usize;
    let grupos = n.div_ceil(8);
    let por = grupos.div_ceil(nt.max(1)).max(1);
    std::thread::scope(|s| {
        for (k, fatia) in out.chunks_mut(por * 8 * w).enumerate() {
            let g0 = k * por;
            let mut job = move || match w {
                1 => junta::<1>(fatia, g0, n, l, d, meio, emax, sign, exp, planos),
                2 => junta::<2>(fatia, g0, n, l, d, meio, emax, sign, exp, planos),
                _ => junta::<4>(fatia, g0, n, l, d, meio, emax, sign, exp, planos),
            };
            if nt <= 1 {
                job()
            } else {
                s.spawn(job);
            }
        }
    });
    if d == l.m {
        confere_tensor(t, &out)?;
    }
    Ok(out)
}
/// Junta sinal, expoente e `d` planos em pesos de W bytes, na fatia `out` que começa no grupo `g0`.
/// 8 pesos por vez: os bytes de até 8 planos formam uma matriz 8×8 de bits; transposta, cada byte
/// vira os bits de mantissa de um peso (transpose8 do Hacker's Delight).
#[allow(clippy::too_many_arguments)]
fn junta<const W: usize>(out: &mut [u8], g0: usize, n: usize, l: Layout, d: u32, meio: u32, emax: u32, sign: &[u8], exp: &[u8], planos: &[&[u8]]) {
    let m = l.m;
    for (gl, bloco) in out.chunks_mut(8 * W).enumerate() {
        let g = g0 + gl;
        let mut mant = [0u32; 8];
        for j0 in (0..d as usize).step_by(8) {
            let mut x = 0u64;
            for (r, pl) in planos[j0..(j0 + 8).min(d as usize)].iter().enumerate() {
                x |= (pl[g] as u64) << (56 - 8 * r);
            }
            x = (x & 0xAA55_AA55_AA55_AA55) | ((x & 0x00AA_00AA_00AA_00AA) << 7) | ((x >> 7) & 0x00AA_00AA_00AA_00AA);
            x = (x & 0xCCCC_3333_CCCC_3333) | ((x & 0x0000_CCCC_0000_CCCC) << 14) | ((x >> 14) & 0x0000_CCCC_0000_CCCC);
            x = (x & 0xF0F0_F0F0_0F0F_0F0F) | ((x & 0x0000_0000_F0F0_F0F0) << 28) | ((x >> 28) & 0x0000_0000_F0F0_F0F0);
            // o plano j0 + r está no bit 7 − r; no peso ele vai para o bit m − 1 − j0 − r
            let sh = m as i32 - 8 - j0 as i32;
            for (c, v) in mant.iter_mut().enumerate() {
                let b = ((x >> (56 - 8 * c)) & 0xff) as u32;
                *v |= if sh >= 0 { b << sh } else { b >> -sh };
            }
        }
        let (sb, ex) = (sign[g], &exp[8 * g..(8 * g + 8).min(n)]);
        for ((c, peso), (&e, &mv)) in bloco.chunks_exact_mut(W).enumerate().zip(ex.iter().zip(mant.iter())) {
            let e = e as u32;
            let mut u = (((sb >> (7 - c)) & 1) as u32) << (l.bits - 1) | e << m | mv;
            if meio != 0 && e != 0 && e != emax {
                u |= meio;
            }
            peso.copy_from_slice(&u.to_le_bytes()[..W]);
        }
    }
}
/// Monta um tensor residual lido até o estrato `d`: com o fecho, os pesos exatos (e o hash
/// conferido); sem ele, o centro da sub-célula do nível d arredondado para BF16 (A.14).
fn junta_res(t: &Tensor, rs: &Res, d: u32, q: &[u8], planos: &[&[u8]], fecho: Option<&[u8]>, nt: usize) -> R<Vec<u8>> {
    let gr = residual::Grades { g: rs.g, base: q };
    let u = match fecho {
        Some(f) => residual::decodifica(&gr, planos, f, nt)?,
        None => residual::aproxima(&gr, planos, d, nt)?,
    };
    let u = match rs.permuta {
        Some(h) => residual::permuta(&u, t.shape[0] as usize, t.shape[1] as usize, h, false),
        None => u,
    };
    let out: Vec<u8> = u.iter().flat_map(|x| x.to_le_bytes()).collect();
    if fecho.is_some() {
        confere_tensor(t, &out)?;
    }
    Ok(out)
}
fn confere_tensor(t: &Tensor, b: &[u8]) -> R<()> {
    match &t.sha256 {
        Some(h) if &sha256_hex(b) != h => falha(format!("hash do tensor não confere: {}", t.name)),
        _ => Ok(()),
    }
}

impl Arquivo {
    /// Bytes do início do arquivo necessários para ler todos os tensores com `prof` planos de
    /// mantissa: o que baixar para usar o modelo nessa profundidade (u32::MAX = arquivo inteiro).
    pub fn bytes_ate(&self, prof: u32) -> u64 {
        let fim = |s: &Option<Stream>| s.as_ref().map_or(0, |s| s.o + s.l);
        let mut n = 0;
        for t in &self.tensores {
            if let Some(rs) = &t.res {
                n = n.max(fim(&rs.q));
                for p in rs.niveis.iter().take(prof.min(self.niveis) as usize) {
                    n = n.max(fim(p));
                }
                if prof > self.niveis {
                    n = n.max(fim(&rs.fecho));
                }
            }
            n = n.max(fim(&t.raw)).max(fim(&t.sign)).max(fim(&t.exp));
            for p in t.planos.iter().take(prof.min(t.planos.len() as u32) as usize) {
                n = n.max(fim(p));
            }
        }
        self.base + n
    }
}

/// Decodifica o tensor `i` de um arquivo já aberto por `abre`, com `prof` planos de mantissa, em
/// `threads` threads (0 = todos os núcleos). `file` pode ser só um prefixo: basta que contenha os
/// streams até o estrato `prof`.
pub fn decode_tensor(file: &[u8], a: &Arquivo, i: usize, prof: u32, threads: usize) -> R<Vec<u8>> {
    match a.tensores.get(i) {
        Some(t) => monta(file, a, t, prof, nucleos(threads)),
        None => falha(format!("tensor {i} não existe")),
    }
}
fn nucleos(threads: usize) -> usize {
    match threads {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    }
}

/// Monta todos os tensores com `f(i, threads)`, na ordem do cabeçalho: cada sequência de pequenos
/// vai em paralelo (um tensor por thread), cada grande sozinho dividido entre as threads. Assim o
/// primeiro erro é sempre o mesmo, com qualquer número de threads.
fn por_tensor(a: &Arquivo, nt: usize, f: impl Fn(usize, usize) -> R<Vec<u8>> + Sync) -> R<Vec<Vec<u8>>> {
    let todos: Vec<usize> = (0..a.tensores.len()).collect();
    por_indices(a, &todos, nt, f)
}
/// Como `por_tensor`, só para os tensores `idx` (nessa ordem).
fn por_indices(a: &Arquivo, idx: &[usize], nt: usize, f: impl Fn(usize, usize) -> R<Vec<u8>> + Sync) -> R<Vec<Vec<u8>>> {
    let total: u64 = a.tensores.iter().map(|t| t.bytes).sum();
    // um tensor residual se divide sozinho em blocos de 64 K pesos: vai inteiro com todas as threads
    let grande = |i: usize| nt > 1 && (a.tensores[i].res.is_some() || a.tensores[i].bytes * 4 * nt as u64 > total);
    let mut ts = Vec::with_capacity(idx.len());
    let mut k = 0;
    while k < idx.len() {
        if grande(idx[k]) {
            ts.push(f(idx[k], nt)?);
            k += 1;
            continue;
        }
        let j = (k..idx.len()).find(|&j| grande(idx[j])).unwrap_or(idx.len());
        ts.extend(em_paralelo(nt, j - k, |q| f(idx[k + q], 1))?);
        k = j;
    }
    Ok(ts)
}

/// Decodifica todos os tensores com `prof` planos de mantissa (u32::MAX = todos), em
/// `op.threads` threads (0 = todos os núcleos). Os tensores grandes (um embedding pode ser um
/// quinto do modelo) são divididos entre as threads; os pequenos vão em paralelo, um por thread.
/// O erro devolvido é o mesmo com qualquer número de threads.
pub fn decode(file: &[u8], prof: u32, op: &Opcoes) -> R<(Arquivo, Vec<Vec<u8>>)> {
    let a = abre(file, op)?;
    let nt = nucleos(op.threads);
    let ts = por_tensor(&a, nt, |i, ntt| monta(file, &a, &a.tensores[i], prof, ntt))?;
    Ok((a, ts))
}

/// .strata → .safetensors byte a byte, conferindo o SHA-256 do original quando o arquivo o guarda.
pub fn to_safetensors(file: &[u8], op: &Opcoes) -> R<Vec<u8>> {
    let (a, ts) = decode(file, u32::MAX, op)?;
    let Some(hb) = &a.fonte else { return falha("o arquivo não guarda o cabeçalho do safetensors original") };
    let (es, total) = valida_safetensors(hb, None)?;
    let por_nome: HashMap<&str, usize> = a.tensores.iter().enumerate().map(|(i, t)| (t.name.as_str(), i)).collect();
    let mut out = Vec::with_capacity(8 + hb.len() + total as usize);
    out.extend_from_slice(&(hb.len() as u64).to_le_bytes());
    out.extend_from_slice(hb);
    for e in &es {
        out.extend_from_slice(&ts[por_nome[e.name.as_str()]]);
    }
    if a.fonte_sha256.as_ref().is_some_and(|s| &sha256_hex(&out) != s) {
        return falha("o .safetensors reconstruído não confere com o SHA-256 do original");
    }
    Ok(out)
}

/// Estrato 0 de um .strata 0.3 → o GGUF de origem byte a byte (A.14), conferindo o SHA-256 dele
/// quando o arquivo o guarda. `file` pode ser só o prefixo `bytes_ate(0)`.
pub fn para_gguf(file: &[u8], op: &Opcoes) -> R<Vec<u8>> {
    let a = abre(file, op)?;
    let Some(g) = &a.gguf else { return falha("o arquivo não guarda um GGUF") };
    let nt = nucleos(op.threads);
    let pedacos = em_paralelo(nt, g.partes.len(), |k| {
        let (ti, como) = g.partes[k];
        let t = &a.tensores[ti];
        match como {
            ComoGguf::Cabecalho | ComoGguf::Bruto => monta(file, &a, t, 0, 1),
            ComoGguf::Q => le_stream(file, 0, &a, t, t.res.as_ref().unwrap().q.as_ref().unwrap(), false),
            ComoGguf::F32 => Ok(monta(file, &a, t, 0, 1)?.chunks_exact(2).flat_map(|b| [0, 0, b[0], b[1]]).collect()),
        }
    })?;
    let cab = pedacos[0].len();
    let mut out = Vec::with_capacity(pedacos.iter().map(|p| p.len() + g.alinhamento as usize).sum());
    for (k, p) in pedacos.iter().enumerate() {
        if k > 0 {
            out.resize(cab + (out.len() - cab).div_ceil(g.alinhamento as usize) * g.alinhamento as usize, 0);
        }
        out.extend_from_slice(p);
    }
    if g.sha256.as_ref().is_some_and(|s| &sha256_hex(&out) != s) {
        return falha("o GGUF remontado não confere com o SHA-256 do original");
    }
    Ok(out)
}

/// Blocos `Q4_K_R<k>` do tensor residual `i` (B.8, kernel/): por super-bloco de 256 pesos, os 144
/// bytes do Q4_K seguidos de k planos de 32 bytes, na ordem do GGUF. `file` pode ser o prefixo
/// `bytes_ate(k)`.
pub fn blocos_q4kr(file: &[u8], a: &Arquivo, i: usize, k: u32) -> R<Vec<u8>> {
    blocos_r(file, a, i, k, residual::Grade::Q4K)
}

/// Blocos `Q6_K_R<k>` (B.8): os 210 bytes do bloco Q6_K seguidos de k planos de 32 bytes.
pub fn blocos_q6kr(file: &[u8], a: &Arquivo, i: usize, k: u32) -> R<Vec<u8>> {
    blocos_r(file, a, i, k, residual::Grade::Q6K)
}

fn blocos_r(file: &[u8], a: &Arquivo, i: usize, k: u32, g: residual::Grade) -> R<Vec<u8>> {
    let Some(t) = a.tensores.get(i) else { return falha(format!("tensor {i} não existe")) };
    let Some(rs) = t.res.as_ref().filter(|r| r.g == g) else { return falha(format!("{}: não é um tensor residual {}", t.name, g.nome())) };
    if k > a.niveis {
        return falha("o arquivo não tem tantos níveis");
    }
    let q = le_stream(file, 0, a, t, rs.q.as_ref().unwrap(), false)?;
    let planos = rs.niveis[..k as usize].iter().map(|s| le_stream(file, 0, a, t, s.as_ref().unwrap(), false)).collect::<R<Vec<_>>>()?;
    let nb = g.bytes_bloco();
    let mut out = Vec::with_capacity(q.len() / nb * (nb + 32 * k as usize));
    for (sb, bq) in q.chunks_exact(nb).enumerate() {
        out.extend_from_slice(bq);
        for p in &planos {
            out.extend_from_slice(&p[32 * sb..32 * sb + 32]);
        }
    }
    Ok(out)
}

/// Id do arquivo: SHA-256 de MAGIC + HL + cabeçalho (autentica o arquivo inteiro, A.13).
pub fn cabecalho_id(file: &[u8]) -> R<String> {
    if file.len() < 12 {
        return falha("arquivo curto demais");
    }
    let hl = u32::from_le_bytes(file[8..12].try_into().unwrap()) as usize;
    match file.get(..12 + hl) {
        Some(h) => Ok(sha256_hex(h)),
        None => falha("cabeçalho incompleto"),
    }
}

/* ---------------- leitura em fluxo ---------------- */

#[derive(Clone, Copy)]
enum Parte {
    Raw,
    Sign,
    Exp,
    Plano(usize),
    Q,
    Nivel(usize),
    Fecho,
}
#[derive(Default)]
struct Lidos {
    raw: Option<Vec<u8>>,
    sign: Option<Vec<u8>>,
    exp: Option<Vec<u8>>,
    planos: Vec<Option<Vec<u8>>>,
    q: Option<Vec<u8>>,
    niveis: Vec<Option<Vec<u8>>>,
    fecho: Option<Vec<u8>>,
    /// streams deste tensor que ainda não chegaram
    faltam: usize,
}

/// Decodifica um .strata enquanto ele chega (download, pipe). Cada stream é conferido e
/// descomprimido assim que seus bytes terminam de chegar, e cada tensor é montado (e seu hash
/// conferido) assim que chega o último stream dele; os bytes já usados saem da memória. Ao fim do
/// download sobra pouco a fazer. Aceita e recusa o mesmo que `decode` com os mesmos bytes e a
/// mesma profundidade; o limite de saída é aplicado sobre os bytes já recebidos.
pub struct Fluxo {
    a: Arquivo,
    op: Opcoes,
    prof: u32,
    /// bytes recebidos e ainda não decodificados; `buf[0]` é o byte `desloc` do arquivo
    buf: Vec<u8>,
    desloc: u64,
    /// streams pedidos, na ordem do arquivo (offsets crescentes)
    fila: Vec<(usize, Parte)>,
    prox: usize,
    lidos: Vec<Lidos>,
    saidas: Vec<Option<Vec<u8>>>,
    saida: u64,
}

impl Fluxo {
    /// Começa pelos primeiros bytes do arquivo; tem que conter pelo menos o cabeçalho inteiro
    /// (12 + HL bytes, HL = uint32 nos bytes 8..12). O resto pode vir junto. Só os streams até
    /// `prof` planos de mantissa são lidos (u32::MAX = todos); os seguintes, se chegarem, são
    /// ignorados, como `decode(arquivo, prof)` os ignora.
    pub fn novo(inicio: &[u8], prof: u32, op: &Opcoes) -> R<Fluxo> {
        if inicio.len() < 12 {
            return falha("arquivo curto demais");
        }
        let hl = u32::from_le_bytes(inicio[8..12].try_into().unwrap()) as usize;
        let cab = &inicio[..(12 + hl).min(inicio.len())];
        // o limite de saída depende dos bytes recebidos: é conferido em alimenta e termina
        let a = abre(cab, &Opcoes { limite_saida: Some(u64::MAX), ..op.clone() })?;
        let mut fila = Vec::new();
        let mut lidos: Vec<Lidos> = Vec::with_capacity(a.tensores.len());
        for (i, t) in a.tensores.iter().enumerate() {
            let mut add = |s: &Option<Stream>, p| {
                if let Some(s) = s {
                    fila.push((s.o, i, p));
                }
            };
            if let Some(rs) = &t.res {
                let d = prof.min(a.niveis + 1);
                let k = d.min(a.niveis) as usize;
                add(&rs.q, Parte::Q);
                for (j, s) in rs.niveis.iter().enumerate().take(k) {
                    add(s, Parte::Nivel(j));
                }
                if d > a.niveis {
                    add(&rs.fecho, Parte::Fecho);
                }
                lidos.push(Lidos { niveis: vec![None; k], faltam: 1 + k + (d > a.niveis) as usize, ..Default::default() });
                continue;
            }
            add(&t.raw, Parte::Raw);
            add(&t.sign, Parte::Sign);
            add(&t.exp, Parte::Exp);
            let d = prof.min(t.planos.len() as u32) as usize;
            for (j, pl) in t.planos.iter().enumerate().take(d) {
                add(pl, Parte::Plano(j));
            }
            let faltam = if t.lay.is_some() { 2 + d } else { 1 };
            lidos.push(Lidos { planos: vec![None; d], faltam, ..Default::default() });
        }
        fila.sort_by_key(|x| x.0);
        let nt = a.tensores.len();
        let mut f = Fluxo {
            a,
            op: op.clone(),
            prof,
            buf: Vec::new(),
            desloc: 0,
            fila: fila.into_iter().map(|(_, i, p)| (i, p)).collect(),
            prox: 0,
            lidos,
            saidas: (0..nt).map(|_| None).collect(),
            saida: 0,
        };
        f.alimenta(inicio)?;
        Ok(f)
    }

    fn stream(&self, i: usize, p: Parte) -> &Stream {
        let t = &self.a.tensores[i];
        match p {
            Parte::Raw => t.raw.as_ref(),
            Parte::Sign => t.sign.as_ref(),
            Parte::Exp => t.exp.as_ref(),
            Parte::Plano(j) => t.planos[j].as_ref(),
            Parte::Q => t.res.as_ref().unwrap().q.as_ref(),
            Parte::Nivel(j) => t.res.as_ref().unwrap().niveis[j].as_ref(),
            Parte::Fecho => t.res.as_ref().unwrap().fecho.as_ref(),
        }
        .unwrap()
    }
    fn limite(&self) -> u64 {
        self.op.limite_saida.unwrap_or_else(|| (1u64 << 30).max(64 * self.recebidos()))
    }
    /// Monta o tensor `i` com os streams guardados (todos já chegaram).
    fn monta_lidos(&self, i: usize, nt: usize) -> R<Vec<u8>> {
        let (t, l) = (&self.a.tensores[i], &self.lidos[i]);
        let falta = || Erro(format!("estrato fora do prefixo lido: {}", t.name));
        if let Some(rs) = &t.res {
            let q = l.q.as_deref().ok_or_else(falta)?;
            let planos = l.niveis.iter().map(|p| p.as_deref().ok_or_else(falta)).collect::<R<Vec<_>>>()?;
            let d = self.prof.min(self.a.niveis + 1);
            if d > self.a.niveis && l.fecho.is_none() {
                return Err(falta());
            }
            return junta_res(t, rs, d, q, &planos, l.fecho.as_deref(), nt);
        }
        let Some(lay) = t.lay else {
            let b = l.raw.as_ref().ok_or_else(falta)?;
            confere_tensor(t, b)?;
            return Ok(b.clone());
        };
        let d = self.prof.min(lay.m);
        let planos = l.planos.iter().map(|p| p.as_deref().ok_or_else(falta)).collect::<R<Vec<_>>>()?;
        let (sign, exp) = (l.sign.as_deref().ok_or_else(falta)?, l.exp.as_deref().ok_or_else(falta)?);
        junta_tensor(t, lay, d, sign, exp, &planos, nt)
    }

    /// Recebe os próximos bytes do arquivo. Os streams que ficaram completos são lidos (hash,
    /// descompressão, forma) em paralelo, e os tensores que ficaram completos são montados.
    pub fn alimenta(&mut self, dados: &[u8]) -> R<()> {
        if self.recebidos() + dados.len() as u64 > self.a.fim {
            return falha("dados depois do fim do corpo");
        }
        self.buf.extend_from_slice(dados);
        let n = self.recebidos();
        let mut fim = self.prox;
        while fim < self.fila.len() {
            let (i, p) = self.fila[fim];
            let s = self.stream(i, p);
            if self.a.base + s.o + s.l > n {
                break;
            }
            fim += 1;
        }
        if fim == self.prox {
            return Ok(());
        }
        let r: u64 = self.fila[self.prox..fim].iter().map(|&(i, p)| self.stream(i, p).r).sum();
        if self.saida + r > self.limite() {
            return falha(format!("os streams declaram mais de {} bytes de saída (limite_saida)", self.limite()));
        }
        let nt = nucleos(self.op.threads);
        let novos = &self.fila[self.prox..fim];
        let out = em_paralelo(nt, novos.len(), |k| {
            let (i, p) = novos[k];
            le_stream(&self.buf, self.desloc, &self.a, &self.a.tensores[i], self.stream(i, p), matches!(p, Parte::Exp))
        })?;
        let mut prontos = Vec::new();
        for (&(i, p), v) in self.fila[self.prox..fim].iter().zip(out) {
            let l = &mut self.lidos[i];
            match p {
                Parte::Raw => l.raw = Some(v),
                Parte::Sign => l.sign = Some(v),
                Parte::Exp => l.exp = Some(v),
                Parte::Plano(j) => l.planos[j] = Some(v),
                Parte::Q => l.q = Some(v),
                Parte::Nivel(j) => l.niveis[j] = Some(v),
                Parte::Fecho => l.fecho = Some(v),
            }
            l.faltam -= 1;
            if l.faltam == 0 {
                prontos.push(i);
            }
        }
        self.saida += r;
        self.prox = fim;
        // o que já foi decodificado não é mais preciso: fica só o começo do próximo stream
        let corte = match self.fila.get(fim) {
            Some(&(i, p)) => self.a.base + self.stream(i, p).o,
            None => n,
        };
        self.buf.drain(..(corte - self.desloc) as usize);
        self.desloc = corte;
        // tensores completos: monta já (e confere o hash), e solta os streams deles
        prontos.sort_unstable();
        let ts = por_indices(&self.a, &prontos, nt, |i, ntt| self.monta_lidos(i, ntt))?;
        for (i, v) in prontos.into_iter().zip(ts) {
            self.saidas[i] = Some(v);
            self.lidos[i] = Lidos::default();
        }
        Ok(())
    }

    /// Bytes recebidos até agora (contando o cabeçalho).
    pub fn recebidos(&self) -> u64 {
        self.desloc + self.buf.len() as u64
    }
    /// O cabeçalho validado: nomes, tipos, `bytes_ate(prof)`, `fim`.
    pub fn arquivo(&self) -> &Arquivo {
        &self.a
    }

    /// Entrega os tensores com os planos pedidos em `novo`. Precisa de pelo menos
    /// `arquivo().bytes_ate(prof)` bytes recebidos; com todos os planos, do arquivo inteiro.
    pub fn termina(self) -> R<(Arquivo, Vec<Vec<u8>>)> {
        if self.recebidos() < self.a.bytes_ate(self.prof) {
            return falha("estrato fora do prefixo lido: peça profundidade menor");
        }
        let total: u64 = self.a.tensores.iter().map(|t| t.bytes).sum();
        if total > self.limite() {
            return falha(format!("o cabeçalho declara mais de {} bytes de saída (limite_saida)", self.limite()));
        }
        let Fluxo { a, saidas, .. } = self;
        let ts = saidas.into_iter().zip(&a.tensores).map(|(s, t)| s.ok_or_else(|| Erro(format!("estrato fora do prefixo lido: {}", t.name)))).collect::<R<Vec<_>>>()?;
        Ok((a, ts))
    }
}
