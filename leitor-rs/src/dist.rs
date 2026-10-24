//! Distribuição sobre um GGUF já publicado (docs/02 B.10): o estrato 0 é o próprio arquivo
//! Q4_K_M do Hub, sem mudar um byte; os níveis 1..K e o fecho do modo residual (A.14) vão em
//! arquivos por estrato, como o desenho 2 da B.9; um índice JSON amarra tudo por SHA-256.
//!
//! Quem já tem o GGUF baixa só os arquivos de estrato para subir de nível ou voltar aos
//! `.safetensors` originais byte a byte. O Hub guarda o GGUF uma vez só.
//!
//! Tudo aqui lê os arquivos por faixas (`read_exact_at`), um tensor por vez, para caber em
//! modelos maiores que a RAM.

use crate::residual::{self, codifica, decodifica, le_gguf, nome_hf, permuta, Grade, Grades, Soma, GGML_Q4_K, GGML_Q6_K};
use crate::{falha, sha256_hex, valida_safetensors, Erro, R};
use base64::Engine;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub const FORMATO: &str = "strata-shards/0.1";
pub const DESENHO: &str = "por-estrato-gguf";
/// Preâmbulo de cada arquivo de estrato (o mesmo da B.9): "STRATAE", versão 1, estrato, parte.
const PREAMBULO: &[u8; 8] = b"STRATAE\x01";
/// Teto de um stream cru (o maior tensor de um modelo de 400 B pesos cabe com folga).
const TETO_STREAM: u64 = 16 << 30;

fn io<T>(r: std::io::Result<T>, onde: &str) -> R<T> {
    r.map_err(|e| Erro(format!("{onde}: {e}")))
}
fn le_faixa(f: &File, o: u64, n: u64, onde: &str) -> R<Vec<u8>> {
    if n > TETO_STREAM {
        return falha(format!("{onde}: faixa grande demais"));
    }
    let mut b = vec![0u8; n as usize];
    io(f.read_exact_at(&mut b, o), onde)?;
    Ok(b)
}
/// Como `le_faixa`, mas com `espera`, aguarda o arquivo crescer até cobrir a faixa (quem baixa
/// grava cada arquivo de estrato em ordem); desiste se ele ficar `espera` sem crescer.
fn le_faixa_espera(f: &File, o: u64, n: u64, onde: &str, espera: Option<std::time::Duration>) -> R<Vec<u8>> {
    if let Some(t) = espera {
        let fim = o.checked_add(n).ok_or_else(|| Erro(format!("{onde}: faixa inválida")))?;
        let (mut visto, mut desde) = (0, std::time::Instant::now());
        loop {
            let tam = io(f.metadata(), onde)?.len();
            if tam >= fim {
                break;
            }
            if tam != visto {
                (visto, desde) = (tam, std::time::Instant::now());
            } else if desde.elapsed() > t {
                return falha(format!("{onde}: o arquivo parou de crescer em {tam} bytes"));
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
    le_faixa(f, o, n, onde)
}
fn sha256_arquivo(f: &File) -> R<(String, u64)> {
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 64 << 20];
    let (mut o, mut r) = (0u64, f);
    loop {
        let n = io(r.read(&mut buf), "sha256")?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        o += n as u64;
    }
    Ok((hex(&h.finalize()), o))
}
fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Cabeçalho de um GGUF lido do começo do arquivo (cresce até caber).
fn cabecalho_gguf(f: &File) -> R<residual::Gguf> {
    let tam = io(f.metadata(), "gguf")?.len();
    let mut n = 16u64 << 20;
    loop {
        let b = le_faixa(f, 0, n.min(tam), "gguf")?;
        match le_gguf(&b) {
            Ok(g) => return Ok(g),
            Err(e) if e.0.contains("truncado") && n < tam && n < (1 << 30) => n *= 4,
            Err(e) => return Err(e),
        }
    }
}

/// Um shard de origem já aberto: cabeçalho e entradas em ordem de offset.
struct Shard {
    nome: String,
    f: File,
    hb: Vec<u8>,
    es: Vec<crate::EntradaSt>,
    dados: u64,
}
fn abre_shard(p: &Path) -> R<Shard> {
    let nome = p.file_name().and_then(|s| s.to_str()).ok_or_else(|| Erro("nome de shard inválido".into()))?.to_string();
    let f = io(File::open(p), &nome)?;
    let tam = io(f.metadata(), &nome)?.len();
    let hl = u64::from_le_bytes(le_faixa(&f, 0, 8, &nome)?.try_into().unwrap());
    if hl > crate::LIMITE_CABECALHO || 8 + hl > tam {
        return falha(format!("{nome}: cabeçalho inválido"));
    }
    let hb = le_faixa(&f, 8, hl, &nome)?;
    let (es, _) = valida_safetensors(&hb, Some(tam - 8 - hl))?;
    Ok(Shard { nome, f, hb, es, dados: 8 + hl })
}

/// Um arquivo de estrato sendo gravado.
struct Saida {
    nome: String,
    w: std::io::BufWriter<File>,
    pos: u64,
    h: Sha256,
}
impl Saida {
    fn nova(dir: &Path, s: u32) -> R<Saida> {
        let nome = format!("model-e{s:02}-00000-of-00001.strata");
        let mut w = std::io::BufWriter::with_capacity(8 << 20, io(File::create(dir.join(&nome)), &nome)?);
        let mut pre = PREAMBULO.to_vec();
        pre.extend_from_slice(&s.to_le_bytes());
        pre.extend_from_slice(&0u32.to_le_bytes());
        io(w.write_all(&pre), &nome)?;
        let mut h = Sha256::new();
        h.update(&pre);
        Ok(Saida { nome, w, pos: pre.len() as u64, h })
    }
    fn poe(&mut self, b: &[u8]) -> R<u64> {
        let o = self.pos;
        io(self.w.write_all(b), &self.nome)?;
        self.h.update(b);
        self.pos += b.len() as u64;
        Ok(o)
    }
}

pub struct Relatorio {
    /// por grade, "camadas" e "total", como o `residual::escreve`
    pub somas: BTreeMap<String, Soma>,
    /// (nome, estrato, bytes) de cada arquivo, o GGUF primeiro
    pub arquivos: Vec<(String, i64, u64)>,
    pub indice: u64,
}

/// Grava o índice e os arquivos de estrato em `dir`, a partir dos shards BF16 originais (na ordem
/// do `model.safetensors.index.json`, se houver) e do GGUF publicado (Q4_K/Q6_K) do mesmo modelo.
pub fn escreve(shards: &[PathBuf], indice_hf: Option<&Path>, gguf: &Path, dir: &Path, k: u32, nt: usize) -> R<Relatorio> {
    escreve_com(shards, indice_hf, gguf, dir, k, nt, false, false)
}

/// Como `escreve`, para modelos maiores que o disco: com `espera`, cada shard é esperado até
/// aparecer (quem baixa o grava com outro nome e renomeia no fim); com `apaga`, cada shard é
/// apagado depois de lido. O SHA-256 de cada shard fica no índice antes de apagar.
#[allow(clippy::too_many_arguments)]
pub fn escreve_com(shards: &[PathBuf], indice_hf: Option<&Path>, gguf: &Path, dir: &Path, k: u32, nt: usize, espera: bool, apaga: bool) -> R<Relatorio> {
    if !(1..=8).contains(&k) {
        return falha("níveis: de 1 a 8");
    }
    let gf = io(File::open(gguf), "gguf")?;
    let g = cabecalho_gguf(&gf)?;
    if g.kv.get("general.architecture").map(String::as_str) != Some("llama") {
        return falha("o escritor só conhece a arquitetura llama");
    }
    let cab = |c: &str| -> R<usize> { g.kv.get(c).and_then(|v| v.parse().ok()).ok_or_else(|| Erro(format!("GGUF sem {c}"))) };
    let (nh, nkv) = (cab("llama.attention.head_count")?, cab("llama.attention.head_count_kv")?);
    let (gsha, gtam) = sha256_arquivo(&gf)?;
    let gnome = gguf.file_name().and_then(|s| s.to_str()).ok_or_else(|| Erro("nome do GGUF inválido".into()))?.to_string();
    // tensor do safetensors -> tensor Q4_K/Q6_K do GGUF
    let mut q_de: HashMap<String, (usize, bool)> = HashMap::new();
    for (i, t) in g.tensores.iter().enumerate() {
        if t.tipo == GGML_Q4_K || t.tipo == GGML_Q6_K {
            let Some(h) = nome_hf(&t.nome) else { return falha(format!("{}: tensor quantizado sem nome no safetensors", t.nome)) };
            q_de.insert(h, (i, false));
        }
    }

    let mut sai: Vec<Saida> = (0..=k + 1).map(|s| Saida::nova(dir, s)).collect::<R<_>>()?;
    let mut tensores = Vec::new();
    let mut streams = Vec::new();
    let mut fontes = Vec::new();
    let mut somas: BTreeMap<String, Soma> = BTreeMap::new();
    let eng = base64::engine::general_purpose::STANDARD;
    for (si, p) in shards.iter().enumerate() {
        if espera {
            let mut avisou = false;
            while !p.exists() {
                if !avisou {
                    eprintln!("esperando {}", p.display());
                    avisou = true;
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }
        let sh = abre_shard(p)?;
        let mut hs = Sha256::new();
        hs.update((sh.hb.len() as u64).to_le_bytes());
        hs.update(&sh.hb);
        for e in &sh.es {
            let bytes = le_faixa(&sh.f, sh.dados + e.begin, e.end - e.begin, &sh.nome)?;
            hs.update(&bytes);
            let ti = tensores.len();
            let mut tj = json!({"name": e.name, "dtype": e.dtype, "shape": e.shape, "bytes": bytes.len(), "sha256": sha256_hex(&bytes), "shard": si});
            match q_de.get_mut(&e.name) {
                Some((gi, usado)) => {
                    if std::mem::replace(usado, true) {
                        return falha(format!("{} aparece em dois shards", e.name));
                    }
                    let t = &g.tensores[*gi];
                    let gr = if t.tipo == GGML_Q4_K { Grade::Q4K } else { Grade::Q6K };
                    if e.dtype != "BF16" || e.shape.len() != 2 || t.ne.len() != 2 || e.shape[0] != t.ne[1] || e.shape[1] != t.ne[0] || e.shape[1] % 256 != 0 {
                        return falha(format!("{}: forma não bate com {}", t.nome, e.name));
                    }
                    let h = if t.nome.ends_with("attn_q.weight") { Some(nh) } else if t.nome.ends_with("attn_k.weight") { Some(nkv) } else { None };
                    let (linhas, cols) = (e.shape[0] as usize, e.shape[1] as usize);
                    let n = linhas * cols;
                    let w: Vec<u16> = bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                    drop(bytes);
                    let wg = match h {
                        Some(h) => permuta(&w, linhas, cols, h, true),
                        None => w,
                    };
                    let qo = g.dados + t.off;
                    let ql = (n / 256 * gr.bytes_bloco()) as u64;
                    let base = le_faixa(&gf, qo, ql, "gguf")?;
                    let c = codifica(&Grades { g: gr, base: &base }, &wg, k, nt)?;
                    for chave in [gr.nome(), "total"].into_iter().chain(e.name.starts_with("model.layers.").then_some("camadas")) {
                        let s = somas.entry(chave.to_string()).or_default();
                        s.pesos += n as u64;
                        s.base += 8 * ql;
                        s.niveis += k as u64 * n as u64;
                        s.fecho += 8 * c.fecho.len() as u64;
                        s.escapes += c.escapes;
                    }
                    tj["n"] = json!(n);
                    tj["residual"] = json!({"grade": gr.nome(), "permuta": h});
                    streams.push(json!({"t": ti, "s": 0, "p": "q", "f": 0, "o": qo, "l": ql, "r": ql, "c": "raw", "h": sha256_hex(&base)}));
                    for (j, pl) in c.planos.iter().enumerate() {
                        let o = sai[j + 1].poe(pl)?;
                        streams.push(json!({"t": ti, "s": j + 1, "p": format!("n{}", j + 1), "f": j + 2, "o": o, "l": pl.len(), "r": n / 8, "c": "raw", "h": sha256_hex(pl)}));
                    }
                    let o = sai[k as usize + 1].poe(&c.fecho)?;
                    streams.push(json!({"t": ti, "s": k + 1, "p": "f", "f": k + 2, "o": o, "l": c.fecho.len(), "r": 2 * n, "c": "fecho", "h": sha256_hex(&c.fecho)}));
                }
                None => {
                    tj["n"] = Value::Null;
                    let z = miniz_oxide::deflate::compress_to_vec(&bytes, 9);
                    let (b, cod) = if z.len() < bytes.len() { (&z[..], "deflate") } else { (&bytes[..], "raw") };
                    let o = sai[0].poe(b)?;
                    streams.push(json!({"t": ti, "s": 0, "p": "raw", "f": 1, "o": o, "l": b.len(), "r": bytes.len(), "c": cod, "h": sha256_hex(b)}));
                }
            }
            tensores.push(tj);
        }
        let tam = io(sh.f.metadata(), &sh.nome)?.len();
        fontes.push(json!({"nome": sh.nome, "bytes": tam, "sha256": hex(&hs.finalize()), "header_b64": eng.encode(&sh.hb)}));
        if espera || apaga {
            eprintln!("shard {}/{} lido: {}", si + 1, shards.len(), sh.nome);
        }
        if apaga {
            drop(sh);
            io(std::fs::remove_file(p), "apagar shard")?;
        }
    }
    if let Some((h, _)) = q_de.iter().find(|(_, (_, u))| !u) {
        return falha(format!("{h}: está no GGUF mas em nenhum shard"));
    }
    let mut arquivos = vec![json!({"nome": gnome, "s": 0, "externo": "gguf", "bytes": gtam, "sha256": gsha})];
    let mut rel = vec![(gnome.clone(), -1i64, gtam)];
    for (s, mut o) in sai.into_iter().enumerate() {
        io(o.w.flush(), &o.nome)?;
        arquivos.push(json!({"nome": o.nome, "s": s, "parte": 0, "bytes": o.pos, "sha256": hex(&o.h.finalize())}));
        rel.push((o.nome, s as i64, o.pos));
    }
    let mut fonte = json!({"shards": fontes});
    if let Some(p) = indice_hf {
        fonte["index_b64"] = json!(eng.encode(io(std::fs::read(p), "index.json")?));
    }
    let idx = json!({
        "format": FORMATO,
        "desenho": DESENHO,
        "strata": "0.3",
        "niveis": k,
        "estratos": k + 2,
        "base": {"arquivo": 0, "tipo": "gguf", "alinhamento": g.alinhamento},
        "arquivos": arquivos,
        "tensors": tensores,
        "streams": streams,
        "fonte": fonte,
        "integridade": {"alg": "sha256"},
    });
    let ib = serde_json::to_vec(&idx).map_err(|e| Erro(e.to_string()))?;
    io(std::fs::write(dir.join("model.strata.index.json"), &ib), "índice")?;
    Ok(Relatorio { somas, arquivos: rel, indice: ib.len() as u64 })
}

/* ---------------- leitor ---------------- */

struct Arq {
    f: File,
    bytes: u64,
    s: i64,
}
struct St {
    t: usize,
    s: u32,
    p: String,
    f: usize,
    o: u64,
    l: u64,
    r: u64,
    c: String,
    h: String,
}

pub struct Volta {
    /// (shard, SHA-256 conferido, bytes)
    pub shards: Vec<(String, String, u64)>,
    pub tensores: usize,
    /// segundos por fase (somados entre tensores; as fases em paralelo contam o tempo de parede)
    pub tempos: Vec<(&'static str, f64)>,
}

const FASES: [&str; 7] = ["hash dos arquivos", "leitura", "hash dos streams", "decodificação", "bytes", "hash da saída", "escrita"];
type Tempos = [AtomicU64; 7];
fn mede<T>(c: &AtomicU64, f: impl FnOnce() -> T) -> T {
    let t = std::time::Instant::now();
    let r = f();
    c.fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    r
}

/// Lê os streams de um tensor e confere o hash de cada um (os hashes em paralelo).
fn le_streams(arqs: &[Option<Arq>], sts: &[&St], nome: &str, tp: &Tempos, espera: Option<std::time::Duration>) -> R<Vec<Vec<u8>>> {
    let bufs = sts
        .iter()
        .map(|st| {
            let a = arqs[st.f].as_ref().ok_or_else(|| Erro("estrato não aberto".into()))?;
            mede(&tp[1], || le_faixa_espera(&a.f, st.o, st.l, nome, if st.f == 0 { None } else { espera }))
        })
        .collect::<R<Vec<_>>>()?;
    let hs: Vec<String> = mede(&tp[2], || {
        std::thread::scope(|sc| bufs.iter().map(|b| sc.spawn(move || sha256_hex(b))).collect::<Vec<_>>().into_iter().map(|j| j.join().unwrap()).collect())
    });
    if let Some(st) = sts.iter().zip(&hs).find(|(st, h)| **h != st.h).map(|(st, _)| st) {
        return falha(format!("hash não confere: {nome} ({})", st.p));
    }
    Ok(bufs)
}

/// Os bytes do tensor, a partir dos streams lidos: cru (talvez comprimido) ou residual.
fn monta(it: &Item, bufs: Vec<Vec<u8>>, prof: Option<u32>, d: u32, nt: usize, tp: &Tempos) -> R<Vec<u8>> {
    match it.tipo {
        Tipo::Cru => {
            let st = it.ler[0];
            let b = bufs.into_iter().next().unwrap();
            if st.c == "deflate" {
                mede(&tp[3], || crate::inflate(&b, st.r))
            } else {
                Ok(b)
            }
        }
        Tipo::Res { gr, h, linhas, cols } => {
            let gs = Grades { g: gr, base: &bufs[0] };
            let pl: Vec<&[u8]> = bufs[1..=d as usize].iter().map(Vec::as_slice).collect();
            let wg = if prof.is_none() {
                mede(&tp[3], || decodifica(&gs, &pl, &bufs[d as usize + 1], nt))?
            } else {
                mede(&tp[3], || residual::aproxima(&gs, &pl, d, nt))?
            };
            drop(bufs);
            Ok(mede(&tp[4], || {
                let w = match h {
                    Some(h) => permuta(&wg, linhas, cols, h, false),
                    None => wg,
                };
                let mut b = vec![0u8; 2 * w.len()];
                for (o, x) in b.chunks_exact_mut(2).zip(&w) {
                    o.copy_from_slice(&x.to_le_bytes());
                }
                b
            }))
        }
    }
}

enum Tipo {
    Cru,
    Res { gr: Grade, h: Option<usize>, linhas: usize, cols: usize },
}
/// Um shard de saída: nome, cabeçalho, bytes, entrada do índice e os tensores na ordem do arquivo.
struct Plano<'a> {
    nome: &'a str,
    hb: Vec<u8>,
    bytes: u64,
    fo: &'a Value,
    itens: Vec<Item<'a>>,
}
struct Item<'a> {
    e: crate::EntradaSt,
    ti: usize,
    tipo: Tipo,
    ler: Vec<&'a St>,
}

fn u(v: &Value, c: &str, onde: &str) -> R<u64> {
    v.get(c).and_then(Value::as_u64).ok_or_else(|| Erro(format!("{onde}: falta {c}")))
}
fn s<'a>(v: &'a Value, c: &str, onde: &str) -> R<&'a str> {
    v.get(c).and_then(Value::as_str).ok_or_else(|| Erro(format!("{onde}: falta {c}")))
}
fn nome_seguro(n: &str) -> bool {
    !n.is_empty() && n.len() < 256 && !n.contains('/') && !n.contains('\\') && n != "." && n != ".."
}

/// Lê o índice e os arquivos ao lado dele (o GGUF pode vir de outro lugar, `base`) e remonta cada
/// shard: até o fim (`prof` = None) os `.safetensors` originais, conferidos pelo SHA-256 de cada
/// shard; até o nível `d`, o centro da sub-célula de cada peso (A.14, leitura parcial), em BF16.
/// `saida` = None só confere, sem gravar.
pub fn volta(indice: &Path, base: Option<&Path>, saida: Option<&Path>, prof: Option<u32>, nt: usize, conferir_arquivos: bool) -> R<Volta> {
    volta_com(indice, base, saida, prof, nt, conferir_arquivos, None)
}

/// Como `volta`; com `espera`, os arquivos de estrato podem ainda estar chegando: cada um é
/// esperado até existir e cada stream até o arquivo cobri-lo (a ordem do índice é a ordem em que
/// o `junta` lê cada arquivo, do começo ao fim). Desiste após `espera` sem progresso.
#[allow(clippy::too_many_arguments)]
pub fn volta_com(indice: &Path, base: Option<&Path>, saida: Option<&Path>, prof: Option<u32>, nt: usize, conferir_arquivos: bool, espera: Option<std::time::Duration>) -> R<Volta> {
    if espera.is_some() && conferir_arquivos {
        return falha("--espera e --tudo não combinam: o arquivo inteiro ainda não chegou");
    }
    let ib = io(std::fs::read(indice), "índice")?;
    if ib.len() as u64 > crate::LIMITE_CABECALHO {
        return falha("índice grande demais");
    }
    let idx: Value = serde_json::from_slice(&ib).map_err(|e| Erro(format!("índice: {e}")))?;
    if idx.get("format").and_then(Value::as_str) != Some(FORMATO) || idx.get("desenho").and_then(Value::as_str) != Some(DESENHO) {
        return falha("índice: formato ou desenho desconhecido");
    }
    let k = u(&idx, "niveis", "índice")? as u32;
    if !(1..=8).contains(&k) || u(&idx, "estratos", "índice")? != k as u64 + 2 {
        return falha("índice: niveis/estratos inválidos");
    }
    if let Some(d) = prof {
        if d > k {
            return falha("profundidade maior que o número de níveis");
        }
    }
    let dir = indice.parent().unwrap_or(Path::new("."));
    let tp: Tempos = Default::default();
    // arquivos: o 0 é o GGUF externo, os outros um por estrato
    let lista = idx.get("arquivos").and_then(Value::as_array).ok_or_else(|| Erro("índice: falta arquivos".into()))?;
    if lista.len() != k as usize + 3 {
        return falha("índice: número de arquivos não bate com os estratos");
    }
    let mut arqs = Vec::new();
    for (i, a) in lista.iter().enumerate() {
        let nome = s(a, "nome", "arquivo")?;
        if !nome_seguro(nome) {
            return falha("índice: nome de arquivo inseguro");
        }
        let bytes = u(a, "bytes", nome)?;
        let est = if i == 0 {
            if a.get("externo").and_then(Value::as_str) != Some("gguf") || u(a, "s", nome)? != 0 {
                return falha("índice: o arquivo 0 tem que ser o GGUF externo");
            }
            -1
        } else {
            let st = u(a, "s", nome)?;
            if st != i as u64 - 1 || u(a, "parte", nome)? != 0 {
                return falha("índice: arquivos de estrato fora de ordem");
            }
            st as i64
        };
        // só abre o que a profundidade pede
        let precisa = i == 0 || i == 1 || est as u32 <= prof.unwrap_or(k + 1);
        if !precisa {
            arqs.push(None);
            continue;
        }
        let p = if i == 0 { base.map(Path::to_path_buf).unwrap_or_else(|| dir.join(nome)) } else { dir.join(nome) };
        let esp = if i == 0 { None } else { espera };
        if let Some(t) = esp {
            let t0 = std::time::Instant::now();
            while !p.exists() {
                if t0.elapsed() > t {
                    return falha(format!("{nome}: não apareceu"));
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        let f = io(File::open(&p), nome)?;
        let tam = io(f.metadata(), nome)?.len();
        if tam != bytes && (esp.is_none() || tam > bytes) {
            return falha(format!("{nome}: tamanho não bate com o índice"));
        }
        if i > 0 {
            let pre = le_faixa_espera(&f, 0, 16, nome, esp)?;
            if &pre[..8] != PREAMBULO || pre[8..12] != (est as u32).to_le_bytes() || pre[12..16] != [0; 4] {
                return falha(format!("{nome}: preâmbulo não bate com o índice"));
            }
        }
        if conferir_arquivos {
            // opcional: cada byte que entra na saída já é conferido pelo hash do seu stream, e cada
            // shard pelo SHA-256 dele; isto confere também o resto (cabeçalho do GGUF, tensores que o
            // STRATA não usa) e custa uma leitura a mais de tudo
            let (h, _) = mede(&tp[0], || sha256_arquivo(&f))?;
            if h != s(a, "sha256", nome)? {
                return falha(format!("{nome}: SHA-256 não confere com o índice"));
            }
        }
        arqs.push(Some(Arq { f, bytes, s: est }));
    }
    let gf = arqs[0].as_ref().unwrap();
    let g = cabecalho_gguf(&gf.f)?;
    // tensores e streams
    let ts = idx.get("tensors").and_then(Value::as_array).ok_or_else(|| Erro("índice: falta tensors".into()))?;
    let mut por_tensor: Vec<Vec<St>> = (0..ts.len()).map(|_| Vec::new()).collect();
    for e in idx.get("streams").and_then(Value::as_array).ok_or_else(|| Erro("índice: falta streams".into()))? {
        let st = St {
            t: u(e, "t", "stream")? as usize,
            s: u(e, "s", "stream")? as u32,
            p: s(e, "p", "stream")?.to_string(),
            f: u(e, "f", "stream")? as usize,
            o: u(e, "o", "stream")?,
            l: u(e, "l", "stream")?,
            r: u(e, "r", "stream")?,
            c: s(e, "c", "stream")?.to_string(),
            h: s(e, "h", "stream")?.to_string(),
        };
        if st.t >= ts.len() || st.f >= arqs.len() || st.s > k + 1 || st.h.len() != 64 {
            return falha("stream: índice fora das listas");
        }
        let es = if st.f == 0 { 0 } else { st.f as i64 - 1 };
        if es != st.s as i64 {
            return falha("stream: estrato não bate com o arquivo");
        }
        if let Some(a) = &arqs[st.f] {
            let ini = if st.f == 0 { g.dados } else { 16 };
            if st.o < ini || st.o.checked_add(st.l).is_none_or(|f| f > a.bytes) || a.s != if st.f == 0 { -1 } else { es } {
                return falha("stream: faixa fora do arquivo");
            }
        }
        if st.r > TETO_STREAM || st.l > TETO_STREAM {
            return falha("stream: grande demais");
        }
        por_tensor[st.t].push(st);
    }
    let fontes = idx.pointer("/fonte/shards").and_then(Value::as_array).ok_or_else(|| Erro("índice: falta fonte.shards".into()))?;
    let eng = base64::engine::general_purpose::STANDARD;
    // 1) o plano: tudo o que dá para conferir sem ler os dados
    let mut onde: HashMap<(u64, &str), usize> = HashMap::new();
    for (ti, t) in ts.iter().enumerate() {
        if let (Some(sh), Some(n)) = (t.get("shard").and_then(Value::as_u64), t.get("name").and_then(Value::as_str)) {
            onde.entry((sh, n)).or_insert(ti);
        }
    }
    let d = prof.unwrap_or(k);
    let mut usados = vec![false; ts.len()];
    let mut shards: Vec<Plano> = Vec::new();
    for (si, fo) in fontes.iter().enumerate() {
        let nome = s(fo, "nome", "shard")?;
        if !nome_seguro(nome) {
            return falha("shard: nome inseguro");
        }
        let hb = eng.decode(s(fo, "header_b64", nome)?).map_err(|_| Erro(format!("{nome}: header_b64 inválido")))?;
        let (es, total) = valida_safetensors(&hb, None)?;
        let bytes_shard = 8 + hb.len() as u64 + total;
        if bytes_shard != u(fo, "bytes", nome)? {
            return falha(format!("{nome}: tamanho não bate"));
        }
        let mut itens = Vec::new();
        for e in es {
            let ti = *onde.get(&(si as u64, e.name.as_str())).ok_or_else(|| Erro(format!("{}: tensor do shard sem entrada no índice", e.name)))?;
            if std::mem::replace(&mut usados[ti], true) {
                return falha("tensor usado duas vezes");
            }
            let tj = &ts[ti];
            if tj.get("dtype").and_then(Value::as_str) != Some(e.dtype.as_str()) || u(tj, "bytes", &e.name)? != e.end - e.begin {
                return falha(format!("{}: dtype ou tamanho não bate com o shard", e.name));
            }
            let sts = &por_tensor[ti];
            let (tipo, ler) = match tj.get("residual") {
                None | Some(Value::Null) => {
                    let [st] = &sts[..] else { return falha(format!("{}: tensor cru com streams demais", e.name)) };
                    if st.s != 0 || st.p != "raw" || st.r != e.end - e.begin || !matches!(st.c.as_str(), "raw" | "deflate") || (st.c == "raw" && st.l != st.r) {
                        return falha(format!("{}: stream cru inválido", e.name));
                    }
                    (Tipo::Cru, vec![st])
                }
                Some(r) => {
                    let gr = r.get("grade").and_then(Value::as_str).and_then(Grade::de_nome).ok_or_else(|| Erro(format!("{}: grade inválida", e.name)))?;
                    let h = match r.get("permuta") {
                        Some(Value::Null) | None => None,
                        Some(v) => Some(v.as_u64().filter(|&h| h >= 1).ok_or_else(|| Erro("permuta inválida".into()))? as usize),
                    };
                    if e.dtype != "BF16" || e.shape.len() != 2 || e.shape[1] % 256 != 0 || h.is_some_and(|h| !(e.shape[0] as usize).is_multiple_of(2 * h)) {
                        return falha(format!("{}: tensor residual com forma inválida", e.name));
                    }
                    let (linhas, cols) = (e.shape[0] as usize, e.shape[1] as usize);
                    let n = linhas * cols;
                    let partes: Vec<String> = std::iter::once("q".to_string()).chain((1..=k).map(|j| format!("n{j}"))).chain(std::iter::once("f".into())).collect();
                    if sts.len() != partes.len() || sts.iter().zip(&partes).enumerate().any(|(j, (st, p))| &st.p != p || st.s != j as u32) {
                        return falha(format!("{}: streams do tensor residual fora da regra", e.name));
                    }
                    let ql = (n / 256 * gr.bytes_bloco()) as u64;
                    if sts[0].l != ql || sts[0].r != ql || sts[0].c != "raw" || sts[1..=k as usize].iter().any(|s| s.l != n as u64 / 8 || s.r != n as u64 / 8 || s.c != "raw") || sts[k as usize + 1].c != "fecho" || sts[k as usize + 1].r != 2 * n as u64 {
                        return falha(format!("{}: tamanhos dos streams fora da regra", e.name));
                    }
                    // até o nível d: q e os planos 1..d; inteiro: também o fecho
                    let ler = sts[..=d as usize].iter().chain(prof.is_none().then(|| &sts[k as usize + 1])).collect();
                    (Tipo::Res { gr, h, linhas, cols }, ler)
                }
            };
            itens.push(Item { e, ti, tipo, ler });
        }
        shards.push(Plano { nome, hb, bytes: bytes_shard, fo, itens });
    }
    if usados.iter().any(|x| !x) {
        return falha("índice com tensor fora de qualquer shard");
    }
    // 2) três fios em fila: um lê e confere os streams do próximo tensor, outro decodifica, e este
    // calcula os hashes e grava o anterior
    let (arqs, tp, shards) = (&arqs, &tp, &shards);
    let (out, ntens) = std::thread::scope(|sc| -> R<_> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<R<Vec<Vec<u8>>>>(0);
        let (tx2, rx2) = std::sync::mpsc::sync_channel::<R<Vec<u8>>>(0);
        sc.spawn(move || {
            for it in shards.iter().flat_map(|sh| &sh.itens) {
                let r = le_streams(arqs, &it.ler, &it.e.name, tp, espera);
                let parou = r.is_err();
                if tx.send(r).is_err() || parou {
                    break;
                }
            }
        });
        sc.spawn(move || {
            for it in shards.iter().flat_map(|sh| &sh.itens) {
                let r = rx.recv().map_err(|_| Erro("leitura parou".into())).and_then(|b| b).and_then(|bufs| monta(it, bufs, prof, d, nt, tp));
                let parou = r.is_err();
                if tx2.send(r).is_err() || parou {
                    break;
                }
            }
        });
        let mut out = Vec::new();
        let mut ntens = 0;
        for Plano { nome, hb, bytes: bytes_shard, fo, itens } in shards {
            let mut w = match saida {
                Some(d) => Some(std::io::BufWriter::with_capacity(8 << 20, io(File::create(d.join(nome)), nome)?)),
                None => None,
            };
            let mut hs = Sha256::new();
            let mut poe = |b: &[u8], w: &mut Option<std::io::BufWriter<File>>| -> R<()> {
                mede(&tp[5], || hs.update(b));
                if let Some(w) = w {
                    mede(&tp[6], || io(w.write_all(b), nome))?;
                }
                Ok(())
            };
            poe(&(hb.len() as u64).to_le_bytes(), &mut w)?;
            poe(hb, &mut w)?;
            for it in itens {
                let bytes = rx2.recv().map_err(|_| Erro("decodificação parou".into()))??;
                // o hash do tensor corre ao lado do hash do shard e da escrita
                let (ht, r) = std::thread::scope(|s2| {
                    let j = prof.is_none().then(|| s2.spawn(|| sha256_hex(&bytes)));
                    let r = poe(&bytes, &mut w);
                    (j.map(|j| j.join().unwrap()), r)
                });
                r?;
                if let Some(ht) = ht {
                    if ht != s(&ts[it.ti], "sha256", &it.e.name)? {
                        return falha(format!("{}: SHA-256 do tensor não confere", it.e.name));
                    }
                }
                ntens += 1;
            }
            if let Some(mut w) = w {
                io(w.flush(), nome)?;
            }
            let h = hex(&hs.finalize());
            if prof.is_none() && h != s(fo, "sha256", nome)? {
                return falha(format!("{nome}: SHA-256 do shard não confere"));
            }
            out.push((nome.to_string(), h, *bytes_shard));
        }
        Ok((out, ntens))
    })?;
    Ok(Volta { shards: out, tensores: ntens, tempos: FASES.iter().zip(tp).map(|(n, c)| (*n, c.load(Relaxed) as f64 / 1e9)).collect() })
}
