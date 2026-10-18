//! Modo residual (strata/0.3, A.14): vetores públicos de tests/vetores/residual, escritor
//! determinístico, remontagem do GGUF, leitura em fluxo e milhares de arquivos mutados.
use serde_json::Value;
use std::fs;
use strata_ref::{abre, blocos_q4kr, blocos_q6kr, cabecalho_id, decode, para_gguf, residual, sha256_hex, to_safetensors, Fluxo, Opcoes, R};

mod comum;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/vetores/residual/");

fn ler(nome: &str) -> Vec<u8> {
    fs::read(format!("{DIR}{nome}")).unwrap()
}
fn manifesto() -> Value {
    serde_json::from_slice(&ler("vetores.json")).unwrap()
}
fn em_fluxo(f: &[u8], pedaco: usize, prof: u32, op: &Opcoes) -> R<Vec<Vec<u8>>> {
    let cab = if f.len() >= 12 { (12 + u32::from_le_bytes(f[8..12].try_into().unwrap()) as usize).min(f.len()) } else { f.len() };
    let mut fx = Fluxo::novo(&f[..cab], prof, op)?;
    for c in f[cab..].chunks(pedaco.max(1)) {
        fx.alimenta(c)?;
    }
    Ok(fx.termina()?.1)
}
fn igual(a: &R<Vec<Vec<u8>>>, b: &R<Vec<Vec<u8>>>) -> bool {
    match (a, b) {
        (Ok(x), Ok(y)) => x == y,
        (Err(_), Err(_)) => true,
        _ => false,
    }
}

#[test]
fn modelo_e_escritor_deterministicos() {
    let (st, gg) = comum::modelo();
    assert_eq!(st, ler("modelo.safetensors"), "tests/comum mudou: rode `cargo run --example gera_residual`");
    assert_eq!(gg, ler("modelo.gguf"));
    for (nome, k) in [("ok-residual.strata", 2), ("ok-residual-k0.strata", 0)] {
        for nt in [1, 3] {
            assert_eq!(residual::escreve(&st, &gg, k, nt).unwrap().arquivo, ler(nome), "{nome} com {nt} threads");
        }
    }
    assert!(residual::escreve(&st, &gg, 9, 1).is_err());
}

#[test]
fn vetores() {
    let op = Opcoes { threads: 2, ..Default::default() };
    for v in manifesto()["vetores"].as_array().unwrap() {
        let nome = v["arquivo"].as_str().unwrap();
        let f = ler(nome);
        match v["espera"].as_str().unwrap() {
            "recusa" => {
                assert!(decode(&f, u32::MAX, &op).is_err(), "{nome}: deveria ser recusado");
                assert!(em_fluxo(&f, 97, u32::MAX, &op).is_err(), "{nome}: o fluxo aceitou");
                continue;
            }
            "recusa-gguf" => {
                assert!(to_safetensors(&f, &op).is_ok() && para_gguf(&f, &op).is_err(), "{nome}");
                continue;
            }
            _ => {}
        }
        assert_eq!(sha256_hex(&to_safetensors(&f, &op).unwrap()), v["sha256_safetensors"].as_str().unwrap(), "{nome}");
        let id = v["id"].as_str().unwrap();
        assert_eq!(cabecalho_id(&f).unwrap(), id, "{nome}");
        let exigente = Opcoes { id_esperado: Some(id.into()), exigir_integridade: true, ..Default::default() };
        assert!(to_safetensors(&f, &exigente).is_ok());
        let a = abre(&f, &op).unwrap();
        assert_eq!(a.niveis as u64, v["niveis"].as_u64().unwrap());
        for p in v["prefixos"].as_array().unwrap() {
            let (d, n) = (p["profundidade"].as_u64().unwrap() as u32, p["bytes"].as_u64().unwrap() as usize);
            assert_eq!(a.bytes_ate(d), n as u64, "{nome} d={d}");
            let (_, ts) = decode(&f[..n], d, &op).unwrap_or_else(|e| panic!("{nome} d={d}: {e}"));
            assert_eq!(sha256_hex(&ts.concat()), p["sha256_tensores"].as_str().unwrap(), "{nome} d={d}");
            assert!(decode(&f[..n - 1], d, &op).is_err(), "{nome} d={d}: leu sem o último byte");
            for pedaco in [1, 7, 4096] {
                assert_eq!(em_fluxo(&f[..n], pedaco, d, &op).unwrap(), ts, "{nome} d={d} fluxo em {pedaco}");
            }
            // o arquivo inteiro lido raso dá o mesmo que o prefixo
            assert_eq!(decode(&f, d, &op).unwrap().1, ts, "{nome} d={d} inteiro");
        }
        // o GGUF sai do estrato 0 e de mais nada
        let p0 = a.bytes_ate(0) as usize;
        assert_eq!(sha256_hex(&para_gguf(&f[..p0], &op).unwrap()), v["sha256_gguf"].as_str().unwrap(), "{nome}");
        assert!(para_gguf(&f[..p0 - 1], &op).is_err(), "{nome}: GGUF sem o último byte do estrato 0");
    }
}

/// Na profundidade d ≤ K, cada peso finito que cai na célula do estrato 0 sai a no máximo meia
/// sub-célula do nível d (|a|/2^(d+1)) do original, mais o arredondamento para BF16; na
/// profundidade K+1 sai o original. Os tensores sem grade não mudam com a profundidade.
#[test]
fn profundidades_ficam_na_celula() {
    let (_, gg) = comum::modelo();
    let g = residual::le_gguf(&gg).unwrap();
    let f = ler("ok-residual.strata");
    let op = Opcoes::default();
    let a = abre(&f, &op).unwrap();
    let todos = decode(&f, u32::MAX, &op).unwrap().1;
    let bf = |b: &[u8]| -> Vec<u16> { b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect() };
    let val = |u: u16| f32::from_bits((u as u32) << 16);
    let (mut conferidos, mut fora) = (0u64, 0u64);
    for d in 0..=a.niveis + 1 {
        let ts = decode(&f, d, &op).unwrap().1;
        for (i, t) in a.tensores.iter().enumerate() {
            let Some((gnome, grade, h)) = (match t.name.as_str() {
                "model.embed_tokens.weight" => Some(("token_embd.weight", residual::Grade::Q6K, None)),
                "model.layers.0.self_attn.q_proj.weight" => Some(("blk.0.attn_q.weight", residual::Grade::Q4K, Some(2))),
                "model.layers.0.self_attn.k_proj.weight" => Some(("blk.0.attn_k.weight", residual::Grade::Q4K, Some(1))),
                _ => None,
            }) else {
                assert_eq!(ts[i], todos[i], "{}: tensor sem grade mudou com a profundidade", t.name);
                continue;
            };
            if d == a.niveis + 1 {
                assert_eq!(ts[i], todos[i]);
                continue;
            }
            let (linhas, cols) = (t.shape[0] as usize, t.shape[1] as usize);
            let em_gguf = |u: Vec<u16>| match h {
                Some(h) => residual::permuta(&u, linhas, cols, h, true),
                None => u,
            };
            let (x, w) = (em_gguf(bf(&ts[i])), em_gguf(bf(&todos[i])));
            let tg = g.tensores.iter().find(|t| t.nome == gnome).unwrap();
            let bb = grade.bytes_bloco();
            let base = &gg[(g.dados + tg.off) as usize..][..x.len() / 256 * bb];
            let (mut am, mut mm, mut qm) = ([0f32; 256], [0f32; 256], [0i32; 256]);
            for sb in 0..x.len() / 256 {
                residual::celulas(grade, &base[sb * bb..(sb + 1) * bb], &mut am, &mut mm, &mut qm);
                for j in 0..256 {
                    let (xv, wv, aa) = (val(x[sb * 256 + j]), val(w[sb * 256 + j]), am[j]);
                    let c = aa * qm[j] as f32 - mm[j];
                    if !wv.is_finite() || aa == 0.0 || (wv - c).abs() > aa.abs() / 2.0 {
                        fora += 1;
                        continue;
                    }
                    let lim = aa.abs() / (1 << (d + 1)) as f32 + xv.abs() / 256.0 + 1e-6;
                    assert!((xv - wv).abs() <= lim, "{} d={d} peso {}: {xv} longe de {wv} (a={aa})", t.name, sb * 256 + j);
                    conferidos += 1;
                }
            }
        }
    }
    assert!(conferidos > 10 * fora, "{conferidos} conferidos, {fora} fora da célula");
}

fn q_gguf6(gg: &[u8], g: &residual::Gguf, nome: &str) -> Vec<u8> {
    let t = g.tensores.iter().find(|t| t.nome == nome).unwrap();
    let n: u64 = t.ne.iter().product();
    gg[(g.dados + t.off) as usize..(g.dados + t.off + n / 256 * 210) as usize].to_vec()
}

#[test]
fn blocos_q4kr_intercalam_planos() {
    let f = ler("ok-residual.strata");
    let op = Opcoes::default();
    let a = abre(&f, &op).unwrap();
    let (_, gg) = comum::modelo();
    let g = residual::le_gguf(&gg).unwrap();
    let q_gguf = |nome: &str| {
        let t = g.tensores.iter().find(|t| t.nome == nome).unwrap();
        let n: u64 = t.ne.iter().product();
        gg[(g.dados + t.off) as usize..(g.dados + t.off + n / 256 * 144) as usize].to_vec()
    };
    let i = a.tensores.iter().position(|t| t.name.ends_with("q_proj.weight")).unwrap();
    let q = q_gguf("blk.0.attn_q.weight");
    assert_eq!(blocos_q4kr(&f, &a, i, 0).unwrap(), q, "k=0 é o Q4_K do GGUF");
    let b2 = blocos_q4kr(&f[..a.bytes_ate(2) as usize], &a, i, 2).unwrap();
    assert_eq!(b2.len(), q.len() / 144 * (144 + 64));
    let b1 = blocos_q4kr(&f, &a, i, 1).unwrap();
    for (sb, (x1, x2)) in b1.chunks_exact(176).zip(b2.chunks_exact(208)).enumerate() {
        assert_eq!(&x2[..144], &q[sb * 144..sb * 144 + 144]);
        assert_eq!(x1, &x2[..176], "o plano 1 vem logo depois do Q4_K");
    }
    assert!(blocos_q4kr(&f, &a, i, 3).is_err(), "só há 2 níveis");
    assert!(blocos_q4kr(&f[..a.bytes_ate(1) as usize], &a, i, 2).is_err(), "plano 2 fora do prefixo");
    let emb = a.tensores.iter().position(|t| t.name == "model.embed_tokens.weight").unwrap();
    assert!(blocos_q4kr(&f, &a, emb, 1).is_err(), "embedding é Q6_K");
    // Q6_K_R: 210 bytes do Q6_K e os mesmos planos
    let q6 = q_gguf6(&gg, &g, "token_embd.weight");
    let e2 = blocos_q6kr(&f, &a, emb, 2).unwrap();
    assert_eq!(blocos_q6kr(&f, &a, emb, 0).unwrap(), q6);
    let e1 = blocos_q6kr(&f, &a, emb, 1).unwrap();
    for (sb, (x1, x2)) in e1.chunks_exact(242).zip(e2.chunks_exact(274)).enumerate() {
        assert_eq!(&x2[..210], &q6[sb * 210..sb * 210 + 210]);
        assert_eq!(x1, &x2[..242]);
    }
    assert!(blocos_q6kr(&f, &a, i, 1).is_err(), "q_proj é Q4_K");
}

#[test]
fn mutacoes_dao_o_mesmo_que_decode() {
    let mut x = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = move |n: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n.max(1) as u64) as usize
    };
    let op = Opcoes { threads: 2, ..Default::default() };
    let (mut aceitos, mut casos) = (0, 0);
    for nome in ["ok-residual.strata", "ok-residual-k0.strata"] {
        let orig = ler(nome);
        let certos = decode(&orig, u32::MAX, &op).unwrap().1;
        let hl = 12 + u32::from_le_bytes(orig[8..12].try_into().unwrap()) as usize;
        for _ in 0..1000 {
            let mut f = orig.clone();
            match rnd(5) {
                0 => {
                    let i = rnd(f.len());
                    f[i] ^= 1 << rnd(8);
                }
                // só no corpo: a maioria das mutações no cabeçalho morre no JSON
                1 => {
                    let i = hl + rnd(f.len() - hl);
                    f[i] ^= 1 << rnd(8);
                }
                2 => f.truncate(rnd(f.len())),
                3 => {
                    for _ in 0..1 + rnd(4) {
                        let i = rnd(f.len());
                        f[i] = rnd(256) as u8;
                    }
                }
                _ => f.extend(std::iter::repeat_n(rnd(256) as u8, 1 + rnd(8))),
            }
            let prof = [0, 1, 2, u32::MAX][rnd(4)];
            let lote = decode(&f, prof, &op).map(|x| x.1);
            let fluxo = em_fluxo(&f, 1 + rnd(300), prof, &op);
            assert!(igual(&lote, &fluxo), "{nome}: decode {:?} × fluxo {:?}", lote.as_ref().err(), fluxo.as_ref().err());
            // com os hashes, nada adulterado passa: o que é aceito por inteiro é o original
            if let (u32::MAX, Ok(ts)) = (prof, &lote) {
                assert_eq!(ts, &certos, "{nome}: aceito com pesos errados");
            }
            let _ = para_gguf(&f, &op);
            aceitos += lote.is_ok() as u32;
            casos += 1;
        }
    }
    println!("{casos} mutações, {aceitos} aceitas pelos dois");
}
