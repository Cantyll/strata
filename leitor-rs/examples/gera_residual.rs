//! Gera os vetores públicos do modo residual (strata/0.3, A.14) em tests/vetores/residual/:
//! o modelo sintético de tests/comum (safetensors + GGUF), dois .strata válidos e os hostis.
//!   cargo run --release --example gera_residual
use serde_json::{json, Value};
use std::fs;
use strata_ref::{abre, cabecalho_id, decode, para_gguf, residual, sha256_hex, to_safetensors, Opcoes, MAGIC};

#[path = "../tests/comum/mod.rs"]
mod comum;

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/vetores/residual/");

fn separa(f: &[u8]) -> (Value, Vec<u8>) {
    let hl = u32::from_le_bytes(f[8..12].try_into().unwrap()) as usize;
    (serde_json::from_slice(&f[12..12 + hl]).unwrap(), f[12 + hl..].to_vec())
}
fn junta(h: &Value, corpo: &[u8]) -> Vec<u8> {
    let hj = serde_json::to_vec(h).unwrap();
    let mut f = MAGIC.to_vec();
    f.push(3);
    f.extend_from_slice(&(hj.len() as u32).to_le_bytes());
    f.extend_from_slice(&hj);
    f.extend_from_slice(corpo);
    f
}
/// índice do stream (tensor `t`, parte `p`) no cabeçalho
fn stream(h: &Value, t: u64, p: &str) -> usize {
    h["streams"].as_array().unwrap().iter().position(|s| s["t"] == t && s["p"] == p).unwrap()
}
fn sem_hashes(h: &mut Value) {
    h.as_object_mut().unwrap().remove("integridade");
    for t in h["tensors"].as_array_mut().unwrap() {
        t.as_object_mut().unwrap().remove("sha256");
    }
    for s in h["streams"].as_array_mut().unwrap() {
        s.as_object_mut().unwrap().remove("h");
    }
}
/// Troca bytes do corpo de um stream (via `mexe`) e regrava o `h` dele, se houver.
fn mexe_stream(f: &[u8], t: u64, p: &str, hashes: bool, mexe: impl Fn(&mut [u8])) -> Vec<u8> {
    let (mut h, mut corpo) = separa(f);
    if !hashes {
        sem_hashes(&mut h);
    }
    let i = stream(&h, t, p);
    let (o, l) = (h["streams"][i]["o"].as_u64().unwrap() as usize, h["streams"][i]["l"].as_u64().unwrap() as usize);
    mexe(&mut corpo[o..o + l]);
    if hashes {
        h["streams"][i]["h"] = json!(sha256_hex(&corpo[o..o + l]));
    }
    junta(&h, &corpo)
}

fn main() {
    let (st, gg) = comum::modelo();
    fs::create_dir_all(DIR).unwrap();
    fs::write(format!("{DIR}modelo.safetensors"), &st).unwrap();
    fs::write(format!("{DIR}modelo.gguf"), &gg).unwrap();
    let op = Opcoes::default();
    let mut vetores = Vec::new();

    let mut validos = Vec::new();
    for (nome, k) in [("ok-residual.strata", 2), ("ok-residual-k0.strata", 0)] {
        let e = residual::escreve(&st, &gg, k, 1).unwrap();
        assert_eq!(e.arquivo, residual::escreve(&st, &gg, k, 4).unwrap().arquivo, "o escritor depende do número de threads");
        let f = e.arquivo;
        assert_eq!(to_safetensors(&f, &op).unwrap(), st);
        assert_eq!(para_gguf(&f[..e.prefixos[0] as usize], &op).unwrap(), gg);
        let a = abre(&f, &op).unwrap();
        let prefixos: Vec<Value> = (0..=k + 1)
            .map(|d| {
                let n = a.bytes_ate(d) as usize;
                let (_, ts) = decode(&f[..n], d, &op).unwrap();
                json!({"profundidade": d, "bytes": n, "sha256_tensores": sha256_hex(&ts.concat())})
            })
            .collect();
        vetores.push(json!({
            "arquivo": nome, "espera": "ok", "niveis": k,
            "sha256_safetensors": sha256_hex(&st), "sha256_gguf": sha256_hex(&gg),
            "id": cabecalho_id(&f).unwrap(), "prefixos": prefixos,
        }));
        fs::write(format!("{DIR}{nome}"), &f).unwrap();
        validos.push(f);
    }
    let ok = &validos[0];
    // tensor 1 = q_proj (Q4_K, permuta 2), 3 = input_layernorm (crua, "f32" no GGUF), 7 = gguf:rope_freqs (extra)
    let cab = |muda: &dyn Fn(&mut Value)| {
        let (mut h, corpo) = separa(ok);
        muda(&mut h);
        junta(&h, &corpo)
    };
    let tabelas = |b: &[u8]| residual::Tabelas::le(b).unwrap().1;
    let hostis: Vec<(&str, &str, Vec<u8>)> = vec![
        ("ruim-grade.strata", "grade desconhecida (Q5_K)", cab(&|h| h["tensors"][1]["residual"]["grade"] = json!("Q5_K"))),
        ("ruim-permuta.strata", "permuta que não divide as linhas", cab(&|h| h["tensors"][1]["residual"]["permuta"] = json!(3))),
        ("ruim-residual-1d.strata", "residual num tensor 1-D", cab(&|h| h["tensors"][3]["residual"] = json!({"grade": "Q4_K", "permuta": null}))),
        ("ruim-fecho-cru.strata", "stream f com codificação raw", cab(&|h| {
            let i = stream(h, 1, "f");
            h["streams"][i]["c"] = json!("raw");
        })),
        ("ruim-niveis.strata", "niveis maior que os planos gravados", cab(&|h| {
            h["niveis"] = json!(3);
            h["estratos"] = json!(5);
        })),
        ("ruim-estratos.strata", "estratos fora da regra", cab(&|h| h["estratos"] = json!(5))),
        ("ruim-q-tamanho.strata", "stream q com tamanho cru errado", cab(&|h| {
            let i = stream(h, 1, "q");
            h["streams"][i]["r"] = json!(h["streams"][i]["r"].as_u64().unwrap() - 144);
        })),
        ("ruim-gguf-parte.strata", "parte q do GGUF apontando para tensor sem grade", cab(&|h| {
            let i = h["gguf"]["partes"].as_array().unwrap().iter().position(|p| p["como"] == "q").unwrap();
            h["gguf"]["partes"][i]["t"] = json!(3);
        })),
        ("ruim-sem-gguf.strata", "tensores extras sem a seção gguf", cab(&|h| {
            h.as_object_mut().unwrap().remove("gguf");
        })),
        ("ruim-extra.strata", "valor de extra desconhecido", cab(&|h| h["tensors"][7]["extra"] = json!("onnx"))),
        // corpo mexido sem hashes: quem recusa é a decodificação do fecho
        ("ruim-tabela-fecho.strata", "frequências da tabela do fecho não somam 2^15", mexe_stream(ok, 1, "f", false, |b| b[7] = b[7].wrapping_add(1))),
        ("ruim-bloco-fecho.strata", "tamanho de bloco do fecho maior que o stream", mexe_stream(ok, 1, "f", false, |b| {
            let i = tabelas(b);
            b[i] = b[i].wrapping_add(1);
        })),
        ("ruim-corpo-fecho.strata", "rANS do fecho adulterado (estado final)", mexe_stream(ok, 1, "f", false, |b| {
            let i = tabelas(b) + 4 + (b.len() - tabelas(b) - 4) / 2;
            b[i] ^= 0x10;
        })),
        // plano trocado com o h regravado: só o sha256 do tensor pega
        ("ruim-plano.strata", "um bit de plano trocado (h regravado)", mexe_stream(ok, 1, "n1", true, |b| b[5] ^= 4)),
    ];
    for (nome, sobre, f) in hostis {
        assert!(decode(&f, u32::MAX, &op).is_err(), "{nome} foi aceito");
        vetores.push(json!({"arquivo": nome, "espera": "recusa", "sobre": sobre}));
        fs::write(format!("{DIR}{nome}"), &f).unwrap();
    }
    // GGUF com SHA errado: os tensores saem certos, só a remontagem do GGUF recusa
    let f = cab(&|h| h["gguf"]["sha256"] = json!("0".repeat(64)));
    assert!(to_safetensors(&f, &op).is_ok() && para_gguf(&f, &op).is_err());
    vetores.push(json!({"arquivo": "ruim-gguf-sha.strata", "espera": "recusa-gguf", "sobre": "sha256 do GGUF errado"}));
    fs::write(format!("{DIR}ruim-gguf-sha.strata"), &f).unwrap();

    let man = json!({
        "sobre": "Vetores do modo residual (strata/0.3, docs/02 A.14), feitos por leitor-rs/examples/gera_residual.rs a partir de modelo.safetensors e modelo.gguf (Llama sintético, tests/comum). \"ok\": volta para o .safetensors e o GGUF com os SHA-256 indicados; cada prefixo, lido na profundidade indicada, dá tensores cuja concatenação tem o SHA-256 indicado. \"recusa\": decode tem que recusar. \"recusa-gguf\": os tensores saem certos e a remontagem do GGUF tem que recusar.",
        "vetores": vetores,
    });
    fs::write(format!("{DIR}vetores.json"), serde_json::to_string_pretty(&man).unwrap() + "\n").unwrap();
    println!("{} vetores em {DIR}", man["vetores"].as_array().unwrap().len());
}
