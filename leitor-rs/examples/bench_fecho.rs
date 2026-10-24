//! Mede a decodificação do fecho de um tensor de um índice B.10 (sem o resto do `junta`).
//!   cargo run --release --example bench_fecho -- <model.strata.index.json> <base.gguf> <tensor> [threads]
use serde_json::Value;
use std::fs::File;
use std::os::unix::fs::FileExt;
use strata_ref::residual::{decodifica, Grade, Grades};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let idx: Value = serde_json::from_slice(&std::fs::read(&a[1]).unwrap()).unwrap();
    let dir = std::path::Path::new(&a[1]).parent().unwrap();
    let nt: usize = a.get(4).map(|x| x.parse().unwrap()).unwrap_or(1);
    let ts = idx["tensors"].as_array().unwrap();
    let ti = ts.iter().position(|t| t["name"] == a[3].as_str()).expect("tensor");
    let arqs: Vec<File> = idx["arquivos"].as_array().unwrap().iter().enumerate().map(|(i, f)| File::open(if i == 0 { a[2].clone().into() } else { dir.join(f["nome"].as_str().unwrap()) }).unwrap()).collect();
    let mut sts: Vec<&Value> = idx["streams"].as_array().unwrap().iter().filter(|s| s["t"].as_u64() == Some(ti as u64)).collect();
    sts.sort_by_key(|s| s["s"].as_u64());
    let ler = |s: &Value| {
        let mut b = vec![0u8; s["l"].as_u64().unwrap() as usize];
        arqs[s["f"].as_u64().unwrap() as usize].read_exact_at(&mut b, s["o"].as_u64().unwrap()).unwrap();
        b
    };
    let bufs: Vec<Vec<u8>> = sts.iter().map(|s| ler(s)).collect();
    let k = bufs.len() - 2;
    let g = Grade::de_nome(ts[ti]["residual"]["grade"].as_str().unwrap()).unwrap();
    let gs = Grades { g, base: &bufs[0] };
    let pl: Vec<&[u8]> = bufs[1..=k].iter().map(Vec::as_slice).collect();
    let t = std::time::Instant::now();
    let w = decodifica(&gs, &pl, &bufs[k + 1], nt).unwrap();
    let s = t.elapsed().as_secs_f64();
    let h = strata_ref::sha256_hex(&w.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
    println!("{}: {} pesos, {:.2} s, {:.1} ns/peso com {nt} fio(s), sha256 {}", a[3], w.len(), s, s * 1e9 / w.len() as f64, &h[..16]);
}
