//! Mede o modo residual de verdade: codifica cada matriz k-quant do GGUF com k níveis e o fecho,
//! confere a volta bit a bit e imprime os bits/peso por parte.
//!
//!   cargo run --release --example residual_medir -- modelo.safetensors base.gguf [k]
use std::time::Instant;
use strata_ref::residual::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let st = std::fs::read(&a[1]).unwrap();
    let gb = std::fs::read(&a[2]).unwrap();
    let k: u32 = a.get(3).map_or(3, |x| x.parse().unwrap());
    let hl = u64::from_le_bytes(st[..8].try_into().unwrap()) as usize;
    let (es, _) = strata_ref::valida_safetensors(&st[8..8 + hl], None).unwrap();
    let dados = &st[8 + hl..];
    let g = le_gguf(&gb).unwrap();
    let (nh, nkv): (usize, usize) = (g.kv["llama.attention.head_count"].parse().unwrap(), g.kv["llama.attention.head_count_kv"].parse().unwrap());
    let nt = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut por: std::collections::BTreeMap<&str, [f64; 5]> = Default::default();
    let t0 = Instant::now();
    for t in &g.tensores {
        let grade = match t.tipo {
            GGML_Q4_K => Grade::Q4K,
            GGML_Q6_K => Grade::Q6K,
            _ => continue,
        };
        let hf = nome_hf(&t.nome).unwrap();
        let e = es.iter().find(|e| e.name == hf).unwrap();
        let (linhas, cols) = (e.shape[0] as usize, e.shape[1] as usize);
        let w: Vec<u16> = dados[e.begin as usize..e.end as usize].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let h = if t.nome.contains("attn_q") { Some(nh) } else if t.nome.contains("attn_k") { Some(nkv) } else { None };
        let wg = match h { Some(h) => permuta(&w, linhas, cols, h, true), None => w.clone() };
        let n = wg.len();
        let base = &gb[(g.dados + t.off) as usize..(g.dados + t.off) as usize + n / 256 * grade.bytes_bloco()];
        let gr = Grades { g: grade, base };
        let c = codifica(&gr, &wg, k, nt).unwrap();
        let pl: Vec<&[u8]> = c.planos.iter().map(Vec::as_slice).collect();
        let t1 = Instant::now();
        let volta = decodifica(&gr, &pl, &c.fecho, nt).unwrap();
        let dt = t1.elapsed().as_secs_f64();
        assert!(volta == wg, "{}: volta diferente", t.nome);
        let x = por.entry(grade.nome()).or_default();
        x[0] += n as f64;
        x[1] += base.len() as f64 * 8.0;
        x[2] += k as f64 * n as f64;
        x[3] += c.fecho.len() as f64 * 8.0;
        x[4] += c.escapes as f64;
        println!("{:28} {:>10} pesos  fecho {:.3} b/peso  escapes {}  decodifica {:.0} M pesos/s", t.nome, n, c.fecho.len() as f64 * 8.0 / n as f64, c.escapes, n as f64 / dt / 1e6);
    }
    let mut tot = [0f64; 5];
    for (g, x) in &por {
        println!("{g}: {:.0} M pesos, base {:.3} + níveis {:.3} + fecho {:.3} = {:.3} b/peso, {} escapes", x[0] / 1e6, x[1] / x[0], x[2] / x[0], x[3] / x[0], (x[1] + x[2] + x[3]) / x[0], x[4]);
        for i in 0..5 {
            tot[i] += x[i];
        }
    }
    println!("total: {:.0} M pesos, base {:.3} + níveis {:.3} + fecho {:.3} = {:.3} b/peso ({:.0} s)", tot[0] / 1e6, tot[1] / tot[0], tot[2] / tot[0], tot[3] / tot[0], (tot[1] + tot[2] + tot[3]) / tot[0], t0.elapsed().as_secs_f64());
}
