//! Mede quanto custa dividir o expoente de cada tensor em blocos rANS independentes (spec B.5):
//! cada bloco com o próprio estado final e um offset de 4 bytes, todos com a tabela do tensor.
//! Blocos independentes podem ser decodificados em paralelo (vários núcleos hoje, GPU depois).
//!
//!   cargo run --release --example rans_blocos -- modelo.safetensors [modelo.strata]
//!
//! Com o .strata, confere que o bloco único reproduz exatamente o tamanho dos streams de expoente
//! gravados pelo codificador em JS. Só mede: não muda o formato.
use std::time::Instant;

const BITS: u32 = 15;
const M: u32 = 1 << BITS;
const L: u32 = 1 << 23;

/// Mesma quantização do codificador (ransFreqs em src/strata-core.js).
fn freqs(cont: &[f64; 256], n: usize) -> [u32; 256] {
    let mut f = [0u32; 256];
    if n == 0 {
        return f;
    }
    let mut soma: i64 = 0;
    for s in 0..256 {
        if cont[s] > 0.0 {
            // Math.round do JS: arredonda .5 para cima
            f[s] = ((cont[s] * M as f64 / n as f64 + 0.5).floor() as u32).max(1);
            soma += f[s] as i64;
        }
    }
    while soma != M as i64 {
        let (mut best, mut custo) = (usize::MAX, f64::INFINITY);
        for s in 0..256 {
            if cont[s] == 0.0 {
                continue;
            }
            let c = if soma > M as i64 {
                if f[s] <= 1 {
                    continue;
                }
                cont[s] * (f[s] as f64 / (f[s] - 1) as f64).log2()
            } else {
                -cont[s] * ((f[s] + 1) as f64 / f[s] as f64).log2()
            };
            if c < custo {
                custo = c;
                best = s;
            }
        }
        if soma > M as i64 {
            f[best] -= 1;
            soma -= 1;
        } else {
            f[best] += 1;
            soma += 1;
        }
    }
    f
}

/// Codifica `sym` com a tabela dada; devolve só os bytes do estado e da renormalização (sem tabela).
fn codifica(sym: &[u8], f: &[u32; 256], cum: &[u32; 257]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(sym.len() / 2 + 16);
    let mut x = L;
    for &s in sym.iter().rev() {
        let fs = f[s as usize];
        let xmax = ((L >> BITS) << 8) * fs;
        while x >= xmax {
            buf.push(x as u8);
            x >>= 8;
        }
        x = (x / fs) * M + (x % fs) + cum[s as usize];
    }
    buf.extend_from_slice(&[(x >> 24) as u8, (x >> 16) as u8, (x >> 8) as u8, x as u8]);
    buf.reverse();
    buf
}

/// O laço de decodificação do leitor (src/lib.rs, `rans`), para um bloco já sem a tabela.
fn decodifica(src: &[u8], out: &mut [u8], f: &[u32; 256], cum: &[u32; 257], slot: &[u8]) -> bool {
    let mut x = u32::from_le_bytes(src[..4].try_into().unwrap());
    let mut p = 4;
    for v in out.iter_mut() {
        let sl = x & (M - 1);
        let s = slot[sl as usize] as usize;
        *v = s as u8;
        x = f[s] * (x >> BITS) + sl - cum[s];
        while x < L {
            let Some(b) = src.get(p) else { return false };
            x = (x << 8) | *b as u32;
            p += 1;
        }
    }
    x == L && p == src.len()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let st = std::fs::read(&a[1]).expect("safetensors");
    let hl = u64::from_le_bytes(st[..8].try_into().unwrap()) as usize;
    let (es, _) = strata_ref::valida_safetensors(&st[8..8 + hl], Some((st.len() - 8 - hl) as u64)).expect("safetensors válido");
    let dados = &st[8 + hl..];
    // expoente de cada tensor BF16
    let mut exps: Vec<(String, Vec<u8>)> = Vec::new();
    for e in &es {
        if e.dtype == "BF16" {
            let b = &dados[e.begin as usize..e.end as usize];
            exps.push((e.name.clone(), b.chunks_exact(2).map(|w| ((u16::from_le_bytes([w[0], w[1]]) >> 7) & 0xff) as u8).collect()));
        }
    }
    let pesos: usize = exps.iter().map(|x| x.1.len()).sum();
    println!("{} tensores BF16, {} pesos", exps.len(), pesos);

    // o que o codificador em JS gravou, para conferir o bloco único
    if let Some(p) = a.get(2) {
        let f = std::fs::read(p).unwrap();
        let hl = u32::from_le_bytes(f[8..12].try_into().unwrap()) as usize;
        let h: serde_json::Value = serde_json::from_slice(&f[12..12 + hl]).unwrap();
        let tot: u64 = h["streams"].as_array().unwrap().iter().filter(|s| s["p"] == "exp").map(|s| s["l"].as_u64().unwrap()).sum();
        println!("streams de expoente no .strata: {tot} B");
    }

    let tabs: Vec<([u32; 256], [u32; 257], usize)> = exps
        .iter()
        .map(|(_, s)| {
            let mut c = [0f64; 256];
            for &x in s {
                c[x as usize] += 1.0;
            }
            let f = freqs(&c, s.len());
            let mut cum = [0u32; 257];
            for i in 0..256 {
                cum[i + 1] = cum[i] + f[i];
            }
            (f, cum, f.iter().filter(|&&x| x > 0).count())
        })
        .collect();

    let nt = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut base = 0u64;
    for bloco in [usize::MAX, 1 << 20, 1 << 18, 1 << 16, 1 << 14, 1 << 12] {
        let mut total = 0u64;
        let mut nblocos = 0usize;
        for ((_, s), (f, cum, k)) in exps.iter().zip(&tabs) {
            let blocos: Vec<&[u8]> = s.chunks(bloco.min(s.len().max(1))).collect();
            let corpo: usize = std::thread::scope(|sc| {
                let hs: Vec<_> = blocos.chunks(blocos.len().div_ceil(nt)).map(|g| sc.spawn(move || g.iter().map(|b| codifica(b, f, cum).len()).sum::<usize>())).collect();
                hs.into_iter().map(|h| h.join().unwrap()).sum()
            });
            // tabela uma vez por tensor; com mais de um bloco, um offset de 4 bytes por bloco
            let extra = if blocos.len() > 1 { 4 * blocos.len() } else { 0 };
            total += (2 + 3 * k + corpo + extra) as u64;
            nblocos += blocos.len();
        }
        if bloco == usize::MAX {
            base = total;
        }
        let rotulo = if bloco == usize::MAX { "um bloco por tensor (hoje)".to_string() } else { format!("blocos de {} K símbolos", bloco >> 10) };
        println!(
            "{rotulo:>28}: {total:>11} B, {nblocos:>7} blocos, +{:>9} B = +{:.5} bit/peso",
            total - base,
            8.0 * (total - base) as f64 / pesos as f64
        );
    }

    // velocidade: o maior tensor (o embedding), numa thread e em todas, saída já tocada
    let (i, _) = exps.iter().enumerate().max_by_key(|(_, x)| x.1.len()).unwrap();
    let ((nome, s), (f, cum, _)) = (&exps[i], &tabs[i]);
    let mut slot = vec![0u8; M as usize];
    for v in 0..256 {
        slot[cum[v] as usize..cum[v + 1] as usize].fill(v as u8);
    }
    let mut out = vec![1u8; s.len()];
    out.fill(0);
    println!("velocidade de decodificação em {nome} ({} símbolos), {nt} núcleos:", s.len());
    for bloco in [usize::MAX, 1 << 20, 1 << 16, 1 << 12] {
        let b = bloco.min(s.len());
        let enc: Vec<Vec<u8>> = s.chunks(b).map(|c| codifica(c, f, cum)).collect();
        for threads in [1, nt] {
            let mut melhor = f64::INFINITY;
            for _ in 0..3 {
                let t = Instant::now();
                let ok = std::thread::scope(|sc| {
                    let por = enc.len().div_ceil(threads);
                    let hs: Vec<_> = out
                        .chunks_mut(por * b)
                        .zip(enc.chunks(por))
                        .map(|(o, e)| {
                            let slot = &slot;
                            sc.spawn(move || o.chunks_mut(b).zip(e).all(|(oo, ee)| decodifica(ee, oo, f, cum, slot)))
                        })
                        .collect();
                    hs.into_iter().all(|h| h.join().unwrap())
                });
                assert!(ok && out == *s, "decodificação diferente");
                melhor = melhor.min(t.elapsed().as_secs_f64());
            }
            let rotulo = if bloco == usize::MAX { "um bloco".to_string() } else { format!("blocos de {} K", bloco >> 10) };
            println!("  {rotulo:>14}, {threads} thread(s): {:.3} s ({:.0} M símbolos/s)", melhor, s.len() as f64 / melhor / 1e6);
        }
    }
}
