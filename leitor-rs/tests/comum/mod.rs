//! Modelo Llama sintético e minúsculo (safetensors BF16 + GGUF Q4_K/Q6_K) para os vetores da 0.3.
//! Determinístico: os mesmos bytes em qualquer máquina.
#![allow(dead_code)]
use strata_ref::residual::{celulas, para_bf16, permuta, valor, Grade};

struct Rnd(u64);
impl Rnd {
    fn u(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.u() % 1_000_001) as f32 / 1_000_000.0
    }
}

fn bf16s(w: &[u16]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Blocos k-quant aleatórios e pesos BF16 perto da grade (na ordem do GGUF): a maioria dentro de
/// uma sub-célula do nível 3, alguns longe da grade, um NaN e um −0.
fn matriz(g: Grade, linhas: usize, r: &mut Rnd) -> (Vec<u8>, Vec<u16>) {
    let n = linhas * 256;
    let bb = g.bytes_bloco();
    let mut base: Vec<u8> = (0..n / 256 * bb).map(|_| r.u() as u8).collect();
    for sb in 0..n / 256 {
        let b = &mut base[sb * bb..(sb + 1) * bb];
        let d = (0x2800 + (r.u() % 0x800) as u16).to_le_bytes(); // d entre ~0,03 e ~0,06
        match g {
            Grade::Q4K => {
                b[0..2].copy_from_slice(&d);
                b[2..4].copy_from_slice(&(0x2000 + (r.u() % 0x800) as u16).to_le_bytes());
            }
            Grade::Q6K => b[208..210].copy_from_slice(&d),
        }
    }
    let (mut a, mut m, mut q) = ([0f32; 256], [0f32; 256], [0i32; 256]);
    let mut w = Vec::with_capacity(n);
    for sb in 0..n / 256 {
        celulas(g, &base[sb * bb..(sb + 1) * bb], &mut a, &mut m, &mut q);
        for j in 0..256 {
            let i = sb * 256 + j;
            let sub = (r.u() % 8) as u32;
            let mut v = valor(a[j], m[j], q[j], sub, 3) + (r.f() - 0.5) * a[j] / 16.0;
            if i % 61 == 7 {
                v += 3.0 * a[j];
            }
            w.push(match i {
                100 => 0x7fc0,
                200 => 0x8000,
                _ => para_bf16(v),
            });
        }
    }
    (base, w)
}

/// (safetensors, gguf)
pub fn modelo() -> (Vec<u8>, Vec<u8>) {
    let mut r = Rnd(0x5eed_cafe_f00d_1234);
    let (nh, nkv) = (2usize, 1usize);
    let (emb_q, emb_w) = matriz(Grade::Q6K, 8, &mut r);
    let (q_q, q_w) = matriz(Grade::Q4K, 8, &mut r);
    let (k_q, k_w) = matriz(Grade::Q4K, 4, &mut r);
    let norma = |r: &mut Rnd| -> Vec<u16> { (0..256).map(|_| para_bf16(0.5 + r.f())).collect() };
    let (n_attn, n_fim) = (norma(&mut r), norma(&mut r));
    let rope: Vec<u8> = (0..4).flat_map(|i| (1.0f32 / (1 + i) as f32).to_le_bytes()).collect();

    // safetensors: pesos na ordem do HF (desfaz a permutação de q e k)
    let st_t: Vec<(&str, &str, Vec<u64>, Vec<u8>)> = vec![
        ("model.embed_tokens.weight", "BF16", vec![8, 256], bf16s(&emb_w)),
        ("model.layers.0.self_attn.q_proj.weight", "BF16", vec![8, 256], bf16s(&permuta(&q_w, 8, 256, nh, false))),
        ("model.layers.0.self_attn.k_proj.weight", "BF16", vec![4, 256], bf16s(&permuta(&k_w, 4, 256, nkv, false))),
        ("model.layers.0.input_layernorm.weight", "BF16", vec![256], bf16s(&n_attn)),
        ("model.norm.weight", "BF16", vec![256], bf16s(&n_fim)),
        ("model.extra.contagem", "I64", vec![2], [7i64, 42].iter().flat_map(|x| x.to_le_bytes()).collect()),
    ];
    let mut cab = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
    let mut off = 0;
    for (nome, dt, sh, b) in &st_t {
        cab += &format!(",\"{nome}\":{{\"dtype\":\"{dt}\",\"shape\":{sh:?},\"data_offsets\":[{off},{}]}}", off + b.len());
        off += b.len();
    }
    cab += "}";
    while cab.len() % 8 != 0 {
        cab.push(' ');
    }
    let mut st = (cab.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(cab.as_bytes());
    for (_, _, _, b) in &st_t {
        st.extend_from_slice(b);
    }

    // GGUF v3
    let f32s = |w: &[u16]| -> Vec<u8> { w.iter().flat_map(|x| ((*x as u32) << 16).to_le_bytes()).collect() };
    let gg_t: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = vec![
        ("output_norm.weight", vec![256], 0, f32s(&n_fim)),
        ("rope_freqs.weight", vec![4], 0, rope),
        ("token_embd.weight", vec![256, 8], 14, emb_q),
        ("blk.0.attn_k.weight", vec![256, 4], 12, k_q),
        ("blk.0.attn_norm.weight", vec![256], 0, f32s(&n_attn)),
        ("blk.0.attn_q.weight", vec![256, 8], 12, q_q),
    ];
    let mut g = b"GGUF".to_vec();
    let txt = |g: &mut Vec<u8>, s: &str| {
        g.extend_from_slice(&(s.len() as u64).to_le_bytes());
        g.extend_from_slice(s.as_bytes());
    };
    g.extend_from_slice(&3u32.to_le_bytes());
    g.extend_from_slice(&(gg_t.len() as u64).to_le_bytes());
    g.extend_from_slice(&5u64.to_le_bytes());
    txt(&mut g, "general.architecture");
    g.extend_from_slice(&8u32.to_le_bytes());
    txt(&mut g, "llama");
    for (k, v) in [("general.alignment", 32u32), ("llama.attention.head_count", 2), ("llama.attention.head_count_kv", 1)] {
        txt(&mut g, k);
        g.extend_from_slice(&4u32.to_le_bytes());
        g.extend_from_slice(&v.to_le_bytes());
    }
    txt(&mut g, "tokenizer.ggml.tokens");
    g.extend_from_slice(&9u32.to_le_bytes());
    g.extend_from_slice(&8u32.to_le_bytes());
    g.extend_from_slice(&2u64.to_le_bytes());
    txt(&mut g, "a");
    txt(&mut g, "b");
    let mut off = 0u64;
    for (nome, ne, tipo, b) in &gg_t {
        txt(&mut g, nome);
        g.extend_from_slice(&(ne.len() as u32).to_le_bytes());
        for x in ne {
            g.extend_from_slice(&x.to_le_bytes());
        }
        g.extend_from_slice(&tipo.to_le_bytes());
        g.extend_from_slice(&off.to_le_bytes());
        off = (off + b.len() as u64).div_ceil(32) * 32;
    }
    g.resize(g.len().div_ceil(32) * 32, 0);
    let ini = g.len();
    for (_, _, _, b) in &gg_t {
        g.resize(ini + (g.len() - ini).div_ceil(32) * 32, 0);
        g.extend_from_slice(b);
    }
    (st, g)
}
