//! Modo residual (strata/0.3, docs/02 A.14): o estrato 0 de um tensor é um k-quant do llama.cpp
//! (Q4_K ou Q6_K) byte a byte; o nível k divide cada célula da grade em 2^k partes (um plano de 1
//! bit por peso por nível); o último estrato fecha o BF16 exato dado a sub-célula do último nível.
//!
//! Tudo aqui é inteiro ou float32 sem contração (FMA): o escritor e qualquer leitor calculam as
//! mesmas células bit a bit, e é isso que garante a volta sem perdas.

use crate::{falha, Erro, R};

/// Pesos por bloco do fecho: cada bloco tem estado rANS próprio e é decodificado em paralelo.
pub const BLOCO_FECHO: usize = 1 << 16;
const BITS: u32 = 15;
const M: u32 = 1 << BITS;
const L: u32 = 1 << 23;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Grade {
    Q4K,
    Q6K,
}
impl Grade {
    pub fn de_nome(s: &str) -> Option<Grade> {
        match s {
            "Q4_K" => Some(Grade::Q4K),
            "Q6_K" => Some(Grade::Q6K),
            _ => None,
        }
    }
    pub fn nome(self) -> &'static str {
        match self {
            Grade::Q4K => "Q4_K",
            Grade::Q6K => "Q6_K",
        }
    }
    /// Bytes por super-bloco de 256 pesos (ggml-common.h).
    pub fn bytes_bloco(self) -> usize {
        match self {
            Grade::Q4K => 144,
            Grade::Q6K => 210,
        }
    }
    fn qmin_qmax(self) -> (i32, i32) {
        match self {
            Grade::Q4K => (0, 15),
            Grade::Q6K => (-32, 31),
        }
    }
}

pub fn f16(h: u16) -> f32 {
    let s = ((h as u32) >> 15) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let v = match e {
        0 => return if s != 0 { -(m as f32) * (1.0 / 16_777_216.0) } else { m as f32 * (1.0 / 16_777_216.0) },
        31 => 0x7f80_0000 | (m << 13),
        _ => ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(s | v)
}

/// Passo `a`, mínimo `m` e índice `q` de cada um dos 256 pesos de um super-bloco: o valor do
/// estrato 0 é `a·q − m`, exatamente como o ggml desquantiza (dequantize_row_q4_K / q6_K).
pub fn celulas(g: Grade, b: &[u8], a: &mut [f32; 256], m: &mut [f32; 256], q: &mut [i32; 256]) {
    match g {
        Grade::Q4K => {
            let d = f16(u16::from_le_bytes([b[0], b[1]]));
            let dmin = f16(u16::from_le_bytes([b[2], b[3]]));
            let s = &b[4..16];
            for j in 0..8 {
                let (sc, mn) = if j < 4 {
                    (s[j] & 63, s[j + 4] & 63)
                } else {
                    ((s[j + 4] & 0xf) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4))
                };
                let (aj, mj) = (d * sc as f32, dmin * mn as f32);
                a[32 * j..32 * j + 32].fill(aj);
                m[32 * j..32 * j + 32].fill(mj);
            }
            let qs = &b[16..144];
            for c in 0..4 {
                for l in 0..32 {
                    q[64 * c + l] = (qs[32 * c + l] & 0xf) as i32;
                    q[64 * c + 32 + l] = (qs[32 * c + l] >> 4) as i32;
                }
            }
        }
        Grade::Q6K => {
            let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
            let d = f16(u16::from_le_bytes([b[208], b[209]]));
            m.fill(0.0);
            for n in 0..2 {
                for l in 0..32 {
                    let is = l / 16;
                    let (lo, hi, h) = (ql[64 * n + l], ql[64 * n + l + 32], qh[32 * n + l]);
                    let base = 128 * n;
                    let v = [(lo & 0xf) | (h & 3) << 4, (hi & 0xf) | ((h >> 2) & 3) << 4, (lo >> 4) | ((h >> 4) & 3) << 4, (hi >> 4) | ((h >> 6) & 3) << 4];
                    for (k, &qq) in v.iter().enumerate() {
                        let i = base + l + 32 * k;
                        q[i] = qq as i32 - 32;
                        a[i] = d * (sc[8 * n + is + 2 * k] as i8) as f32;
                    }
                }
            }
        }
    }
}

/// Valor no centro da sub-célula `r` (0..2^k, contada no sentido de `q` crescente) do nível `k`:
/// `v = a·(t/2^(k+1)) − m`, `t = 2^(k+1)·q + 2r + 1 − 2^k` (B.8). Com k = 0, é o estrato 0.
#[inline]
pub fn valor(a: f32, m: f32, q: i32, r: u32, k: u32) -> f32 {
    let t = (q << (k + 1)) + 2 * r as i32 + 1 - (1 << k);
    borda(a, m, t, k)
}
#[inline]
fn borda(a: f32, m: f32, t: i32, k: u32) -> f32 {
    a * (t as f32 * (1.0 / (1u32 << (k + 1)) as f32)) - m
}

/// Ordinal de um BF16: crescente com o valor, −0 logo antes de +0, NaN fora de [0x7F, 0xFF80].
#[inline]
pub fn ordinal(u: u16) -> u16 {
    if u & 0x8000 != 0 {
        0x7fff - (u & 0x7fff)
    } else {
        u + 0x8000
    }
}
#[inline]
pub fn de_ordinal(o: u16) -> u16 {
    if o >= 0x8000 {
        o - 0x8000
    } else {
        0x8000 | (0x7fff - o)
    }
}
/// Menor ordinal cujo valor é ≥ x (x não NaN).
#[inline]
fn prim(x: f32) -> u16 {
    if x == f32::NEG_INFINITY {
        0x7f
    } else if x == f32::INFINITY {
        0xff80
    } else if x == 0.0 {
        0x7fff
    } else {
        let b = x.to_bits();
        if x > 0.0 {
            0x8000 + (b >> 16) as u16 + (b & 0xffff != 0) as u16
        } else {
            0x7fff - ((b & 0x7fff_ffff) >> 16) as u16
        }
    }
}
const LINHA: (u16, u16) = (0x7f, 0xff80);
/// Ordinais [lo, hi] dos BF16 dentro da sub-célula `r` do nível `k` (vazia se lo > hi): de
/// `borda(t−1)` inclusive a `borda(t+1)` exclusive. As sub-células das pontas da grade são
/// abertas para fora; uma grade sem passo (a = 0, ou não finita) vira a reta inteira.
pub fn faixa(g: Grade, a: f32, m: f32, q: i32, r: u32, k: u32) -> (u16, u16) {
    if a == 0.0 || !a.is_finite() || !m.is_finite() {
        return LINHA;
    }
    let t = (q << (k + 1)) + 2 * r as i32 + 1 - (1 << k);
    let (e1, e2) = (borda(a, m, t - 1, k), borda(a, m, t + 1, k));
    if e1.is_nan() || e2.is_nan() {
        return LINHA;
    }
    let (qmin, qmax) = g.qmin_qmax();
    let (aberta_baixo, aberta_cima) = (r == 0 && q <= qmin, r == (1 << k) - 1 && q >= qmax);
    let (mut lo, mut hi, ab_lo, ab_hi) = if a > 0.0 { (e1, e2, aberta_baixo, aberta_cima) } else { (e2, e1, aberta_cima, aberta_baixo) };
    if ab_lo {
        lo = f32::NEG_INFINITY;
    }
    if ab_hi {
        hi = f32::INFINITY;
    }
    let o_lo = prim(lo);
    let o_hi = if hi == f32::INFINITY { 0xff80 } else { prim(hi).wrapping_sub(1) };
    (o_lo, o_hi)
}

/// Arredonda para BF16 (ao par mais próximo), como o torch.
#[inline]
pub fn para_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}
#[inline]
fn eh_nan(u: u16) -> bool {
    u & 0x7f80 == 0x7f80 && u & 0x7f != 0
}

/// Sub-célula do nível `k` que contém o peso `u` (descendo nível a nível) e se ela o contém.
/// Pesos fora da célula (os que o k-quant saturou, ou NaN) ficam na sub-célula mais próxima.
pub fn escolhe(g: Grade, a: f32, m: f32, q: i32, u: u16, k: u32) -> (u32, bool) {
    if eh_nan(u) {
        return (0, false);
    }
    let o = ordinal(u);
    let mut r = 0u32;
    for nk in 1..=k {
        // a metade 2r é a de q menor: abaixo dela no valor se a > 0, acima se a < 0
        let (lo, hi) = faixa(g, a, m, q, 2 * r, nk);
        let primeira = if a < 0.0 { o >= lo } else { o <= hi };
        r = if primeira { 2 * r } else { 2 * r + 1 };
    }
    let (lo, hi) = faixa(g, a, m, q, r, k);
    (r, lo <= o && o <= hi)
}

/* ---------------- fecho: o BF16 exato dado a sub-célula ---------------- */

/// Contexto de um peso no fecho: o campo de expoente do passo `a` (a escala local da grade).
#[inline]
fn contexto(a: f32) -> usize {
    ((a.to_bits() >> 23) & 0xff) as usize
}

/// Tabelas do fecho de um tensor: por contexto, as frequências (soma 2^15) das oitavas BF16
/// (sinal + expoente = ordinal >> 7, 512 valores) dos pesos.
pub struct Tabelas {
    ctx: Vec<Option<Tab>>,
}
/// Um contexto: oitavas presentes em ordem crescente, suas frequências e as somas acumuladas.
struct Tab {
    os: Vec<u16>,
    fs: Vec<u32>,
    p: Vec<u64>,
}
impl Tab {
    fn nova(os: Vec<u16>, fs: Vec<u32>) -> Tab {
        let mut p = Vec::with_capacity(fs.len() + 1);
        p.push(0);
        for &f in &fs {
            p.push(p.last().unwrap() + f as u64);
        }
        Tab { os, fs, p }
    }
}

/// Quantiza contagens para frequências que somam 2^15 (a mesma regra do rANS do expoente, A.11).
pub fn frequencias(cont: &[u64]) -> Vec<u32> {
    let n: u64 = cont.iter().sum();
    let mut f = vec![0u32; cont.len()];
    if n == 0 {
        return f;
    }
    let mut soma: i64 = 0;
    for s in 0..cont.len() {
        if cont[s] > 0 {
            f[s] = ((cont[s] as f64 * M as f64 / n as f64 + 0.5).floor() as u32).max(1);
            soma += f[s] as i64;
        }
    }
    while soma != M as i64 {
        let (mut best, mut custo) = (usize::MAX, f64::INFINITY);
        for s in 0..cont.len() {
            if cont[s] == 0 {
                continue;
            }
            let c = if soma > M as i64 {
                if f[s] <= 1 {
                    continue;
                }
                cont[s] as f64 * (f[s] as f64 / (f[s] - 1) as f64).log2()
            } else {
                -(cont[s] as f64) * ((f[s] + 1) as f64 / f[s] as f64).log2()
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

impl Tabelas {
    /// Formato: u16 número de contextos; por contexto (crescente): u8 contexto, u16 k, k × (u16
    /// oitava, u16 freq), oitavas crescentes, freqs > 0 somando 2^15.
    pub fn le(src: &[u8]) -> R<(Tabelas, usize)> {
        let ruim = || Err(Erro("fecho: tabela inválida".into()));
        let mut ctx: Vec<Option<Tab>> = (0..256).map(|_| None).collect();
        let Some(n) = src.first_chunk::<2>() else { return ruim() };
        let n = u16::from_le_bytes(*n) as usize;
        let mut p = 2;
        let mut ant = -1i32;
        for _ in 0..n {
            let Some(c) = src.get(p) else { return ruim() };
            let c = *c as i32;
            let Some(kb) = src.get(p + 1..p + 3) else { return ruim() };
            let k = u16::from_le_bytes([kb[0], kb[1]]) as usize;
            p += 3;
            if c <= ant || k == 0 || k > 512 {
                return ruim();
            }
            ant = c;
            let Some(es) = src.get(p..p + 4 * k) else { return ruim() };
            let (mut os, mut fs, mut tot, mut oa) = (Vec::with_capacity(k), Vec::with_capacity(k), 0u32, -1i32);
            for e in es.chunks_exact(4) {
                let (o, f) = (u16::from_le_bytes([e[0], e[1]]), u16::from_le_bytes([e[2], e[3]]) as u32);
                if (o as i32) <= oa || o > 511 || f == 0 || tot + f > M {
                    return ruim();
                }
                oa = o as i32;
                tot += f;
                os.push(o);
                fs.push(f);
            }
            if tot != M {
                return ruim();
            }
            ctx[c as usize] = Some(Tab::nova(os, fs));
            p += 4 * k;
        }
        Ok((Tabelas { ctx }, p))
    }
    fn escreve(cont: &[Vec<u64>], out: &mut Vec<u8>) -> Tabelas {
        let mut ctx: Vec<Option<Tab>> = (0..256).map(|_| None).collect();
        let usados: Vec<usize> = (0..256).filter(|&c| cont[c].iter().any(|&x| x > 0)).collect();
        out.extend_from_slice(&(usados.len() as u16).to_le_bytes());
        for &c in &usados {
            let f = frequencias(&cont[c]);
            let (os, fs): (Vec<u16>, Vec<u32>) = (0..512).filter(|&o| f[o] > 0).map(|o| (o as u16, f[o])).unzip();
            out.push(c as u8);
            out.extend_from_slice(&(os.len() as u16).to_le_bytes());
            for (o, f) in os.iter().zip(&fs) {
                out.extend_from_slice(&o.to_le_bytes());
                out.extend_from_slice(&(*f as u16).to_le_bytes());
            }
            ctx[c] = Some(Tab::nova(os, fs));
        }
        Tabelas { ctx }
    }
}

/// Modelo de um peso: as `c` oitavas da tabela que cortam a faixa [lo, hi], cada uma com massa
/// (freq da oitava × quantos BF16 dela caem na faixa), e o escape (peso fora da faixa: 16 bits
/// crus). Com massa acumulada CM_i, o símbolo i ocupa [cum_i, cum_{i+1}) com
/// cum_i = ⌊(R − c)·CM_i / total⌋ + i e R = 2^15 − 1: toda oitava tem freq ≥ 1, a soma é R e o escape
/// fica com o último slot. Sem oitava na faixa, o escape leva os 2^15. Custo O(log c) por peso.
struct Modelo<'t> {
    tab: Option<&'t Tab>,
    i0: usize,
    c: usize,
    lo: u16,
    hi: u16,
    /// massa a descontar da primeira e da última oitava (as pontas parciais) e o total
    menos_pri: u64,
    menos_ult: u64,
    total: u64,
}
const R_FECHO: u64 = (M - 1) as u64;
impl<'t> Modelo<'t> {
    fn novo(t: &'t Tabelas, ctx: usize, lo: u16, hi: u16) -> Modelo<'t> {
        let mut md = Modelo { tab: None, i0: 0, c: 0, lo, hi, menos_pri: 0, menos_ult: 0, total: 0 };
        let Some(tab) = &t.ctx[ctx] else { return md };
        if lo > hi {
            return md;
        }
        let i0 = tab.os.partition_point(|&o| o < lo >> 7);
        let i1 = tab.os.partition_point(|&o| o <= hi >> 7);
        if i1 <= i0 {
            return md;
        }
        let cnt = |o: u16| (hi.min((o << 7) | 127) - lo.max(o << 7)) as u64 + 1;
        md.tab = Some(tab);
        md.i0 = i0;
        md.c = i1 - i0;
        md.menos_pri = tab.fs[i0] as u64 * (128 - cnt(tab.os[i0]));
        // com uma oitava só, cnt já descontou as duas pontas
        md.menos_ult = if md.c > 1 { tab.fs[i1 - 1] as u64 * (128 - cnt(tab.os[i1 - 1])) } else { 0 };
        md.total = 128 * (tab.p[i1] - tab.p[i0]) - md.menos_pri - md.menos_ult;
        md
    }
    #[inline]
    fn esc(&self) -> u32 {
        if self.c == 0 {
            M
        } else {
            1
        }
    }
    /// cum_i, 0 ≤ i ≤ c
    #[inline]
    fn cum(&self, i: usize) -> u32 {
        let tab = self.tab.unwrap();
        let mut cm = 128 * (tab.p[self.i0 + i] - tab.p[self.i0]);
        if i >= 1 {
            cm -= self.menos_pri;
        }
        if i == self.c {
            cm -= self.menos_ult;
        }
        ((R_FECHO - self.c as u64) * cm / self.total) as u32 + i as u32
    }
    /// (cum, freq, primeiro ordinal na faixa, quantos) da oitava `o`, se ela é candidata
    fn simbolo(&self, o: u16) -> Option<(u32, u32, u16, u32)> {
        let tab = self.tab?;
        let j = tab.os[self.i0..self.i0 + self.c].binary_search(&o).ok()?;
        Some(self.do_indice(j))
    }
    fn do_indice(&self, j: usize) -> (u32, u32, u16, u32) {
        let o = self.tab.unwrap().os[self.i0 + j];
        let (ini, fim) = (self.lo.max(o << 7), self.hi.min((o << 7) | 127));
        let (c0, c1) = (self.cum(j), self.cum(j + 1));
        (c0, c1 - c0, ini, (fim - ini) as u32 + 1)
    }
    /// símbolo que contém o slot `s` < R (busca binária em cum)
    fn acha(&self, s: u32) -> (u32, u32, u16, u32) {
        let (mut a, mut b) = (0, self.c);
        while b - a > 1 {
            let m = (a + b) / 2;
            if self.cum(m) <= s {
                a = m;
            } else {
                b = m;
            }
        }
        self.do_indice(a)
    }
}

/// Codifica um bloco: símbolos (cum, freq) em ordem; devolve estado (4 bytes LE) + corpo.
fn rans_codifica(sim: &[(u32, u32)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(sim.len() / 2 + 8);
    let mut x = L;
    for &(cum, f) in sim.iter().rev() {
        let xmax = ((L >> BITS) << 8) * f;
        while x >= xmax {
            buf.push(x as u8);
            x >>= 8;
        }
        x = (x / f) * M + (x % f) + cum;
    }
    buf.extend_from_slice(&[(x >> 24) as u8, (x >> 16) as u8, (x >> 8) as u8, x as u8]);
    buf.reverse();
    buf
}

struct Dec<'a> {
    src: &'a [u8],
    p: usize,
    x: u32,
}
impl<'a> Dec<'a> {
    fn novo(src: &'a [u8]) -> R<Dec<'a>> {
        let Some(e) = src.first_chunk::<4>() else { return falha("fecho: bloco curto demais") };
        let x = u32::from_le_bytes(*e);
        if !(L..0x8000_0000).contains(&x) {
            return falha("fecho: estado inicial inválido");
        }
        Ok(Dec { src, p: 4, x })
    }
    #[inline]
    fn slot(&self) -> u32 {
        self.x & (M - 1)
    }
    #[inline]
    fn avanca(&mut self, cum: u32, f: u32) -> R<()> {
        self.x = f * (self.x >> BITS) + self.slot() - cum;
        while self.x < L {
            let Some(b) = self.src.get(self.p) else { return falha("fecho: bloco truncado") };
            self.x = (self.x << 8) | *b as u32;
            self.p += 1;
        }
        Ok(())
    }
    fn fim(&self) -> R<()> {
        if self.x != L || self.p != self.src.len() {
            return falha("fecho: bloco corrompido");
        }
        Ok(())
    }
}

/// Posição uniforme `i` entre `n` (n ≤ 128): cum = ⌊i·2^15/n⌋.
#[inline]
fn uniforme(i: u32, n: u32) -> (u32, u32) {
    let c = i * M / n;
    (c, (i + 1) * M / n - c)
}

/// Grade de um tensor já cortada em super-blocos: passo, mínimo e índice de cada peso.
pub struct Grades<'a> {
    pub g: Grade,
    pub base: &'a [u8],
}
impl Grades<'_> {
    pub fn n(&self) -> usize {
        self.base.len() / self.g.bytes_bloco() * 256
    }
    /// Passo, mínimo e índice dos pesos [i0, i0 + saida.len()), com i0 múltiplo de 256.
    fn para(&self, i0: usize, a: &mut [f32], m: &mut [f32], q: &mut [i32]) {
        let bb = self.g.bytes_bloco();
        let (mut aa, mut mm, mut qq) = ([0f32; 256], [0f32; 256], [0i32; 256]);
        for (s, j) in (i0 / 256..).zip((0..a.len()).step_by(256)) {
            celulas(self.g, &self.base[s * bb..(s + 1) * bb], &mut aa, &mut mm, &mut qq);
            a[j..j + 256].copy_from_slice(&aa);
            m[j..j + 256].copy_from_slice(&mm);
            q[j..j + 256].copy_from_slice(&qq);
        }
    }
}

/// Índice `r` (nível k) de cada peso, dos planos: o plano j (1 = mais significativo) tem 1 bit por
/// peso, o peso i no bit i & 7 do byte i >> 3.
#[inline]
fn r_dos_planos(planos: &[&[u8]], i: usize) -> u32 {
    let mut r = 0;
    for p in planos {
        r = (r << 1) | ((p[i >> 3] >> (i & 7)) & 1) as u32;
    }
    r
}

/// Resultado da codificação de um tensor residual.
pub struct Codificado {
    /// planos dos níveis 1..k, n/8 bytes cada
    pub planos: Vec<Vec<u8>>,
    pub fecho: Vec<u8>,
    /// pesos fora da sub-célula do último nível (vão crus no fecho)
    pub escapes: u64,
}

/// Codifica um tensor: `u` são os pesos BF16 na ordem do GGUF, `k` o número de níveis.
pub fn codifica(gr: &Grades, u: &[u16], k: u32, nt: usize) -> R<Codificado> {
    let n = gr.n();
    if u.len() != n || !n.is_multiple_of(256) {
        return falha("residual: número de pesos não bate com a grade");
    }
    // passo 1: sub-célula de cada peso, escapes e contagens por contexto (em paralelo por fatia)
    let fatias = n.div_ceil(BLOCO_FECHO);
    struct Parcial {
        r: Vec<u8>,
        esc: Vec<bool>,
        /// contexto << 9 | oitava, um por peso dentro da faixa (4 bytes: cabe o embedding de 8B na RAM)
        cont: Vec<u32>,
    }
    let parciais = crate::em_paralelo(nt, fatias, |b| {
        let (i0, i1) = (b * BLOCO_FECHO, ((b + 1) * BLOCO_FECHO).min(n));
        let len = i1 - i0;
        let (mut a, mut m, mut q) = (vec![0f32; len], vec![0f32; len], vec![0i32; len]);
        gr.para(i0, &mut a, &mut m, &mut q);
        let (mut r, mut esc, mut cont) = (vec![0u8; len], vec![false; len], Vec::with_capacity(len));
        for j in 0..len {
            let (rr, dentro) = escolhe(gr.g, a[j], m[j], q[j], u[i0 + j], k);
            r[j] = rr as u8;
            esc[j] = !dentro;
            if dentro {
                cont.push(((contexto(a[j]) as u32) << 9) | (ordinal(u[i0 + j]) >> 7) as u32);
            }
        }
        Ok(Parcial { r, esc, cont })
    })?;
    let mut cont = vec![vec![0u64; 512]; 256];
    let mut escapes = 0u64;
    for p in &parciais {
        for &co in &p.cont {
            cont[(co >> 9) as usize][(co & 511) as usize] += 1;
        }
        escapes += p.esc.iter().filter(|&&e| e).count() as u64;
    }
    let mut fecho = Vec::new();
    let tab = Tabelas::escreve(&cont, &mut fecho);
    // planos
    let mut planos = vec![vec![0u8; n / 8]; k as usize];
    for (b, p) in parciais.iter().enumerate() {
        let i0 = b * BLOCO_FECHO;
        for (j, &rr) in p.r.iter().enumerate() {
            let i = i0 + j;
            for (lvl, pl) in planos.iter_mut().enumerate() {
                pl[i >> 3] |= ((rr >> (k - 1 - lvl as u32)) & 1) << (i & 7);
            }
        }
    }
    // passo 2: blocos do fecho
    let blocos = crate::em_paralelo(nt, fatias, |b| {
        let (i0, i1) = (b * BLOCO_FECHO, ((b + 1) * BLOCO_FECHO).min(n));
        let len = i1 - i0;
        let (mut a, mut m, mut q) = (vec![0f32; len], vec![0f32; len], vec![0i32; len]);
        gr.para(i0, &mut a, &mut m, &mut q);
        let p = &parciais[b];
        let mut sim = Vec::with_capacity(len * 2);
        for j in 0..len {
            let w = u[i0 + j];
            let (lo, hi) = faixa(gr.g, a[j], m[j], q[j], p.r[j] as u32, k);
            let md = Modelo::novo(&tab, contexto(a[j]), lo, hi);
            let o = ordinal(w);
            let cand = if p.esc[j] { None } else { md.simbolo(o >> 7) };
            match cand {
                Some((cum, f, ini, cnt)) => {
                    sim.push((cum, f));
                    if cnt > 1 {
                        sim.push(uniforme((o - ini) as u32, cnt));
                    }
                }
                None => {
                    if !p.esc[j] {
                        return falha("residual: peso dentro da faixa sem oitava na tabela");
                    }
                    sim.push((M - md.esc(), md.esc()));
                    sim.push(((w >> 8) as u32 * 128, 128));
                    sim.push(((w & 0xff) as u32 * 128, 128));
                }
            }
        }
        Ok(rans_codifica(&sim))
    })?;
    for b in &blocos {
        fecho.extend_from_slice(&(b.len() as u32).to_le_bytes());
    }
    for b in &blocos {
        fecho.extend_from_slice(b);
    }
    Ok(Codificado { planos, fecho, escapes })
}

/// Valores de um tensor residual lidos até o nível `k` (k ≤ número de planos dados), arredondados
/// para BF16, na ordem do GGUF.
pub fn aproxima(gr: &Grades, planos: &[&[u8]], k: u32, nt: usize) -> R<Vec<u16>> {
    let n = gr.n();
    let fatias = n.div_ceil(BLOCO_FECHO);
    let partes = crate::em_paralelo(nt, fatias, |b| {
        let (i0, i1) = (b * BLOCO_FECHO, ((b + 1) * BLOCO_FECHO).min(n));
        let len = i1 - i0;
        let (mut a, mut m, mut q) = (vec![0f32; len], vec![0f32; len], vec![0i32; len]);
        gr.para(i0, &mut a, &mut m, &mut q);
        Ok((0..len).map(|j| para_bf16(valor(a[j], m[j], q[j], r_dos_planos(&planos[..k as usize], i0 + j), k))).collect::<Vec<u16>>())
    })?;
    Ok(partes.concat())
}

/// Decodifica o fecho: os pesos BF16 exatos, na ordem do GGUF. `planos` são os k planos.
pub fn decodifica(gr: &Grades, planos: &[&[u8]], fecho: &[u8], nt: usize) -> R<Vec<u16>> {
    let n = gr.n();
    let k = planos.len() as u32;
    let (tab, mut p) = Tabelas::le(fecho)?;
    let fatias = n.div_ceil(BLOCO_FECHO);
    let Some(tam) = fecho.get(p..p + 4 * fatias) else { return falha("fecho: tabela de blocos truncada") };
    p += 4 * fatias;
    let mut pos = Vec::with_capacity(fatias + 1);
    let mut o = p as u64;
    for c in tam.chunks_exact(4) {
        pos.push(o);
        o += u32::from_le_bytes(c.try_into().unwrap()) as u64;
    }
    if o != fecho.len() as u64 {
        return falha("fecho: tamanhos dos blocos não batem com o stream");
    }
    pos.push(o);
    let partes = crate::em_paralelo(nt, fatias, |b| {
        let (i0, i1) = (b * BLOCO_FECHO, ((b + 1) * BLOCO_FECHO).min(n));
        let len = i1 - i0;
        let (mut a, mut m, mut q) = (vec![0f32; len], vec![0f32; len], vec![0i32; len]);
        gr.para(i0, &mut a, &mut m, &mut q);
        let mut d = Dec::novo(&fecho[pos[b] as usize..pos[b + 1] as usize])?;
        let mut out = vec![0u16; len];
        for j in 0..len {
            let r = r_dos_planos(planos, i0 + j);
            let (lo, hi) = faixa(gr.g, a[j], m[j], q[j], r, k);
            let md = Modelo::novo(&tab, contexto(a[j]), lo, hi);
            let s = d.slot();
            let esc = md.esc();
            if s >= M - esc {
                d.avanca(M - esc, esc)?;
                let hi8 = d.slot() >> 7;
                d.avanca(hi8 * 128, 128)?;
                let lo8 = d.slot() >> 7;
                d.avanca(lo8 * 128, 128)?;
                out[j] = ((hi8 << 8) | lo8) as u16;
                continue;
            }
            let (cum, f, ini, cnt) = md.acha(s);
            d.avanca(cum, f)?;
            let i = if cnt > 1 {
                let s = d.slot();
                let i = ((s + 1) * cnt - 1) / M;
                let (c, f) = uniforme(i, cnt);
                d.avanca(c, f)?;
                i
            } else {
                0
            };
            out[j] = de_ordinal(ini + i as u16);
        }
        d.fim()?;
        Ok(out)
    })?;
    Ok(partes.concat())
}

/// Permutação de linhas que o convert_hf_to_gguf aplica em q_proj e k_proj (RoPE):
/// `w.reshape(h, 2, linhas/h/2, cols).swapaxes(1, 2)`. `para_gguf` = sentido do conversor.
pub fn permuta(w: &[u16], linhas: usize, cols: usize, h: usize, para_gguf: bool) -> Vec<u16> {
    let (por, meio) = (linhas / h, linhas / h / 2);
    let mut out = vec![0u16; w.len()];
    for cab in 0..h {
        for dois in 0..2 {
            for i in 0..meio {
                let hf = cab * por + dois * meio + i;
                let gg = cab * por + 2 * i + dois;
                let (de, para) = if para_gguf { (hf, gg) } else { (gg, hf) };
                out[para * cols..(para + 1) * cols].copy_from_slice(&w[de * cols..(de + 1) * cols]);
            }
        }
    }
    out
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn ordinais_e_prim() {
        for u in 0..=0xffffu16 {
            if !eh_nan(u) {
                assert_eq!(de_ordinal(ordinal(u)), u);
            }
        }
        // prim(x) = menor ordinal com valor >= x, conferido por força bruta em amostras
        let vals: Vec<(u16, f32)> = (0x7fu16..=0xff80).map(|o| (o, f32::from_bits((de_ordinal(o) as u32) << 16))).collect();
        for x in [0.0f32, -0.0, 1.0, -1.0, 1.0000001, -1.0000001, 3.4e38, -3.4e38, 1e-40, -1e-40, 2.5e-3, -7.77e-5] {
            let esperado = vals.iter().find(|(_, v)| *v >= x).unwrap().0;
            assert_eq!(prim(x), esperado, "{x}");
        }
    }

    #[test]
    fn ida_e_volta_sintetica() {
        // 4 super-blocos Q4_K com escalas variadas e pesos perto da grade, alguns fora e um NaN
        let mut base = vec![0u8; 4 * 144];
        let mut s = 0x1234_5678u32;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for b in base.iter_mut() {
            *b = rnd() as u8;
        }
        for sb in 0..4 {
            base[sb * 144..sb * 144 + 2].copy_from_slice(&0x2c00u16.to_le_bytes()); // d = 0.0625
            base[sb * 144 + 2..sb * 144 + 4].copy_from_slice(&0x2400u16.to_le_bytes());
        }
        let gr = Grades { g: Grade::Q4K, base: &base };
        let (mut a, mut m, mut q) = (vec![0f32; 1024], vec![0f32; 1024], vec![0i32; 1024]);
        gr.para(0, &mut a, &mut m, &mut q);
        let u: Vec<u16> = (0..1024)
            .map(|i| {
                let y = a[i] * q[i] as f32 - m[i];
                let ruido = ((rnd() % 2001) as f32 - 1000.0) / 1000.0 * a[i] * if i % 97 == 0 { 3.0 } else { 0.5 };
                if i == 500 { 0x7fc1 } else { para_bf16(y + ruido) }
            })
            .collect();
        for k in 0..=3 {
            let c = codifica(&gr, &u, k, 2).unwrap();
            let pl: Vec<&[u8]> = c.planos.iter().map(Vec::as_slice).collect();
            assert_eq!(decodifica(&gr, &pl, &c.fecho, 3).unwrap(), u, "k = {k}");
            assert!(c.escapes >= 1);
            // corromper qualquer byte do corpo do fecho é recusado ou muda a saída
            let mut f = c.fecho.clone();
            let i = f.len() - 3;
            f[i] ^= 0x10;
            assert!(decodifica(&gr, &pl, &f, 1).map_or(true, |v| v != u));
        }
    }
}

/* ---------------- GGUF (só para o escritor) ---------------- */

pub struct TensorGguf {
    pub nome: String,
    pub ne: Vec<u64>,
    pub tipo: u32,
    pub off: u64,
}
pub struct Gguf {
    pub tensores: Vec<TensorGguf>,
    /// início dos dados dos tensores (fim do cabeçalho, já alinhado)
    pub dados: u64,
    pub alinhamento: u64,
    /// chaves com valor inteiro ou texto
    pub kv: std::collections::HashMap<String, String>,
}
pub const GGML_F32: u32 = 0;
pub const GGML_Q4_K: u32 = 12;
pub const GGML_Q6_K: u32 = 14;

pub fn le_gguf(b: &[u8]) -> R<Gguf> {
    struct C<'a>(&'a [u8], usize);
    impl C<'_> {
        fn tira(&mut self, n: usize) -> R<&[u8]> {
            let Some(s) = self.0.get(self.1..self.1 + n) else { return falha("GGUF truncado") };
            self.1 += n;
            Ok(s)
        }
        fn u32(&mut self) -> R<u32> {
            Ok(u32::from_le_bytes(self.tira(4)?.try_into().unwrap()))
        }
        fn u64(&mut self) -> R<u64> {
            Ok(u64::from_le_bytes(self.tira(8)?.try_into().unwrap()))
        }
        fn texto(&mut self) -> R<String> {
            let n = self.u64()? as usize;
            Ok(String::from_utf8_lossy(self.tira(n)?).into_owned())
        }
        /// lê um valor do tipo `t`; devolve-o como texto se for escalar
        fn valor(&mut self, t: u32) -> R<Option<String>> {
            Ok(match t {
                0 | 1 | 7 => Some(self.tira(1)?[0].to_string()),
                2 | 3 => Some(u16::from_le_bytes(self.tira(2)?.try_into().unwrap()).to_string()),
                4 | 5 => Some(self.u32()?.to_string()),
                6 => Some(f32::from_le_bytes(self.tira(4)?.try_into().unwrap()).to_string()),
                10 | 11 => Some(self.u64()?.to_string()),
                12 => Some(f64::from_le_bytes(self.tira(8)?.try_into().unwrap()).to_string()),
                8 => Some(self.texto()?),
                9 => {
                    let (ti, n) = (self.u32()?, self.u64()?);
                    for _ in 0..n {
                        self.valor(ti)?;
                    }
                    None
                }
                _ => return falha("GGUF: tipo de valor desconhecido"),
            })
        }
    }
    let mut c = C(b, 0);
    if c.tira(4)? != b"GGUF" {
        return falha("não é um GGUF");
    }
    if c.u32()? != 3 {
        return falha("GGUF: só a versão 3");
    }
    let (nt, nkv) = (c.u64()?, c.u64()?);
    let mut kv = std::collections::HashMap::new();
    for _ in 0..nkv {
        let k = c.texto()?;
        let t = c.u32()?;
        if let Some(v) = c.valor(t)? {
            kv.insert(k, v);
        }
    }
    let mut tensores = Vec::new();
    for _ in 0..nt {
        let nome = c.texto()?;
        let nd = c.u32()?;
        let ne = (0..nd).map(|_| c.u64()).collect::<R<Vec<_>>>()?;
        let (tipo, off) = (c.u32()?, c.u64()?);
        tensores.push(TensorGguf { nome, ne, tipo, off });
    }
    let alinhamento = kv.get("general.alignment").and_then(|v| v.parse().ok()).unwrap_or(32u64);
    let dados = (c.1 as u64).div_ceil(alinhamento) * alinhamento;
    Ok(Gguf { tensores, dados, alinhamento, kv })
}

/// Nome do tensor no safetensors (Llama do Hugging Face) para um nome do GGUF.
pub fn nome_hf(g: &str) -> Option<String> {
    let fixos = [("token_embd.weight", "model.embed_tokens.weight"), ("output_norm.weight", "model.norm.weight"), ("output.weight", "lm_head.weight")];
    if let Some((_, h)) = fixos.iter().find(|(gg, _)| *gg == g) {
        return Some(h.to_string());
    }
    let p: Vec<&str> = g.split('.').collect();
    if p.len() != 4 || p[0] != "blk" || p[3] != "weight" {
        return None;
    }
    let parte = match p[2] {
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",
        "attn_norm" => "input_layernorm",
        "ffn_norm" => "post_attention_layernorm",
        _ => return None,
    };
    Some(format!("model.layers.{}.{parte}.weight", p[1]))
}

/* ---------------- escritor ---------------- */

/// Uma linha do relatório do escritor, por grade: pesos e bits por parte.
#[derive(Default, Clone, Copy)]
pub struct Soma {
    pub pesos: u64,
    pub base: u64,
    pub niveis: u64,
    pub fecho: u64,
    pub escapes: u64,
}

pub struct Escrito {
    pub arquivo: Vec<u8>,
    /// por grade ("Q4_K", "Q6_K") e "camadas" (só as matrizes das camadas, como na seção 15)
    pub somas: std::collections::BTreeMap<String, Soma>,
    /// bytes do arquivo até cada estrato (P_d)
    pub prefixos: Vec<u64>,
}

fn deflate_se_menor(b: &[u8]) -> (Vec<u8>, &'static str) {
    let z = miniz_oxide::deflate::compress_to_vec(b, 9);
    if z.len() < b.len() {
        (z, "deflate")
    } else {
        (b.to_vec(), "raw")
    }
}

/// Escreve um .strata 0.3 em modo residual a partir do safetensors original (BF16) e de um GGUF
/// quantizado do mesmo modelo (Llama): cada matriz Q4_K/Q6_K do GGUF vira estrato 0 + `k` níveis +
/// fecho; as normas ficam cruas; o resto do GGUF (cabeçalho, rope_freqs) vai como tensores extras,
/// para que o estrato 0 remonte o GGUF byte a byte.
pub fn escreve(st: &[u8], gb: &[u8], k: u32, nt: usize) -> R<Escrito> {
    use serde_json::{json, Value};
    use std::collections::{BTreeMap, HashMap};
    if k > 8 {
        return falha("no máximo 8 níveis");
    }
    let hl = u64::from_le_bytes(*st.first_chunk::<8>().ok_or_else(|| Erro("safetensors curto".into()))?) as usize;
    let hb = st.get(8..8 + hl).ok_or_else(|| Erro("safetensors truncado".into()))?;
    let (es, total) = crate::valida_safetensors(hb, Some((st.len() - 8 - hl) as u64))?;
    if total != (st.len() - 8 - hl) as u64 {
        return falha("safetensors com bytes sobrando");
    }
    let dados = &st[8 + hl..];
    let g = le_gguf(gb)?;
    if g.kv.get("general.architecture").map(String::as_str) != Some("llama") {
        return falha("o escritor só conhece a arquitetura llama");
    }
    let cab = |c: &str| -> R<usize> { g.kv.get(c).and_then(|v| v.parse().ok()).ok_or_else(|| Erro(format!("GGUF sem {c}"))) };
    let (nh, nkv) = (cab("llama.attention.head_count")?, cab("llama.attention.head_count_kv")?);
    let por_nome: HashMap<&str, usize> = es.iter().enumerate().map(|(i, e)| (e.name.as_str(), i)).collect();

    // o que cada tensor do GGUF é: grade de um tensor do safetensors, F32 de uma norma, ou extra
    enum Como {
        Q(usize, Grade, Option<usize>),
        F32(usize),
        Bruto(Vec<u8>),
    }
    let mut partes = Vec::new();
    let mut usados = vec![false; es.len()];
    let fim_dados = |t: &TensorGguf, n: u64| -> R<&[u8]> {
        gb.get((g.dados + t.off) as usize..(g.dados + t.off + n) as usize).ok_or_else(|| Erro(format!("GGUF: {} fora do arquivo", t.nome)))
    };
    for t in &g.tensores {
        let el: u64 = t.ne.iter().product();
        let hf = nome_hf(&t.nome).and_then(|h| por_nome.get(h.as_str()).copied());
        let como = match (t.tipo, hf) {
            (GGML_Q4_K | GGML_Q6_K, Some(i)) => {
                let e = &es[i];
                let gr = if t.tipo == GGML_Q4_K { Grade::Q4K } else { Grade::Q6K };
                if e.dtype != "BF16" || e.shape.len() != 2 || t.ne.len() != 2 || e.shape[0] != t.ne[1] || e.shape[1] != t.ne[0] || e.shape[1] % 256 != 0 {
                    return falha(format!("{}: forma não bate com {}", t.nome, e.name));
                }
                let h = if t.nome.ends_with("attn_q.weight") { Some(nh) } else if t.nome.ends_with("attn_k.weight") { Some(nkv) } else { None };
                Como::Q(i, gr, h)
            }
            (GGML_F32, Some(i)) if es[i].dtype == "BF16" && es[i].shape.iter().product::<u64>() == el => {
                let e = &es[i];
                let f = fim_dados(t, 4 * el)?;
                let ok = dados[e.begin as usize..e.end as usize].chunks_exact(2).zip(f.chunks_exact(4)).all(|(b, x)| x[0] == 0 && x[1] == 0 && x[2..] == b[..]);
                if ok {
                    Como::F32(i)
                } else {
                    Como::Bruto(f.to_vec())
                }
            }
            (GGML_F32, None) => Como::Bruto(fim_dados(t, 4 * el)?.to_vec()),
            _ => return falha(format!("{}: tipo {} não suportado pelo escritor", t.nome, t.tipo)),
        };
        match &como {
            Como::Q(i, ..) | Como::F32(i) => {
                if std::mem::replace(&mut usados[*i], true) {
                    return falha(format!("{} usado duas vezes", es[*i].name));
                }
            }
            Como::Bruto(_) => {}
        }
        partes.push((t, como));
    }

    // tensores do .strata: os do safetensors (na ordem dos dados) e os extras do GGUF
    struct St {
        nome: String,
        dtype: String,
        shape: Vec<u64>,
        bytes: Vec<u8>,
        res: Option<(Grade, Option<usize>)>,
        extra: bool,
    }
    let mut ts: Vec<St> = es
        .iter()
        .map(|e| St { nome: e.name.clone(), dtype: e.dtype.clone(), shape: e.shape.clone(), bytes: dados[e.begin as usize..e.end as usize].to_vec(), res: None, extra: false })
        .collect();
    let mut gguf_partes = vec![json!({"t": ts.len(), "como": "cabecalho"})];
    ts.push(St { nome: "gguf:cabecalho".into(), dtype: "U8".into(), shape: vec![g.dados], bytes: gb[..g.dados as usize].to_vec(), res: None, extra: true });
    for (t, como) in &partes {
        match como {
            Como::Q(i, gr, h) => {
                ts[*i].res = Some((*gr, *h));
                gguf_partes.push(json!({"t": i, "como": "q"}));
            }
            Como::F32(i) => gguf_partes.push(json!({"t": i, "como": "f32"})),
            Como::Bruto(b) => {
                gguf_partes.push(json!({"t": ts.len(), "como": "bruto"}));
                ts.push(St { nome: format!("gguf:{}", t.nome), dtype: "U8".into(), shape: vec![b.len() as u64], bytes: b.clone(), res: None, extra: true });
            }
        }
    }
    // tensores que o GGUF não tem (ou tem num tipo sem grade) ficam crus, no estrato 0

    // streams: (estrato, tensor, parte, bytes gravados, tamanho cru, codificação)
    type Stream = (u64, usize, String, Vec<u8>, u64, &'static str);
    let mut sts: Vec<Stream> = Vec::new();
    let mut somas: BTreeMap<String, Soma> = BTreeMap::new();
    let mut gguf_q: HashMap<usize, &[u8]> = HashMap::new();
    for (t, como) in &partes {
        if let Como::Q(i, gr, _) = como {
            let n = ts[*i].shape.iter().product::<u64>() as usize;
            gguf_q.insert(*i, fim_dados(t, (n / 256 * gr.bytes_bloco()) as u64)?);
        }
    }
    for (ti, t) in ts.iter().enumerate() {
        let Some((gr, h)) = t.res else {
            let (b, c) = deflate_se_menor(&t.bytes);
            sts.push((0, ti, "raw".into(), b, t.bytes.len() as u64, c));
            continue;
        };
        let (linhas, cols) = (t.shape[0] as usize, t.shape[1] as usize);
        let w: Vec<u16> = t.bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let wg = match h {
            Some(h) => permuta(&w, linhas, cols, h, true),
            None => w,
        };
        let base = gguf_q[&ti];
        let grs = Grades { g: gr, base };
        let c = codifica(&grs, &wg, k, nt)?;
        let n = wg.len() as u64;
        for chave in [gr.nome(), "total"].into_iter().chain(t.nome.starts_with("model.layers.").then_some("camadas")) {
            let s = somas.entry(chave.to_string()).or_default();
            s.pesos += n;
            s.base += 8 * base.len() as u64;
            s.niveis += k as u64 * n;
            s.fecho += 8 * c.fecho.len() as u64;
            s.escapes += c.escapes;
        }
        sts.push((0, ti, "q".into(), base.to_vec(), base.len() as u64, "raw"));
        for (j, p) in c.planos.into_iter().enumerate() {
            sts.push((j as u64 + 1, ti, format!("n{}", j + 1), p, n / 8, "raw"));
        }
        sts.push((k as u64 + 1, ti, "f".into(), c.fecho, 2 * n, "fecho"));
    }
    sts.sort_by_key(|s| (s.0, s.1));

    let mut off = 0u64;
    let mut streams = Vec::with_capacity(sts.len());
    let mut prefixos = vec![0u64; k as usize + 2];
    for (s, t, p, b, r, c) in &sts {
        streams.push(json!({"t": t, "s": s, "p": p, "o": off, "l": b.len(), "r": r, "c": c, "h": crate::sha256_hex(b)}));
        off += b.len() as u64;
        prefixos[*s as usize] = off;
    }
    let tensors: Vec<Value> = ts
        .iter()
        .map(|t| {
            let mut o = json!({"name": t.nome, "dtype": t.dtype, "shape": t.shape, "bytes": t.bytes.len(), "sha256": crate::sha256_hex(&t.bytes)});
            let m = o.as_object_mut().unwrap();
            match t.res {
                Some((gr, h)) => {
                    m.insert("n".into(), json!(t.shape.iter().product::<u64>()));
                    m.insert("residual".into(), json!({"grade": gr.nome(), "permuta": h}));
                }
                None => {
                    m.insert("n".into(), Value::Null);
                }
            }
            if t.extra {
                m.insert("extra".into(), json!("gguf"));
            }
            o
        })
        .collect();
    let eng = base64::engine::general_purpose::STANDARD;
    let h = json!({
        "format": "strata/0.3",
        "ordem": "estrato-maior",
        "estratos": k + 2,
        "niveis": k,
        "tensors": tensors,
        "streams": streams,
        "fonte": {"tipo": "safetensors", "header_b64": base64::Engine::encode(&eng, hb), "sha256": crate::sha256_hex(st)},
        "gguf": {"alinhamento": g.alinhamento, "partes": gguf_partes, "sha256": crate::sha256_hex(gb)},
        "pai": null,
        "integridade": {"alg": "sha256"},
    });
    let hj = serde_json::to_vec(&h).map_err(|e| Erro(e.to_string()))?;
    let mut arquivo = Vec::with_capacity(12 + hj.len() + off as usize);
    arquivo.extend_from_slice(&crate::MAGIC);
    arquivo.push(3);
    arquivo.extend_from_slice(&(hj.len() as u32).to_le_bytes());
    arquivo.extend_from_slice(&hj);
    let base = arquivo.len() as u64;
    for (_, _, _, b, _, _) in &sts {
        arquivo.extend_from_slice(b);
    }
    // estratos vazios herdam o prefixo anterior
    for i in 1..prefixos.len() {
        prefixos[i] = prefixos[i].max(prefixos[i - 1]);
    }
    let prefixos = prefixos.into_iter().map(|p| base + p).collect();
    Ok(Escrito { arquivo, somas, prefixos })
}
