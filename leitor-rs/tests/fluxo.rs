//! A leitura em fluxo (Fluxo) tem que aceitar e recusar o mesmo que `decode`, com os mesmos bytes
//! de saída, em qualquer tamanho de pedaço: nos vetores públicos, nos prefixos e em milhares de
//! arquivos mutados.
use serde_json::Value;
use std::fs;
use strata_ref::{decode, sha256_hex, Fluxo, Opcoes, R};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/vetores/");

/// Entrega `f` em pedaços de `pedaco` bytes (o primeiro com o cabeçalho inteiro) e junta com `prof`.
fn em_fluxo(f: &[u8], pedaco: usize, prof: u32, op: &Opcoes) -> R<Vec<Vec<u8>>> {
    let cab = if f.len() >= 12 { (12 + u32::from_le_bytes(f[8..12].try_into().unwrap()) as usize).min(f.len()) } else { f.len() };
    let mut fx = Fluxo::novo(&f[..cab], op)?;
    for c in f[cab..].chunks(pedaco.max(1)) {
        fx.alimenta(c)?;
    }
    Ok(fx.termina(prof)?.1)
}
fn igual(a: &R<Vec<Vec<u8>>>, b: &R<Vec<Vec<u8>>>) -> bool {
    match (a, b) {
        (Ok(x), Ok(y)) => x == y,
        (Err(_), Err(_)) => true,
        _ => false,
    }
}

#[test]
fn vetores_em_qualquer_pedaco() {
    let man: Value = serde_json::from_slice(&fs::read(format!("{DIR}vetores.json")).unwrap()).unwrap();
    let op = Opcoes { threads: 3, ..Default::default() };
    for v in man["vetores"].as_array().unwrap() {
        let nome = v["arquivo"].as_str().unwrap();
        let f = fs::read(format!("{DIR}{nome}")).unwrap();
        let lote = decode(&f, u32::MAX, &op).map(|x| x.1);
        // "recusa" vale para a volta ao .safetensors; o SHA-256 do original só é conferido nela
        assert!(lote.is_ok() || v["espera"] == "recusa", "{nome}");
        for pedaco in [1, 3, 64, 4096, usize::MAX] {
            assert!(igual(&lote, &em_fluxo(&f, pedaco, u32::MAX, &op)), "{nome} em pedaços de {pedaco}");
        }
        for p in v["prefixos"].as_array().into_iter().flatten() {
            let (d, n) = (p["profundidade"].as_u64().unwrap() as u32, p["bytes"].as_u64().unwrap() as usize);
            let ts = em_fluxo(&f[..n], 5, d, &op).unwrap_or_else(|e| panic!("{nome} d={d}: {e}"));
            assert_eq!(sha256_hex(&ts.concat()), p["sha256_tensores"].as_str().unwrap(), "{nome} d={d}");
            assert!(em_fluxo(&f[..n - 1], 5, d, &op).is_err(), "{nome} d={d}: leu sem o último byte");
        }
    }
}

#[test]
fn mutacoes_dao_o_mesmo_que_decode() {
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move |n: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n.max(1) as u64) as usize
    };
    let op = Opcoes { threads: 2, ..Default::default() };
    let (mut aceitos, mut casos) = (0, 0);
    for nome in ["ok-exemplo.strata", "ok-tipos-mistos.strata", "ok-sem-hashes.strata"] {
        let orig = fs::read(format!("{DIR}{nome}")).unwrap();
        for _ in 0..1500 {
            let mut f = orig.clone();
            match rnd(4) {
                0 => {
                    let i = rnd(f.len());
                    f[i] ^= 1 << rnd(8);
                }
                1 => f.truncate(rnd(f.len())),
                2 => {
                    for _ in 0..1 + rnd(4) {
                        let i = rnd(f.len());
                        f[i] = rnd(256) as u8;
                    }
                }
                _ => f.extend(std::iter::repeat_n(rnd(256) as u8, 1 + rnd(8))),
            }
            let lote = decode(&f, u32::MAX, &op).map(|x| x.1);
            let fluxo = em_fluxo(&f, 1 + rnd(200), u32::MAX, &op);
            assert!(igual(&lote, &fluxo), "{nome}: decode {:?} × fluxo {:?}", lote.as_ref().err(), fluxo.as_ref().err());
            aceitos += lote.is_ok() as u32;
            casos += 1;
        }
    }
    println!("{casos} mutações, {aceitos} aceitas pelos dois");
}
