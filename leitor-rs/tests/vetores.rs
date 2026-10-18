//! Confere o leitor Rust contra os vetores públicos de tests/vetores (gerados pelo leitor JS).
use serde_json::Value;
use std::fs;
use strata_ref::{abre, cabecalho_id, decode, sha256_hex, to_safetensors, Opcoes};

#[test]
fn vetores() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/vetores/");
    let man: Value = serde_json::from_slice(&fs::read(format!("{dir}vetores.json")).unwrap()).unwrap();
    let (mut ok, mut recusados) = (0, 0);
    for v in man["vetores"].as_array().unwrap() {
        let nome = v["arquivo"].as_str().unwrap();
        let f = fs::read(format!("{dir}{nome}")).unwrap();
        let r = to_safetensors(&f, &Opcoes::default());
        if v["espera"] == "recusa" {
            assert!(r.is_err(), "{nome}: deveria ser recusado");
            recusados += 1;
            continue;
        }
        let st = r.unwrap_or_else(|e| panic!("{nome}: {e}"));
        assert_eq!(sha256_hex(&st), v["sha256_safetensors"].as_str().unwrap(), "{nome}");
        if let Some(id) = v["id"].as_str() {
            assert_eq!(cabecalho_id(&f).unwrap(), id, "{nome}: id");
            let com_id = Opcoes { id_esperado: Some(id.into()), exigir_integridade: true, ..Default::default() };
            assert!(to_safetensors(&f, &com_id).is_ok());
        }
        for p in v["prefixos"].as_array().into_iter().flatten() {
            let (d, n) = (p["profundidade"].as_u64().unwrap() as u32, p["bytes"].as_u64().unwrap() as usize);
            assert_eq!(abre(&f, &Opcoes::default()).unwrap().bytes_ate(d), n as u64, "{nome} d={d}: bytes_ate");
            let (_, ts) = decode(&f[..n], d, &Opcoes::default()).unwrap_or_else(|e| panic!("{nome} d={d}: {e}"));
            assert_eq!(sha256_hex(&ts.concat()), p["sha256_tensores"].as_str().unwrap(), "{nome} d={d}");
            // um byte a menos que o prefixo tem que faltar
            assert!(decode(&f[..n - 1], d, &Opcoes::default()).is_err(), "{nome} d={d}: leu sem o último byte");
        }
        ok += 1;
    }
    let sem = fs::read(format!("{dir}ok-sem-hashes.strata")).unwrap();
    assert!(to_safetensors(&sem, &Opcoes { exigir_integridade: true, ..Default::default() }).is_err());
    println!("{ok} vetores válidos e {recusados} hostis conferidos");
}
