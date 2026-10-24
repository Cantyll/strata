//! Distribuição sobre um GGUF publicado (B.10): arquivos de estrato ao lado do GGUF, volta bit a
//! bit para o safetensors, leituras parciais iguais às do .strata 0.3 de um arquivo e recusas.
use std::fs;
use std::path::{Path, PathBuf};
use strata_ref::{decode, dist, residual, sha256_hex, Opcoes};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/vetores/residual/");

fn pasta(nome: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("strata-dist-{nome}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}
fn copia_gguf(d: &Path) -> PathBuf {
    let p = d.join("modelo.gguf");
    fs::copy(format!("{DIR}modelo.gguf"), &p).unwrap();
    p
}
fn escreve(d: &Path, k: u32) {
    let gg = copia_gguf(d);
    dist::escreve(&[format!("{DIR}modelo.safetensors").into()], None, &gg, d, k, 2).unwrap();
}

#[test]
fn volta_bit_a_bit_e_niveis_iguais_ao_arquivo_unico() {
    let d = pasta("volta");
    escreve(&d, 3);
    let idx = d.join("model.strata.index.json");
    let v = dist::volta(&idx, None, None, None, 2, true).unwrap();
    let orig = fs::read(format!("{DIR}modelo.safetensors")).unwrap();
    assert_eq!(v.shards[0].1, sha256_hex(&orig));
    let o = pasta("volta-saida");
    dist::volta(&idx, None, Some(&o), None, 3, false).unwrap();
    assert_eq!(fs::read(o.join("modelo.safetensors")).unwrap(), orig);
    // leitura até o nível d = a mesma do .strata 0.3 de um arquivo só, tensor a tensor
    let st = residual::escreve(&orig, &fs::read(format!("{DIR}modelo.gguf")).unwrap(), 3, 2).unwrap().arquivo;
    for p in 0..=3u32 {
        let o = pasta(&format!("prof{p}"));
        dist::volta(&idx, None, Some(&o), Some(p), 1, false).unwrap();
        let parcial = fs::read(o.join("modelo.safetensors")).unwrap();
        let (a, ts) = decode(&st, p, &Opcoes::default()).unwrap();
        let hl = u64::from_le_bytes(parcial[..8].try_into().unwrap()) as usize;
        let mut pos = 8 + hl;
        let (es, _) = strata_ref::valida_safetensors(&parcial[8..8 + hl], None).unwrap();
        for e in es {
            let i = a.tensores.iter().position(|t| t.name == e.name).unwrap();
            assert_eq!(&parcial[pos..pos + (e.end - e.begin) as usize], &ts[i][..], "prof {p} {}", e.name);
            pos += (e.end - e.begin) as usize;
        }
    }
}

#[test]
fn recusa_qualquer_byte_trocado() {
    let d = pasta("recusa");
    escreve(&d, 2);
    let idx = d.join("model.strata.index.json");
    let mut nomes: Vec<_> = fs::read_dir(&d).unwrap().map(|e| e.unwrap().path()).filter(|p| p != &idx).collect();
    nomes.sort();
    for p in nomes {
        let b = fs::read(&p).unwrap();
        for i in [b.len() / 3, b.len() / 2, b.len() - 1] {
            let mut m = b.clone();
            m[i] ^= 0x01;
            fs::write(&p, &m).unwrap();
            // o GGUF inteiro só é conferido com --tudo; os bytes dele que entram na saída, sempre
            let tudo = p.extension().is_some_and(|x| x == "gguf");
            assert!(dist::volta(&idx, None, None, None, 2, tudo).is_err(), "{p:?} byte {i}");
        }
        fs::write(&p, &b[..b.len() - 1]).unwrap();
        assert!(dist::volta(&idx, None, None, None, 2, false).is_err(), "{p:?} truncado");
        fs::write(&p, &b).unwrap();
    }
    assert!(dist::volta(&idx, None, None, None, 2, false).is_ok());
    // sem --tudo, um byte trocado dentro de um stream q do GGUF é recusado
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(&idx).unwrap()).unwrap();
    let q = v["streams"].as_array().unwrap().iter().find(|s| s["p"] == "q").unwrap();
    let gp = d.join(v["arquivos"][0]["nome"].as_str().unwrap());
    let b = fs::read(&gp).unwrap();
    let mut m = b.clone();
    m[(q["o"].as_u64().unwrap() + q["l"].as_u64().unwrap() / 2) as usize] ^= 0x01;
    fs::write(&gp, &m).unwrap();
    assert!(dist::volta(&idx, None, None, None, 2, false).is_err());
    fs::write(&gp, &b).unwrap();
    // índice com um stream apontando para fora do arquivo, ou para outro estrato
    let ib = fs::read_to_string(&idx).unwrap();
    let mut v: serde_json::Value = serde_json::from_str(&ib).unwrap();
    let s = v["streams"].as_array_mut().unwrap().iter_mut().find(|s| s["p"] == "n1").unwrap();
    s["o"] = serde_json::json!(1u64 << 40);
    fs::write(&idx, serde_json::to_vec(&v).unwrap()).unwrap();
    assert!(dist::volta(&idx, None, None, None, 2, false).is_err());
    let mut v: serde_json::Value = serde_json::from_str(&ib).unwrap();
    let s = v["streams"].as_array_mut().unwrap().iter_mut().find(|s| s["p"] == "n1").unwrap();
    s["f"] = serde_json::json!(4);
    fs::write(&idx, serde_json::to_vec(&v).unwrap()).unwrap();
    assert!(dist::volta(&idx, None, None, None, 2, false).is_err());
}

#[test]
fn nivel_parcial_nao_abre_o_fecho() {
    let d = pasta("parcial");
    escreve(&d, 3);
    let idx = d.join("model.strata.index.json");
    // sem os arquivos dos níveis 2, 3 e do fecho: até o nível 1 ainda funciona
    for s in 2..=4 {
        fs::remove_file(d.join(format!("model-e{s:02}-00000-of-00001.strata"))).unwrap();
    }
    assert!(dist::volta(&idx, None, None, Some(1), 2, false).is_ok());
    assert!(dist::volta(&idx, None, None, Some(2), 2, false).is_err());
    assert!(dist::volta(&idx, None, None, None, 2, false).is_err());
}

#[test]
fn espera_os_arquivos_chegarem() {
    use std::io::Write;
    use std::time::Duration;
    let d = pasta("espera-origem");
    escreve(&d, 3);
    let orig = fs::read(format!("{DIR}modelo.safetensors")).unwrap();
    // a pasta de chegada começa só com o índice e o GGUF; os estratos chegam aos poucos, em ordem
    let c = pasta("espera-chegada");
    fs::copy(d.join("model.strata.index.json"), c.join("model.strata.index.json")).unwrap();
    copia_gguf(&c);
    let nomes: Vec<String> = (0..=4).map(|s| format!("model-e{s:02}-00000-of-00001.strata")).collect();
    let (dd, cc, nn) = (d.clone(), c.clone(), nomes.clone());
    let baixa = std::thread::spawn(move || {
        let bs: Vec<Vec<u8>> = nn.iter().map(|n| fs::read(dd.join(n)).unwrap()).collect();
        let mut fs_: Vec<fs::File> = nn.iter().map(|n| fs::File::create(cc.join(n)).unwrap()).collect();
        let mut pos = vec![0usize; bs.len()];
        while pos.iter().zip(&bs).any(|(p, b)| p < &b.len()) {
            for (i, b) in bs.iter().enumerate() {
                let fim = (pos[i] + 997).min(b.len());
                fs_[i].write_all(&b[pos[i]..fim]).unwrap();
                pos[i] = fim;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });
    let o = pasta("espera-saida");
    let idx = c.join("model.strata.index.json");
    let v = dist::volta_com(&idx, None, Some(&o), None, 2, false, Some(Duration::from_secs(10))).unwrap();
    baixa.join().unwrap();
    assert_eq!(v.shards[0].1, sha256_hex(&orig));
    assert_eq!(fs::read(o.join("modelo.safetensors")).unwrap(), orig);
    // um arquivo que para no meio: desiste depois do prazo, não devolve nada errado
    let p = c.join(&nomes[4]);
    let b = fs::read(&p).unwrap();
    fs::write(&p, &b[..b.len() / 2]).unwrap();
    assert!(dist::volta_com(&idx, None, None, None, 2, false, Some(Duration::from_millis(300))).is_err());
    // --espera e --tudo não combinam
    assert!(dist::volta_com(&idx, None, None, None, 2, true, Some(Duration::from_secs(1))).is_err());
}
