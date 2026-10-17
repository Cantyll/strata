//! strata-ref: linha de comando do leitor de referência.
//!
//!   strata-ref confere <arq.strata> [--exigir] [--id HEX] [--threads N]   valida, decodifica e confere os hashes
//!   strata-ref volta <arq.strata> <saida.safetensors>        reconstrói o .safetensors original
//!   strata-ref id <arq.strata>                               imprime o id (SHA-256 do cabeçalho)
//!   strata-ref lote <pasta>                                  uma linha "nome classe sha" por arquivo (fuzzing)
use std::{env, fs, panic, process::exit, time::Instant};
use strata_ref::{cabecalho_id, decode, to_safetensors, Opcoes};

fn main() {
    let a: Vec<String> = env::args().collect();
    let op = Opcoes {
        exigir_integridade: a.iter().any(|x| x == "--exigir"),
        id_esperado: a.iter().position(|x| x == "--id").and_then(|i| a.get(i + 1).cloned()),
        limite_saida: None,
        threads: a.iter().position(|x| x == "--threads").and_then(|i| a.get(i + 1)?.parse().ok()).unwrap_or(0),
    };
    let ler = |p: &str| fs::read(p).unwrap_or_else(|e| { eprintln!("{p}: {e}"); exit(2) });
    match (a.get(1).map(String::as_str), a.get(2), a.get(3)) {
        (Some("confere"), Some(p), _) => {
            let f = ler(p);
            let t0 = Instant::now();
            match decode(&f, u32::MAX, &op) {
                Ok((arq, ts)) => {
                    let s = t0.elapsed().as_secs_f64();
                    let saida: usize = ts.iter().map(Vec::len).sum();
                    println!(
                        "ok: {} tensores, {} bytes de saída, hashes {}, {:.2} s ({:.0} MB/s)",
                        arq.tensores.len(),
                        saida,
                        if arq.integridade { "conferidos" } else { "ausentes" },
                        s,
                        saida as f64 / 1e6 / s
                    );
                }
                Err(e) => {
                    println!("recusado: {e}");
                    exit(1)
                }
            }
        }
        (Some("volta"), Some(p), Some(saida)) => match to_safetensors(&ler(p), &op) {
            Ok(st) => fs::write(saida, st).unwrap_or_else(|e| { eprintln!("{saida}: {e}"); exit(2) }),
            Err(e) => {
                println!("recusado: {e}");
                exit(1)
            }
        },
        (Some("id"), Some(p), _) => match cabecalho_id(&ler(p)) {
            Ok(id) => println!("{id}"),
            Err(e) => {
                println!("recusado: {e}");
                exit(1)
            }
        },
        (Some("lote"), Some(dir), _) => {
            panic::set_hook(Box::new(|_| {}));
            let mut nomes: Vec<_> = fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
            nomes.sort();
            for p in nomes {
                let f = fs::read(&p).unwrap();
                let nome = p.file_name().unwrap().to_string_lossy();
                match panic::catch_unwind(|| to_safetensors(&f, &op)) {
                    Ok(Ok(st)) => println!("{nome} ok {}", strata_ref::sha256_hex(&st)),
                    Ok(Err(_)) => println!("{nome} recusa -"),
                    Err(_) => println!("{nome} quebra -"),
                }
            }
        }
        _ => {
            eprintln!("uso: strata-ref confere|volta|id|lote ... (ver src/main.rs)");
            exit(2)
        }
    }
}
