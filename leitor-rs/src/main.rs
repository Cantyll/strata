//! strata-ref: linha de comando do leitor de referência.
//!
//!   strata-ref confere <arq.strata> [--exigir] [--id HEX] [--threads N]   valida, decodifica e confere os hashes
//!   strata-ref volta <arq.strata> <saida.safetensors>        reconstrói o .safetensors original
//!   strata-ref fluxo <arq.strata> [MB por pedaço] [MB/s]     lê em fluxo, como num download (no ritmo dado), e mede
//!   strata-ref id <arq.strata>                               imprime o id (SHA-256 do cabeçalho)
//!   strata-ref residual <orig.safetensors> <base.gguf> <saida.strata> [--niveis K]
//!                                                            escreve um .strata 0.3 em modo residual (A.14)
//!   strata-ref gguf <arq.strata> <saida.gguf>                remonta o GGUF do estrato 0 (0.3)
//!   strata-ref distribui <base.gguf> <pasta> <shard1.safetensors> ... [--index index.json] [--niveis K]
//!                       [--espera] [--apaga-shards]   espera cada shard aparecer / apaga depois de ler
//!                                                            arquivos de estrato ao lado de um GGUF publicado (B.10)
//!   strata-ref junta <model.strata.index.json> [pasta] [--base x.gguf] [--prof D] [--tudo] [--espera [--espera-s S]]
//!                                                            remonta os shards (sem pasta: só confere os SHA-256)
//!   --prof D em confere e fluxo: lê só até o estrato D
//!   strata-ref lote <pasta>                                  uma linha "nome classe sha" por arquivo (fuzzing)
use std::{env, fs, panic, process::exit, time::Instant};
use strata_ref::{cabecalho_id, decode, para_gguf, to_safetensors, Fluxo, Opcoes};

fn main() {
    let a: Vec<String> = env::args().collect();
    let op = Opcoes {
        exigir_integridade: a.iter().any(|x| x == "--exigir"),
        id_esperado: a.iter().position(|x| x == "--id").and_then(|i| a.get(i + 1).cloned()),
        limite_saida: None,
        threads: a.iter().position(|x| x == "--threads").and_then(|i| a.get(i + 1)?.parse().ok()).unwrap_or(0),
    };
    let num = |f: &str| a.iter().position(|x| x == f).and_then(|i| a.get(i + 1)?.parse::<u32>().ok());
    let prof = num("--prof").unwrap_or(u32::MAX);
    let ler = |p: &str| fs::read(p).unwrap_or_else(|e| { eprintln!("{p}: {e}"); exit(2) });
    match (a.get(1).map(String::as_str), a.get(2), a.get(3)) {
        (Some("confere"), Some(p), _) => {
            let f = ler(p);
            let t0 = Instant::now();
            match decode(&f, prof, &op) {
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
        (Some("fluxo"), Some(p), mb) => {
            let f = ler(p);
            let ped = mb.and_then(|m| m.parse::<usize>().ok()).unwrap_or(8) << 20;
            // rede simulada: o pedaço k só "chega" em t0 + bytes até ele / taxa
            let taxa = a.get(4).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0) * 1e6;
            let hl = 12 + f.get(8..12).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()) as usize);
            let t0 = Instant::now();
            let r = Fluxo::novo(&f[..hl.min(f.len())], prof, &op).and_then(|mut fx| {
                let mut chegou = hl.min(f.len());
                let fim = fx.arquivo().bytes_ate(prof) as usize;
                for c in f[hl.min(f.len())..fim.max(hl).min(f.len())].chunks(ped) {
                    chegou += c.len();
                    if taxa > 0.0 {
                        let quando = chegou as f64 / taxa;
                        let agora = t0.elapsed().as_secs_f64();
                        if quando > agora {
                            std::thread::sleep(std::time::Duration::from_secs_f64(quando - agora));
                        }
                    }
                    fx.alimenta(c)?;
                }
                let t1 = t0.elapsed().as_secs_f64();
                fx.termina().map(|(_, ts)| (t1, ts))
            });
            match r {
                Ok((t1, ts)) => println!(
                    "ok: {} bytes de saída; até o último byte {:.2} s, depois dele {:.2} s",
                    ts.iter().map(Vec::len).sum::<usize>(),
                    t1,
                    t0.elapsed().as_secs_f64() - t1
                ),
                Err(e) => {
                    println!("recusado: {e}");
                    exit(1)
                }
            }
        }
        (Some("residual"), Some(st), Some(gg)) => {
            let Some(saida) = a.get(4) else { eprintln!("falta o arquivo de saída"); exit(2) };
            let t0 = Instant::now();
            let nt = if op.threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { op.threads };
            match strata_ref::residual::escreve(&ler(st), &ler(gg), num("--niveis").unwrap_or(3), nt) {
                Ok(e) => {
                    fs::write(saida, &e.arquivo).unwrap_or_else(|x| { eprintln!("{saida}: {x}"); exit(2) });
                    for (g, s) in &e.somas {
                        let b = |x: u64| x as f64 / s.pesos as f64;
                        println!(
                            "{g:>8}: {:>5.0} M pesos  estrato 0 {:.3} + níveis {:.3} + fecho {:.3} = {:.3} bits/peso  ({} escapes)",
                            s.pesos as f64 / 1e6, b(s.base), b(s.niveis), b(s.fecho), b(s.base + s.niveis + s.fecho), s.escapes
                        );
                    }
                    let pesos = e.somas.get("total").map_or(1, |s| s.pesos) as f64;
                    for (d, p) in e.prefixos.iter().enumerate() {
                        println!("até o estrato {d}: {p} bytes ({:.3} bits por peso das matrizes)", 8.0 * *p as f64 / pesos);
                    }
                    println!("{} bytes em {:.0} s", e.arquivo.len(), t0.elapsed().as_secs_f64());
                }
                Err(e) => {
                    println!("recusado: {e}");
                    exit(1)
                }
            }
        }
        (Some("distribui"), Some(gg), Some(dir)) => {
            let t0 = Instant::now();
            let nt = if op.threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { op.threads };
            let opcoes = ["--index", "--niveis", "--threads"];
            let shards: Vec<std::path::PathBuf> = a[4..]
                .iter()
                .enumerate()
                .filter(|(i, x)| !x.starts_with("--") && !(*i > 0 && opcoes.contains(&a[4 + i - 1].as_str())))
                .map(|(_, x)| x.into())
                .collect();
            let idx = a.iter().position(|x| x == "--index").and_then(|i| a.get(i + 1)).map(std::path::PathBuf::from);
            let _ = fs::create_dir_all(dir);
            let tem = |f: &str| a.iter().any(|x| x == f);
            match strata_ref::dist::escreve_com(&shards, idx.as_deref(), gg.as_ref(), dir.as_ref(), num("--niveis").unwrap_or(3), nt, tem("--espera"), tem("--apaga-shards")) {
                Ok(r) => {
                    for (g, s) in &r.somas {
                        let b = |x: u64| x as f64 / s.pesos as f64;
                        println!(
                            "{g:>8}: {:>5.0} M pesos  estrato 0 {:.3} + níveis {:.3} + fecho {:.3} = {:.3} bits/peso  ({} escapes)",
                            s.pesos as f64 / 1e6, b(s.base), b(s.niveis), b(s.fecho), b(s.base + s.niveis + s.fecho), s.escapes
                        );
                    }
                    for (n, s, b) in &r.arquivos {
                        println!("{:>14} bytes  estrato {s:>2}  {n}", b);
                    }
                    println!("{:>14} bytes  índice", r.indice);
                    println!("{:.0} s", t0.elapsed().as_secs_f64());
                }
                Err(e) => {
                    println!("recusado: {e}");
                    exit(1)
                }
            }
        }
        (Some("junta"), Some(idx), dir) => {
            let t0 = Instant::now();
            let nt = if op.threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { op.threads };
            let dir = dir.filter(|d| !d.starts_with("--"));
            let base = a.iter().position(|x| x == "--base").and_then(|i| a.get(i + 1)).map(std::path::PathBuf::from);
            let prof = num("--prof");
            if let Some(d) = dir {
                let _ = fs::create_dir_all(d);
            }
            let espera = a.iter().any(|x| x == "--espera").then(|| std::time::Duration::from_secs(num("--espera-s").unwrap_or(600) as u64));
            match strata_ref::dist::volta_com(idx.as_ref(), base.as_deref(), dir.map(|d| d.as_ref()), prof, nt, a.iter().any(|x| x == "--tudo"), espera) {
                Ok(v) => {
                    for (n, h, b) in &v.shards {
                        println!("{n}: {b} bytes, sha256 {h}{}", if prof.is_none() { " (confere)" } else { "" });
                    }
                    println!("ok: {} tensores em {:.0} s", v.tensores, t0.elapsed().as_secs_f64());
                    eprintln!("tempos: {}", v.tempos.iter().map(|(n, t)| format!("{n} {t:.1} s")).collect::<Vec<_>>().join(" · "));
                }
                Err(e) => {
                    println!("recusado: {e}");
                    exit(1)
                }
            }
        }
        (Some("gguf"), Some(p), Some(saida)) => match para_gguf(&ler(p), &op) {
            Ok(g) => fs::write(saida, g).unwrap_or_else(|e| { eprintln!("{saida}: {e}"); exit(2) }),
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
            eprintln!("uso: strata-ref confere|volta|fluxo|residual|gguf|distribui|junta|id|lote ... (ver src/main.rs)");
            exit(2)
        }
    }
}
