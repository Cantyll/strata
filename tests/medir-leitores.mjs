// Mede, num modelo real, o custo da segurança: tamanho dos hashes no cabeçalho e velocidade de leitura.
//   node --max-old-space-size=8192 tests/medir-leitores.mjs codifica modelo.safetensors saida.strata
//   node --max-old-space-size=8192 tests/medir-leitores.mjs le saida.strata [original.safetensors] [--core outro-core.js]
// "le" reconstrói o .safetensors com toSafetensors (todas as regras e hashes) e confere contra o original.
// Com --core aponta para outro strata-core.js (ex.: o leitor de antes do endurecimento, via git show).
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';

const args = process.argv.slice(2), iCore = args.indexOf('--core');
const core = iCore >= 0 ? path.resolve(args.splice(iCore, 2)[1]) : new URL('../src/strata-core.js', import.meta.url).pathname;
new Function(fs.readFileSync(core, 'utf8'))();
const S = globalThis.Strata;
const sha = (u8) => { // em pedaços: update não aceita mais de 2 GiB de uma vez
  const h = createHash('sha256'); for (let o = 0; o < u8.length; o += 1 << 30) h.update(u8.subarray(o, o + (1 << 30))); return h.digest('hex');
};
const s = (t0) => ((performance.now() - t0) / 1000).toFixed(1) + ' s';
const [modo, entrada, outro] = args;
// readFileSync não passa de 2 GiB
function lerGrande(p) {
  const fd = fs.openSync(p, 'r'), n = fs.fstatSync(fd).size, u = new Uint8Array(n);
  for (let o = 0; o < n;) o += fs.readSync(fd, u, o, Math.min(1 << 30, n - o), o);
  fs.closeSync(fd); return u;
}

if (modo === 'codifica') {
  const st = lerGrande(entrada), t0 = performance.now();
  const src = S.readSafetensors(st), shaSt = sha(st);
  const enc = await S.encode(src.tensors, { sourceHeaderBytes: src.headerBytes, sourceSha256: shaSt });
  fs.writeFileSync(outro, enc.file);
  // o mesmo cabeçalho sem os hashes, para medir quanto eles custam
  const h = structuredClone(enc.header);
  delete h.integridade; for (const e of h.streams) delete e.h; for (const t of h.tensors) delete t.sha256; delete h.fonte.sha256;
  const semHash = new TextEncoder().encode(JSON.stringify(h)).length;
  const nf = enc.items.filter((x) => x.kind === 'float').reduce((a, x) => a + x.n, 0);
  console.log(`${enc.items.length} tensores, ${enc.table.length} streams, ${nf} pesos; codificado em ${s(t0)}`);
  console.log(`.strata ${enc.file.length} B = ${(8 * enc.file.length / nf).toFixed(3)} bits/peso; safetensors original sha256 ${shaSt}`);
  console.log(`cabeçalho: ${semHash} B sem hashes -> ${enc.headerLen} B com hashes (+${enc.headerLen - semHash} B = ${(100 * (enc.headerLen - semHash) / enc.file.length).toFixed(4)}% do arquivo, ${(8 * (enc.headerLen - semHash) / nf).toFixed(5)} bits/peso)`);
  console.log(`id do cabeçalho: ${await S.cabecalhoId(enc.file)}`);
} else if (modo === 'le') {
  const f = lerGrande(entrada), t0 = performance.now();
  let out;
  if (S.toSafetensors) out = await S.toSafetensors(f);
  else { const d = await S.decode(f); out = S.rebuildSafetensors(d.sourceHeaderBytes, d.tensors); }
  const dt = (performance.now() - t0) / 1000;
  console.log(`${path.basename(core)}: ${f.length} B -> ${out.length} B em ${dt.toFixed(1)} s (${(out.length / 1e6 / dt).toFixed(0)} MB/s de saída)`);
  if (outro) console.log('SHA-256 igual ao original:', sha(out) === sha(lerGrande(outro)));
} else {
  console.error('uso: medir-leitores.mjs codifica|le ...'); process.exit(2);
}
