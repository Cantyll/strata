// Tamanho sem perdas de qualquer .safetensors (BF16, FP16, FP32, FP8), tensor por tensor, com volta bit a bit.
// Diferente de test-rans.mjs, que mede só BF16: aqui cada tipo tem os seus planos de mantissa e entra na conta.
//   1. codifica cada tensor com o core (rANS no expoente) e decodifica, conferindo byte a byte;
//   2. remonta o .safetensors em fluxo e confere o SHA-256 com o original;
//   3. imprime, por tipo, bits/peso por estrato e a fração do tamanho original.
// Uso:
//   node tests/test-tamanho.mjs modelo.safetensors [outro.safetensors ...] [--json saida.json]
// Sai com código 1 se qualquer conferência falhar.
import fs from 'node:fs';
import { createHash } from 'node:crypto';

const src = fs.readFileSync(new URL('../src/strata-core.js', import.meta.url), 'utf8');
new Function(src)();
const S = globalThis.Strata;
if (!S.hasCodec) { console.error('CompressionStream ausente: use Node 18 ou mais novo'); process.exit(2); }

let fails = 0, checks = 0;
function check(ok, msg) { checks++; if (!ok) { fails++; console.log('  FALHOU:', msg); } return ok; }
function same(a, b) { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false; return true; }
function readAt(fd, pos, len) {
  const out = new Uint8Array(len); let o = 0;
  while (o < len) { const r = fs.readSync(fd, out, o, Math.min(len - o, 1 << 30), pos + o); if (!r) throw new Error('arquivo curto'); o += r; }
  return out;
}

async function measure(path) {
  console.log(`\n=== ${path} ===`);
  const fd = fs.openSync(path, 'r'), size = fs.fstatSync(fd).size;
  const N = Number(new DataView(readAt(fd, 0, 8).buffer).getBigUint64(0, true));
  const headerBytes = readAt(fd, 8, N);
  const parsed = S.parseSafetensorsHeader(new Uint8Array(8), headerBytes);
  const base = 8 + N;
  const h0 = createHash('sha256');
  for (let p = 0; p < size; p += 1 << 28) h0.update(readAt(fd, p, Math.min(1 << 28, size - p)));
  const sha0 = h0.digest('hex');
  const h1 = createHash('sha256');
  const len8 = new Uint8Array(8); new DataView(len8.buffer).setBigUint64(0, BigInt(N), true);
  h1.update(len8); h1.update(headerBytes);
  const byType = {};      // dtype -> { n, orig, strata: bytes por estrato }
  let raw = 0, rawEnc = 0, cursor = 0;
  const t0 = performance.now();
  for (const ent of parsed.entries) {
    if (ent.begin !== cursor) throw new Error('safetensors com buraco entre tensores; esta conferência supõe dados contíguos');
    const bytes = readAt(fd, base + ent.begin, ent.end - ent.begin);
    cursor = ent.end;
    const enc = await S.encode([{ name: ent.name, dtype: ent.dtype, shape: ent.shape, bytes }]);
    const back = (await S.decode(enc.file)).tensors[0].bytes;
    check(same(back, bytes), `${ent.name}: volta difere do original`);
    h1.update(back);
    const it = enc.items[0];
    if (it.kind !== 'float') { raw += bytes.length; rawEnc += enc.table.reduce((a, e) => a + e.l, 0); continue; }
    const g = byType[ent.dtype] || (byType[ent.dtype] = { n: 0, orig: 0, bits: it.L.bits, strata: new Float64Array(it.L.m + 1) });
    g.n += it.n; g.orig += bytes.length;
    for (const e of enc.table) g.strata[e.s] += e.l;
  }
  fs.closeSync(fd);
  const sha1 = h1.digest('hex');
  check(sha0 === sha1, `SHA-256 da volta difere: ${sha0} vs ${sha1}`);
  console.log(`${parsed.entries.length} tensores em ${((performance.now() - t0) / 1000).toFixed(0)} s; ` +
    `SHA-256 original ${sha0.slice(0, 16)} volta ${sha1.slice(0, 16)} ${sha0 === sha1 ? 'IGUAL' : 'DIFERENTE'}`);
  const out = { model: path, bytes: size, sha256: sha0, roundtrip_sha_equal: sha0 === sha1, raw_bytes: raw, raw_encoded: rawEnc, types: {} };
  let total = rawEnc;
  for (const [dt, g] of Object.entries(byType)) {
    let x = 0; const cum = Array.from(g.strata, (v) => (x += v, 8 * x / g.n));
    total += x;
    console.log(`\n${dt}: ${g.n} pesos, ${(g.orig / 1e6).toFixed(1)} MB -> ${(x / 1e6).toFixed(1)} MB ` +
      `(${(100 * x / g.orig).toFixed(1)}%, ${cum[cum.length - 1].toFixed(3)} de ${g.bits} b/peso)`);
    console.log('estrato | b/peso acumulado');
    cum.forEach((v, k) => console.log(`${k} | ${v.toFixed(3)}`));
    out.types[dt] = { n: g.n, orig_bytes: g.orig, encoded_bytes: x, cum_bpw: cum };
  }
  if (raw) console.log(`\noutros tipos (crus ou deflate): ${(raw / 1e6).toFixed(1)} MB -> ${(rawEnc / 1e6).toFixed(1)} MB`);
  console.log(`arquivo inteiro (corpo, sem o cabeçalho JSON): ${(size / 1e6).toFixed(1)} MB -> ${(total / 1e6).toFixed(1)} MB (${(100 * total / size).toFixed(1)}%)`);
  out.encoded_body_bytes = total;
  return out;
}

const args = process.argv.slice(2), ji = args.indexOf('--json');
const jsonOut = ji >= 0 ? args.splice(ji, 2)[1] : null;
const results = [];
for (const p of args) results.push(await measure(p));
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(results, null, 1) + '\n');
console.log(`\n${checks} conferências, ${fails} falhas`);
process.exit(fails ? 1 : 0);
