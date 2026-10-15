// Roundtrip do codec STRATA no Node 18+, sem navegador.
// Confere, para cada tensor e cada profundidade d = 0..M:
//   1. decode(encode(x), d) == truncMid(x, L, d), elemento a elemento (arquivo inteiro e só o prefixo d);
//   2. no topo, rebuildSafetensors devolve o .safetensors com SHA-256 idêntico (node:crypto, não o core);
//   3. decodeTensorAt lê com o prefixo mínimo e falha quando falta 1 byte;
//   4. decode de um arquivo cortado, pedindo mais estratos do que o corte tem, falha em vez de inventar bits.
// Uso:
//   node tests/test-roundtrip.mjs                    # modelos sintéticos + bordas de tipos
//   node tests/test-roundtrip.mjs modelo.safetensors # também um arquivo real (ex.: Llama-3.2-1B)
// Sai com código 1 se qualquer conferência falhar.
import fs from 'node:fs';
import { createHash } from 'node:crypto';

const src = fs.readFileSync(new URL('../src/strata-core.js', import.meta.url), 'utf8');
new Function(src)();
const S = globalThis.Strata;
if (!S.hasCodec) { console.error('CompressionStream ausente: use Node 18 ou mais novo'); process.exit(2); }

const sha = (u8) => createHash('sha256').update(u8).digest('hex');
let fails = 0, checks = 0;
function check(ok, msg) { checks++; if (!ok) { fails++; console.log('  FALHOU:', msg); } return ok; }
async function throws(fn) { try { await fn(); return null; } catch (e) { return e; } }

// Compara um tensor decodificado com truncMid aplicado ao original. Devolve o 1º índice divergente ou -1.
function firstMismatch(orig, got, d) {
  if (orig.kind !== 'float') {
    const a = orig.bytes, b = got.bytes;
    if (!b || a.length !== b.length) return 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return i;
    return -1;
  }
  const L = orig.L, dd = Math.min(d, L.m), v = got.vals;
  if (!v || v.length !== orig.n) return 0;
  for (let i = 0; i < orig.n; i++) if (v[i] !== S.truncMid(orig.vals[i] >>> 0, L, dd)) return i;
  return -1;
}

async function roundtrip(label, stFile) {
  console.log(`\n=== ${label} ===`);
  const t0 = performance.now();
  const src = S.readSafetensors(stFile);
  const enc = await S.encode(src.tensors, { sourceHeaderBytes: src.headerBytes });
  const t1 = performance.now();
  const items = enc.items, M = enc.M, base = 12 + enc.headerLen;
  const nFloat = items.filter((x) => x.kind === 'float').reduce((a, x) => a + x.n, 0);
  console.log(`${items.length} tensores, ${nFloat} pesos float, M=${M}; safetensors ${stFile.length} B -> .strata ${enc.file.length} B ` +
    `(${(100 * enc.file.length / stFile.length).toFixed(2)}%), encode ${(t1 - t0).toFixed(0)} ms`);

  // 1. decode == truncMid em todas as profundidades, arquivo inteiro e prefixo
  console.log('d | prefixo B | bits/peso no disco | decode(inteiro)==truncMid | decode(prefixo)==truncMid | ms');
  for (let d = 0; d <= M; d++) {
    const ta = performance.now();
    const full = await S.decode(enc.file, d);
    const cut = await S.decode(enc.file.subarray(0, enc.prefix[d]), d);
    let okFull = true, okCut = true;
    for (let k = 0; k < items.length; k++) {
      const a = firstMismatch(items[k], full.tensors[k], d), b = firstMismatch(items[k], cut.tensors[k], d);
      if (a >= 0) { okFull = false; check(false, `${items[k].name} d=${d} (inteiro) diverge no índice ${a}`); }
      if (b >= 0) { okCut = false; check(false, `${items[k].name} d=${d} (prefixo) diverge no índice ${b}`); }
      checks += 2;
    }
    const bpw = nFloat ? (8 * enc.prefix[d] / nFloat).toFixed(3) : '-';
    console.log(d, enc.prefix[d], bpw, okFull, okCut, (performance.now() - ta).toFixed(0));
  }

  // 2. topo: volta byte a byte para o .safetensors
  const top = await S.decode(enc.file);
  const rebuilt = S.rebuildSafetensors(top.sourceHeaderBytes, top.tensors);
  const h0 = sha(stFile), h1 = sha(rebuilt);
  check(rebuilt.length === stFile.length && h0 === h1, `SHA-256 diferente no topo: ${h0} vs ${h1}`);
  console.log('topo: SHA-256 original', h0.slice(0, 16), 'reconstruído', h1.slice(0, 16), h0 === h1 ? 'IGUAL' : 'DIFERENTE');

  // 3. decodeTensorAt: prefixo mínimo funciona; 1 byte a menos falha (limite simulado e arquivo cortado de verdade)
  let dtaOk = 0, dtaBad = 0;
  const sample = items.length > 12 ? items.filter((_, i) => i % Math.ceil(items.length / 12) === 0) : items;
  for (const it of sample) {
    const ti = items.indexOf(it);
    const depths = it.kind === 'float' ? [0, 1, Math.min(3, M), M] : [0];
    for (const d of depths) {
      const need = enc.table.filter((e) => e.t === ti && e.s <= d).reduce((a, e) => Math.max(a, base + e.o + e.l), base);
      const r = await throws(async () => {
        const got = await S.decodeTensorAt(enc.file, it.name, d, need);
        const m = firstMismatch(it, it.kind === 'float' ? got : { bytes: got.bytes }, d);
        if (m >= 0) throw new Error(`diverge no índice ${m}`);
      });
      if (check(!r, `decodeTensorAt(${it.name}, d=${d}, limite=${need}) devia funcionar: ${r && r.message}`)) dtaOk++; else dtaBad++;
      const e1 = await throws(() => S.decodeTensorAt(enc.file, it.name, d, need - 1));
      const e2 = await throws(() => S.decodeTensorAt(enc.file.subarray(0, need - 1), it.name, d));
      const good = (e) => e && /prefixo/.test(e.message);
      if (check(good(e1) && good(e2), `decodeTensorAt(${it.name}, d=${d}) com ${need - 1} B devia falhar com 'fora do prefixo': ` +
        `limite -> ${e1 ? e1.message : 'não falhou'}; arquivo cortado -> ${e2 ? e2.message : 'não falhou'}`)) dtaOk++; else dtaBad++;
    }
  }
  console.log(`decodeTensorAt: ${dtaOk} conferências certas, ${dtaBad} erradas (${sample.length} tensores amostrados)`);

  // 4. decode de prefixo pedindo mais estratos do que o corte contém
  let silent = 0;
  for (let d = 0; d < M; d++) {
    const e = await throws(() => S.decode(enc.file.subarray(0, enc.prefix[d]), d + 1));
    if (!e) silent++;
  }
  check(silent === 0, `decode(prefixo d, profundidade d+1) não falhou em ${silent} de ${M} cortes: devolve bits inventados sem aviso`);
  const e = await throws(() => S.decode(enc.file.subarray(0, enc.prefix[0])));
  check(!!e, 'decode(prefixo d=0) sem profundidade não falhou');
  console.log(`decode de prefixo pedindo estrato além do corte: ${silent ? 'NÃO falha em ' + silent + ' cortes' : 'falha corretamente'}`);
  return enc;
}

function edgeTensors() {
  const f16 = new Uint16Array([0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7e00, 0x1234, 0xabcd]);
  const f32 = new Uint32Array([0, 0x80000000, 1, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x3f800000, 0xbf9e0652, 0x00800000]);
  const bf = new Uint16Array([0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x3f80, 0x0001, 0x007f, 0x0080, 0xbeef, 0x3fab, 0xc0de]);
  const f8 = new Uint8Array(256); for (let i = 0; i < 256; i++) f8[i] = i;
  const i64 = new BigInt64Array([1n, -2n, 123456789012345n]);
  const rng = S.mulberry32(99), r32 = new Uint32Array(3000), r16 = new Uint16Array(3001);
  for (let i = 0; i < r32.length; i++) r32[i] = (rng() * 4294967296) >>> 0;
  for (let i = 0; i < r16.length; i++) r16[i] = (rng() * 65536) >>> 0;
  return [
    { name: 'f16', dtype: 'F16', shape: [f16.length], bytes: new Uint8Array(f16.buffer) },
    { name: 'f32', dtype: 'F32', shape: [f32.length], bytes: new Uint8Array(f32.buffer) },
    { name: 'bf16', dtype: 'BF16', shape: [bf.length], bytes: new Uint8Array(bf.buffer) },
    { name: 'e4m3', dtype: 'F8_E4M3', shape: [256], bytes: f8.slice() },
    { name: 'e5m2', dtype: 'F8_E5M2', shape: [256], bytes: f8.slice() },
    { name: 'i64', dtype: 'I64', shape: [3], bytes: new Uint8Array(i64.buffer) },
    { name: 'f32_aleatorio', dtype: 'F32', shape: [3000], bytes: new Uint8Array(r32.buffer) },
    { name: 'bf16_aleatorio_impar', dtype: 'BF16', shape: [3001], bytes: new Uint8Array(r16.buffer) },
  ];
}

const meta = { format: 'pt' };
await roundtrip('sintético BF16 1M (semente 7)', S.buildSafetensors(S.synthModel(1_000_000, 7), meta).file);
await roundtrip('sintético BF16 4M (semente 11)', S.buildSafetensors(S.synthModel(4_000_000, 11), meta).file);
await roundtrip('bordas: FP16, FP32, BF16, FP8, I64, bits aleatórios', S.buildSafetensors(edgeTensors(), null).file);
for (const p of process.argv.slice(2)) await roundtrip(p, new Uint8Array(fs.readFileSync(p)));

console.log(`\n${checks} conferências, ${fails} falhas`);
process.exit(fails ? 1 : 0);
