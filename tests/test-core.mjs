// Teste do núcleo STRATA no Node: round-trip bit-exato, propriedade de prefixo, números para o doc.
import fs from 'node:fs';
const src = fs.readFileSync(new URL('../src/strata-core.js', import.meta.url), 'utf8');
new Function(src)();
const S = globalThis.Strata;
const MB = (b) => (b / 1e6).toFixed(3) + ' MB';

async function run(params) {
  console.log(`\n=== modelo sintético BF16, ${params / 1e6}M params ===`);
  const tensors = S.synthModel(params, 7);
  const { file: st, headerBytes } = S.buildSafetensors(tensors, { format: 'pt', origem: 'sintetico' });
  // caminho de leitura real: cabeçalho + tensores pelo offset
  const N = Number(new DataView(st.buffer).getBigUint64(0, true));
  const parsed = S.parseSafetensorsHeader(st.subarray(0, 8), st.subarray(8, 8 + N));
  const base = 8 + N;
  const read = parsed.entries.map((e) => ({ name: e.name, dtype: e.dtype, shape: e.shape, bytes: st.slice(base + e.begin, base + e.end) }));
  const t0 = performance.now();
  const enc = await S.encode(read, { sourceHeaderBytes: st.subarray(8, 8 + N) });
  const t1 = performance.now();
  const dec = await S.decode(enc.file);
  const rebuilt = S.rebuildSafetensors(dec.sourceHeaderBytes, dec.tensors);
  const t2 = performance.now();
  const h0 = await S.sha256Hex(st), h1 = await S.sha256Hex(rebuilt);
  console.log('safetensors', MB(st.length), '-> .strata', MB(enc.file.length), `(${(100 * enc.file.length / st.length).toFixed(1)}%)`,
    `enc ${(t1 - t0).toFixed(0)}ms dec ${(t2 - t1).toFixed(0)}ms`);
  console.log('SHA-256 igual:', h0 === h1, h0.slice(0, 16));
  // tamanhos por estrato
  const nf = enc.items.filter((x) => x.kind === 'float').reduce((a, x) => a + x.n, 0);
  const byS = new Array(enc.M + 1).fill(0);
  for (const e of enc.table) byS[e.s] += e.l;
  let cum = 0;
  const met = S.depthMetrics(enc.items, enc.M, 2e6);
  console.log('estrato | bytes | bits/param acumulado | prefixo arquivo | erro rel RMS | SNR dB');
  for (let d = 0; d <= enc.M; d++) {
    cum += byS[d];
    const rel = met.depth[d];
    console.log(d, byS[d], (8 * cum / nf).toFixed(3), `${(100 * enc.prefix[d] / enc.file.length).toFixed(1)}%`,
      (100 * rel).toFixed(3) + '%', rel ? (-20 * Math.log10(rel)).toFixed(1) : 'exato');
  }
  console.log('INT4 bloco32 (4.5 bpw) erro rel:', (100 * met.int4).toFixed(3) + '%', ' INT8 bloco32 (8.5 bpw):', (100 * met.int8).toFixed(3) + '%');
  // entropia do expoente e por bit
  const all = enc.items.filter((x) => x.kind === 'float');
  let eh = 0, en = 0; for (const it of all) { eh += S.byteEntropy(it.st.exp) * it.n; en += it.n; }
  console.log('entropia expoente (ordem 0, média ponderada):', (eh / en).toFixed(3), 'bits');
  const big = all.reduce((a, b) => (a.n > b.n ? a : b));
  console.log('entropia por bit (MSB→LSB) em', big.name, S.bitPositionEntropy(big.vals, big.L, 1).map((h) => h.toFixed(2)).join(' '));
  // propriedade de prefixo: decodificar só os primeiros bytes
  for (const d of [0, 1, 3]) {
    const cut = enc.file.subarray(0, enc.prefix[d]);
    const part = await S.decode(cut, d);
    const ok = part.tensors.every((t, i) => !enc.items[i].vals || t.vals.length === enc.items[i].n);
    console.log(`prefixo d=${d}: ${MB(cut.length)} lido, decodificou: ${ok}`);
  }
  // genealogia
  const child = S.synthChild(read, 7);
  const pItems = read.map(S.prep), cItems = child.tensors.map(S.prep);
  for (const d of [3, 7]) {
    const lin = await S.lineage(pItems, cItems, d);
    if (d === 3) {
      console.log('genealogia: filho isolado', MB(lin.standalone), 'como delta', MB(lin.delta), `(${(100 * lin.delta / lin.standalone).toFixed(1)}%)`);
      for (const r of lin.rows) console.log('  ', r.name.padEnd(42), r.mode.padEnd(6), String(r.bytes).padStart(8), 'mudou', r.changed !== undefined ? (100 * r.changed).toFixed(1) + '%' : '-',
        r.cosTrunc !== undefined ? `cos(trunc d=${d}) ${r.cosTrunc.toFixed(3)} cos(1bit) ${r.cosBit.toFixed(3)}` : '', r.perStratum ? r.perStratum.join(',') : '');
    } else {
      for (const r of lin.rows.filter((r) => r.mode === 'DELTA')) console.log(`   d=7 ${r.name} cos(trunc) ${r.cosTrunc.toFixed(3)}`);
    }
  }
  return enc;
}

async function edge() {
  console.log('\n=== bordas: FP16, FP32, FP8, I64, especiais ===');
  const f16 = new Uint16Array([0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0xfc00, 0x7e00, 0x1234, 0xabcd]);
  const f32 = new Uint32Array([0, 0x80000000, 1, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x3f800000, 0xbf9e0652, 0x00800000]);
  const bf = new Uint16Array([0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x3f80, 0x0001, 0x007f, 0x0080, 0xbeef]);
  const f8 = new Uint8Array(256); for (let i = 0; i < 256; i++) f8[i] = i;
  const i64 = new BigInt64Array([1n, -2n, 123456789012345n]);
  const tensors = [
    { name: 'a', dtype: 'F16', shape: [f16.length], bytes: new Uint8Array(f16.buffer) },
    { name: 'b', dtype: 'F32', shape: [f32.length], bytes: new Uint8Array(f32.buffer) },
    { name: 'c', dtype: 'BF16', shape: [bf.length], bytes: new Uint8Array(bf.buffer) },
    { name: 'd', dtype: 'F8_E4M3', shape: [256], bytes: f8.slice() },
    { name: 'e', dtype: 'F8_E5M2', shape: [256], bytes: f8.slice() },
    { name: 'f', dtype: 'I64', shape: [3], bytes: new Uint8Array(i64.buffer) },
  ];
  const { file, headerBytes } = S.buildSafetensors(tensors, null);
  const enc = await S.encode(tensors, { sourceHeaderBytes: headerBytes });
  const dec = await S.decode(enc.file);
  const rebuilt = S.rebuildSafetensors(dec.sourceHeaderBytes, dec.tensors);
  console.log('bordas round-trip SHA igual:', (await S.sha256Hex(file)) === (await S.sha256Hex(rebuilt)));
  for (const d of [0, 1, 2, 5]) { const p = await S.decode(enc.file.subarray(0, enc.prefix[d]), d); console.log('  prefixo', d, 'ok', p.tensors.length); }
  // f32ToF16 confere com tabela em amostra
  const toF = S.makeToFloat('F16'); let bad = 0;
  for (let h = 0; h < 65536; h++) { const x = toF(h); if (!isFinite(x)) continue; if (S.f32ToF16(x) !== h && !(x === 0)) bad++; }
  console.log('f32ToF16 inverso exato para todos FP16 finitos (exceto ±0):', bad === 0, bad);
  // truncMid em FP32 não gera NaN/inf a partir de finitos
  const toF32 = S.makeToFloat('F32'); let nanBad = 0;
  for (const u of f32) { const x = toF32(u); if (!isFinite(x)) continue; for (let d = 0; d <= 23; d++) if (!isFinite(toF32(S.truncMid(u, S.LAYOUTS.F32, d)))) nanBad++; }
  console.log('truncMid FP32 sem não-finitos:', nanBad === 0);
}

await run(1_000_000);
await run(4_000_000);
await edge();
