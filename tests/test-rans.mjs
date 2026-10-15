// rANS do expoente no Node 18+.
// Sem argumentos: testes de unidade do rANS (bordas, distribuições, stream cortado ou com sobra).
// Com um .safetensors real, além disso, tensor por tensor (memória de um tensor por vez, serve para > 2 GB):
//   1. codifica o tensor com o core duas vezes, com rANS no expoente e só com deflate (o codec da v0.1);
//   2. decodifica o .strata com rANS na profundidade total e confere byte a byte com o original;
//   3. remonta o .safetensors inteiro em fluxo e confere o SHA-256 com o do arquivo original;
//   4. imprime bits/peso por estrato com cada codec e a entropia de ordem 0 do expoente.
// Uso:
//   node tests/test-rans.mjs
//   node tests/test-rans.mjs modelo.safetensors [--json saida.json]
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
function entropyBits(u8) {
  const c = new Float64Array(256); for (let i = 0; i < u8.length; i++) c[u8[i]]++;
  let h = 0; for (const v of c) if (v) h -= v * Math.log2(v / u8.length);
  return h;
}

/* ---------------- unidade ---------------- */
function unit() {
  console.log('=== rANS: testes de unidade ===');
  const rng = S.mulberry32(1234);
  const cases = [];
  cases.push(['vazio', new Uint8Array(0)]);
  cases.push(['um símbolo', new Uint8Array([200])]);
  cases.push(['constante', new Uint8Array(5000).fill(126)]);
  const u = new Uint8Array(300000); for (let i = 0; i < u.length; i++) u[i] = (rng() * 256) | 0;
  cases.push(['uniforme em 256', u]);
  const g = new Uint8Array(1000000);
  for (let i = 0; i < g.length; i++) g[i] = Math.max(0, Math.min(255, Math.round(120 + 2.5 * S.gauss(rng))));
  g[17] = 0; g[99] = 255; g[500] = 3; // símbolos raros com freq mínima
  cases.push(['expoente gaussiano + raros', g]);
  const b = new Uint8Array(200000); for (let i = 0; i < b.length; i++) b[i] = rng() < 0.9995 ? 1 : 2;
  cases.push(['muito enviesado', b]);
  const all = new Uint8Array(256 * 3); for (let i = 0; i < all.length; i++) all[i] = i & 255;
  cases.push(['256 símbolos, 3 de cada', all]);
  for (const [label, x] of cases) {
    const e = S.ransEncode(x), d = S.ransDecode(e, x.length);
    const h = entropyBits(x) / 8;
    const ok = check(same(d, x), `${label}: decodificado difere`);
    console.log(`${ok ? 'ok' : '--'} ${label}: ${x.length} símbolos -> ${e.length} B (entropia ${h.toFixed(1)} B)`);
    // determinismo: codificar de novo dá os mesmos bytes
    check(same(S.ransEncode(x), e), `${label}: codificar duas vezes deu bytes diferentes`);
  }
  // distância da entropia num caso grande: tabela + estado + perda de quantizar em 2^15
  const e = S.ransEncode(g), h = entropyBits(g) / 8;
  check(e.length < h * 1.001 + 64, `gaussiano: ${e.length} B passa de 0,1% acima da entropia (${h.toFixed(0)} B)`);
  // stream danificado: tem que falhar, não devolver lixo calado
  const bad = [['sem o último byte', e.subarray(0, e.length - 1)], ['um byte a mais', new Uint8Array([...e, 0])],
    ['tabela cortada', e.subarray(0, 5)], ['k > 256', new Uint8Array([1, 2, 0, 0, 0, 0, 0, 0])]];
  for (const [label, x] of bad) {
    let err = null; try { S.ransDecode(x, g.length); } catch (er) { err = er; }
    check(err && /rANS/.test(err.message), `${label}: devia falhar com erro de rANS`);
    console.log(`${err ? 'ok' : '--'} ${label}: ${err ? err.message : 'não falhou'}`);
  }
}

/* ---------------- modelo real ---------------- */
function readAt(fd, pos, len) {
  const out = new Uint8Array(len); let o = 0;
  while (o < len) { const r = fs.readSync(fd, out, o, Math.min(len - o, 1 << 30), pos + o); if (!r) throw new Error('arquivo curto'); o += r; }
  return out;
}

async function real(path) {
  console.log(`\n=== ${path} ===`);
  const fd = fs.openSync(path, 'r'), size = fs.fstatSync(fd).size;
  const N = Number(new DataView(readAt(fd, 0, 8).buffer).getBigUint64(0, true));
  const headerBytes = readAt(fd, 8, N);
  const parsed = S.parseSafetensorsHeader(new Uint8Array(8), headerBytes);
  const base = 8 + N;
  // SHA-256 do original, lido em blocos
  const h0 = createHash('sha256');
  for (let p = 0; p < size; p += 1 << 28) h0.update(readAt(fd, p, Math.min(1 << 28, size - p)));
  const sha0 = h0.digest('hex');
  // SHA-256 da volta: cabeçalho original + cada tensor decodificado, em ordem de offset
  const h1 = createHash('sha256');
  const len8 = new Uint8Array(8); new DataView(len8.buffer).setBigUint64(0, BigInt(N), true);
  h1.update(len8); h1.update(headerBytes);
  let cursor = 0;
  const M = 7, sums = { all: { n: 0, ans: new Float64Array(M + 1), dfl: new Float64Array(M + 1), H: 0, expAns: 0, expDfl: 0 } };
  sums.layers = { n: 0, ans: new Float64Array(M + 1), dfl: new Float64Array(M + 1), H: 0, expAns: 0, expDfl: 0 };
  const rows = [];
  const t0 = performance.now();
  for (const ent of parsed.entries) {
    if (ent.begin !== cursor) throw new Error('safetensors com buraco entre tensores; esta conferência supõe dados contíguos');
    const bytes = readAt(fd, base + ent.begin, ent.end - ent.begin);
    cursor = ent.end;
    const t = { name: ent.name, dtype: ent.dtype, shape: ent.shape, bytes };
    const encA = await S.encode([t]), encD = await S.encode([t], { rans: false });
    const dec = await S.decode(encA.file);
    const back = dec.tensors[0].bytes;
    check(same(back, bytes), `${ent.name}: volta com rANS difere do original`);
    h1.update(back);
    const it = encA.items[0];
    if (it.kind !== 'float' || it.L.m !== M) { console.log(`   ${ent.name}: ${ent.dtype}, fora da conta de bits/peso`); continue; }
    const by = (enc) => { const v = new Float64Array(M + 1); for (const e of enc.table) v[e.s] += e.l; return v; };
    const expOf = (enc) => enc.table.find((e) => e.p === 'exp');
    const a = by(encA), d = by(encD), ea = expOf(encA), ed = expOf(encD);
    const H = entropyBits(it.st.exp);
    const isLayer = ent.name.startsWith('model.layers.') && ent.shape.length === 2;
    for (const g of isLayer ? [sums.all, sums.layers] : [sums.all]) {
      g.n += it.n; g.H += H; g.expAns += 8 * ea.l; g.expDfl += 8 * ed.l;
      for (let k = 0; k <= M; k++) { g.ans[k] += 8 * a[k]; g.dfl[k] += 8 * d[k]; }
    }
    rows.push({ name: ent.name, n: it.n, exp_entropy: H / it.n, exp_rans: 8 * ea.l / it.n, exp_codec: ea.c, exp_deflate: 8 * ed.l / it.n });
  }
  fs.closeSync(fd);
  const sha1 = h1.digest('hex');
  check(sha0 === sha1, `SHA-256 da volta difere: ${sha0} vs ${sha1}`);
  console.log(`${parsed.entries.length} tensores em ${((performance.now() - t0) / 1000).toFixed(0)} s; ` +
    `SHA-256 original ${sha0.slice(0, 16)} volta ${sha1.slice(0, 16)} ${sha0 === sha1 ? 'IGUAL' : 'DIFERENTE'}`);
  const worst = rows.reduce((w, r) => (r.exp_rans - r.exp_entropy > w.exp_rans - w.exp_entropy ? r : w), rows[0]);
  console.log(`pior distância rANS - entropia num tensor: ${(worst.exp_rans - worst.exp_entropy).toFixed(4)} b/peso (${worst.name}, ${worst.n} pesos)`);
  const out = {};
  for (const [key, label] of [['all', 'todos os tensores BF16'], ['layers', 'matrizes de model.layers (as cortadas no ΔPPL)']]) {
    const g = sums[key]; if (!g.n) continue;
    const ca = [], cd = []; let x = 0, y = 0;
    for (let k = 0; k <= M; k++) { x += g.ans[k]; y += g.dfl[k]; ca.push(x / g.n); cd.push(y / g.n); }
    console.log(`\n${label}: ${g.n} pesos`);
    console.log(`expoente: entropia ${(g.H / g.n).toFixed(4)} | rANS ${(g.expAns / g.n).toFixed(4)} | deflate ${(g.expDfl / g.n).toFixed(4)} b/peso`);
    console.log('estrato | deflate (v0.1) b/peso | rANS (v0.2) b/peso | ganho');
    for (let k = 0; k <= M; k++) console.log(`${k} | ${cd[k].toFixed(4)} | ${ca[k].toFixed(4)} | ${(cd[k] - ca[k]).toFixed(4)}`);
    out[key] = { n: g.n, exp_entropy: g.H / g.n, exp_rans: g.expAns / g.n, exp_deflate: g.expDfl / g.n,
      bytes_rans: Array.from(g.ans, (v) => v / 8), bytes_deflate: Array.from(g.dfl, (v) => v / 8), cum_bpw_rans: ca, cum_bpw_deflate: cd };
  }
  return { model: path.split('/').pop(), sha256: sha0, roundtrip_sha_equal: sha0 === sha1, ...out, tensors: rows };
}

unit();
const args = process.argv.slice(2), ji = args.indexOf('--json');
const jsonOut = ji >= 0 ? args.splice(ji, 2)[1] : null;
const results = [];
for (const p of args) results.push(await real(p));
if (jsonOut) fs.writeFileSync(jsonOut, JSON.stringify(results, null, 1) + '\n');
console.log(`\n${checks} conferências, ${fails} falhas`);
process.exit(fails ? 1 : 0);
