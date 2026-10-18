// Fuzzing do leitor .strata: arquivos válidos, mutados de propósito, entram no leitor e cada caso
// é classificado pelo que o leitor fez com ele.
//
//   recusa     o leitor recusou o arquivo com um erro dele (Error ou StrataError, ou JSON inválido)
//   intacto    leu e devolveu o .safetensors original, SHA-256 idêntico (a mutação não mudou nada)
//   silencioso leu SEM erro e devolveu pesos diferentes do original: o pior caso
//   quebra     estourou um TypeError/RangeError: o leitor usou um campo inválido sem conferir
//   recurso    tentou alocar ou descomprimir mais que o limite do teste (bomba de memória)
//
// Uso:
//   node tests/fuzz-leitor.mjs [--casos 4000] [--semente 1] [--core src/strata-core.js]
//                              [--rust caminho/do/strata-ref] [--salvar pasta] [--exigir]
//   --rust     roda o leitor de referência em Rust nos mesmos casos e compara as duas saídas
//   --exigir   passa {exigirIntegridade: true} ao leitor JS (só o leitor endurecido conhece a opção)
//   --sem-hash sementes gravadas sem nenhum hash (nem o do .safetensors original): mede só a validação
// Sai com código 1 se houver caso silencioso, quebra ou recurso.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import zlib from 'node:zlib';

const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i < 0 ? d : process.argv[i + 1]; };
const flag = (k) => process.argv.includes('--' + k);
const CASOS = +arg('casos', 4000), SEMENTE = +arg('semente', 1);
const corePath = path.resolve(arg('core', new URL('../src/strata-core.js', import.meta.url).pathname));
const RUST = arg('rust', null), SALVAR = arg('salvar', null), EXIGIR = flag('exigir'), SEM_HASH = flag('sem-hash');

/* ---- guarda de recursos: nenhuma alocação ou descompressão passa de LIMITE bytes ---- */
const LIMITE = 64 << 20;
class Guarda extends Error { constructor(m) { super(m); this.name = 'Guarda'; } }
for (const nome of ['Uint8Array', 'Uint16Array', 'Uint32Array', 'Float32Array', 'Float64Array']) {
  const T = globalThis[nome];
  globalThis[nome] = new Proxy(T, {
    construct(alvo, args, nt) {
      if (typeof args[0] === 'number' && args[0] * alvo.BYTES_PER_ELEMENT > LIMITE) throw new Guarda(`${nome}(${args[0]})`);
      return Reflect.construct(alvo, args, nt === globalThis[nome] ? alvo : nt);
    },
  });
}
const DS = globalThis.DecompressionStream;
globalThis.DecompressionStream = class {
  constructor(f) {
    const ds = new DS(f); let tot = 0;
    this.writable = ds.writable;
    this.readable = ds.readable.pipeThrough(new TransformStream({
      transform(ch, c) { tot += ch.byteLength; if (tot > LIMITE) c.error(new Guarda('inflate > ' + LIMITE)); else c.enqueue(ch); },
    }));
  }
};

new Function(fs.readFileSync(corePath, 'utf8'))();
const S = globalThis.Strata;
const sha = (u8) => createHash('sha256').update(u8).digest('hex');
const te = new TextEncoder(), td = new TextDecoder();

/* ---- sementes: arquivos válidos pequenos, com o .safetensors de origem ---- */
function tensor(name, dtype, shape, bytes) { return { name, dtype, shape, bytes }; }
function aleat(n, rng) { const u = new Uint8Array(n); for (let i = 0; i < n; i++) u[i] = (rng() * 256) | 0; return u; }
async function sementes() {
  const rng = S.mulberry32(99), out = [];
  const bf = (vals) => { const u = new Uint16Array(vals.length); vals.forEach((v, i) => { u[i] = S.f32ToBf16(v); }); return new Uint8Array(u.buffer); };
  const conjuntos = [
    ['exemplo da spec', [tensor('exemplo.weight', 'BF16', [2, 4], bf([1, -0.5, 0.199, 0, 3.14, -123.5, 0.00995, 0.000999])),
      tensor('exemplo.ids', 'I64', [2], new Uint8Array(new BigInt64Array([7n, 42n]).buffer))]],
    ['sintético BF16 30k', S.synthModel(30000, 7)],
    ['tipos mistos', [
      tensor('a.f16', 'F16', [64, 3], aleat(384, rng)), tensor('b.f32', 'F32', [50], aleat(200, rng)),
      tensor('c.e4m3', 'F8_E4M3', [33], aleat(33, rng)), tensor('d.e5m2', 'F8_E5M2', [17], aleat(17, rng)),
      tensor('e.u8', 'U8', [40], aleat(40, rng)), tensor('f.bool', 'BOOL', [9], new Uint8Array(9).fill(1)),
      tensor('g.vazio', 'BF16', [0], new Uint8Array(0)), tensor('h.bf16', 'BF16', [300], aleat(600, rng))]],
  ];
  for (const [nome, ts] of conjuntos) {
    const { file: st } = S.buildSafetensors(ts, { origem: 'fuzz' });
    const src = S.readSafetensors(st);
    const enc = await S.encode(src.tensors, { sourceHeaderBytes: src.headerBytes, sourceSha256: SEM_HASH ? undefined : sha(st), integridade: !SEM_HASH });
    out.push({ nome, st, file: enc.file, shaSt: sha(st) });
  }
  return out;
}

/* ---- mutações ---- */
function partes(file) {
  const hl = new DataView(file.buffer, file.byteOffset).getUint32(8, true);
  return { hl, header: JSON.parse(td.decode(file.subarray(12, 12 + hl))), corpo: file.subarray(12 + hl) };
}
function monta(header, corpo, magic) {
  const hb = te.encode(JSON.stringify(header)), f = new Uint8Array(12 + hb.length + corpo.length);
  f.set(magic || S.MAGIC, 0); new DataView(f.buffer).setUint32(8, hb.length, true); f.set(hb, 12); f.set(corpo, 12 + hb.length);
  return f;
}
const ESTRANHOS = [-1, 0, 1, 2, 7, 255, 65536, 2 ** 31, 2 ** 32, 2 ** 32 + 1, 2 ** 53, 1.5, -0.5, 1e308, 'x', '', null, true, [], {}, [1], '__proto__', 'constructor'];
const CHAVES_S = ['t', 's', 'p', 'o', 'l', 'r', 'c'], CHAVES_T = ['name', 'dtype', 'shape', 'n', 'bytes'];
const P_VALS = ['sign', 'exp', 'm1', 'm7', 'm8', 'm23', 'raw', '__proto__', 'constructor', 'toString'];
const C_VALS = ['raw', 'deflate', 'rans', 'gzip', ''];
const DTYPES = ['BF16', 'F16', 'F32', 'F8_E4M3', 'F8_E5M2', 'I64', 'U8', 'BOOL', 'XYZ', '__proto__'];

const BOMBA_DEFLATE = new Uint8Array(zlib.deflateRawSync(Buffer.alloc(128 << 20), { level: 9 })); // 128 MB de zeros em ~128 KB
function bombaRans() { // um só símbolo com freq 2^15: cada símbolo custa 0 bit, n é livre
  const b = new Uint8Array(2 + 3 + 4); b[0] = 1; b[2] = 0; b[3] = 0; b[4] = 0x80; // freq 32768 = 0x8000
  new DataView(b.buffer).setUint32(5, 1 << 23, true); return b;
}

async function muta(sem, rng) {
  const pick = (a) => a[(rng() * a.length) | 0], ri = (n) => (rng() * n) | 0;
  const f = sem.file, { header, corpo } = partes(f), base = 12 + (f.length - corpo.length - 12);
  const tipo = pick(['bit', 'bitCorpo', 'bitCorpo', 'corte', 'hl', 'campo', 'campo', 'campo', 'remove', 'duplica', 'troca',
    'bomba', 'fonte', 'dtype', 'forma', 'magic', 'lixo', 'rehash', 'rehash']);
  let g;
  switch (tipo) {
    case 'bit': case 'bitCorpo': {
      g = f.slice(); const ini = tipo === 'bit' ? 0 : base, k = 1 + ri(8);
      for (let j = 0; j < k; j++) { const i = ini + ri(g.length - ini); g[i] ^= 1 << ri(8); }
      break;
    }
    case 'rehash': { // quem escreve o arquivo também escreve os hashes: bits trocados num stream
      // comprimido, com o hash do stream refeito, chegam ao descompressor (o hash do tensor não muda)
      const h = structuredClone(header), cand = h.streams.filter((e) => e.c !== 'raw' && e.l > 0);
      if (!cand.length) { g = f.slice(); break; }
      const e = pick(cand), c2 = corpo.slice(), k = 1 + ri(4);
      for (let j = 0; j < k; j++) c2[e.o + ri(e.l)] ^= 1 << ri(8);
      if (e.h) e.h = sha(c2.subarray(e.o, e.o + e.l));
      g = monta(h, c2); break;
    }
    case 'corte': g = f.slice(0, ri(f.length)); break;
    case 'hl': g = f.slice(); new DataView(g.buffer).setUint32(8, pick([0, 1, 11, g.length, g.length - 12, 2 ** 31, 2 ** 32 - 1, ri(g.length * 2)]), true); break;
    case 'magic': g = f.slice(); g[7] = pick([0, 1, 3, 255]); if (rng() < 0.5) g[ri(7)] ^= 1; break;
    case 'lixo': { g = new Uint8Array(f.length + 1 + ri(64)); g.set(f); g.set(aleat(g.length - f.length, rng), f.length); break; }
    case 'campo': {
      const h = structuredClone(header);
      if (rng() < 0.7 && h.streams.length) {
        const e = pick(h.streams), k = pick(CHAVES_S);
        e[k] = k === 'p' && rng() < 0.6 ? pick(P_VALS) : k === 'c' && rng() < 0.6 ? pick(C_VALS) : pick(ESTRANHOS);
      } else if (rng() < 0.85) {
        const t = pick(h.tensors), k = pick(CHAVES_T); t[k] = pick(ESTRANHOS);
      } else {
        const k = pick(['format', 'estratos', 'tensors', 'streams', 'fonte', 'pai', 'ordem']); h[k] = pick(ESTRANHOS);
      }
      g = monta(h, corpo); break;
    }
    case 'remove': {
      const h = structuredClone(header);
      if (rng() < 0.5 && h.streams.length) h.streams.splice(ri(h.streams.length), 1);
      else if (rng() < 0.5 && h.streams.length) delete pick(h.streams)[pick(CHAVES_S)];
      else delete pick(h.tensors)[pick(CHAVES_T)];
      g = monta(h, corpo); break;
    }
    case 'duplica': {
      const h = structuredClone(header);
      if (rng() < 0.5) { const e = pick(h.streams); h.streams.splice(ri(h.streams.length), 0, structuredClone(e)); }
      else { const t = pick(h.tensors); h.tensors.push(structuredClone(t)); }
      g = monta(h, corpo); break;
    }
    case 'troca': { // troca offsets/tamanhos entre dois streams (o leitor ingênuo lê o stream errado)
      const h = structuredClone(header), a = pick(h.streams), b = pick(h.streams);
      for (const k of rng() < 0.5 ? ['o', 'l'] : ['o', 'l', 'r', 'c']) [a[k], b[k]] = [b[k], a[k]];
      g = monta(h, corpo); break;
    }
    case 'bomba': {
      const h = structuredClone(header), cand = h.streams.filter((e) => e.p === 'exp');
      const e = cand.length ? pick(cand) : pick(h.streams), t = h.tensors[e.t];
      const rans = rng() < 0.5, dado = rans ? bombaRans() : BOMBA_DEFLATE;
      const n = pick([2 ** 31, 2 ** 28, 1e9, 200 << 20]);
      if (rans) { e.c = 'rans'; e.p = 'exp'; } else e.c = 'deflate';
      e.r = n; if (t && rng() < 0.7) { t.n = n; t.shape = [n]; t.bytes = n * 2; }
      // o stream novo vai para o fim do corpo
      e.o = corpo.length; e.l = dado.length;
      const c2 = new Uint8Array(corpo.length + dado.length); c2.set(corpo); c2.set(dado, corpo.length);
      g = monta(h, c2); break;
    }
    case 'fonte': {
      const h = structuredClone(header); if (!h.fonte) { g = f.slice(); break; }
      const sh = JSON.parse(td.decode(Uint8Array.from(atob(h.fonte.header_b64), (c) => c.charCodeAt(0))));
      const nomes = Object.keys(sh).filter((k) => k !== '__metadata__'), nm = pick(nomes);
      const r = rng();
      if (r < 0.4) sh[nm].data_offsets = [pick(ESTRANHOS), pick(ESTRANHOS)];
      else if (r < 0.6) delete sh[nm];
      else if (r < 0.8) sh[nm].data_offsets = [sh[nm].data_offsets[0], sh[nm].data_offsets[1] + pick([-1, 1, 2 ** 30])];
      else h.fonte.header_b64 = pick(['', '!!!', 'AAAA', 123]);
      if (r < 0.8) h.fonte.header_b64 = btoa(JSON.stringify(sh));
      g = monta(h, corpo); break;
    }
    case 'dtype': { const h = structuredClone(header); pick(h.tensors).dtype = pick(DTYPES); g = monta(h, corpo); break; }
    case 'forma': {
      const h = structuredClone(header), t = pick(h.tensors);
      t.shape = pick([[], [0], [-1], [2 ** 32, 2 ** 32], [1.5], ['3'], [t.n ? t.n + 1 : 9], null]);
      if (rng() < 0.5) t.n = typeof t.shape?.[0] === 'number' ? t.shape[0] : t.n;
      g = monta(h, corpo); break;
    }
  }
  return { tipo, file: g };
}

/* ---- alvo JS: lê tudo e volta para .safetensors ---- */
const ERROS_LEITOR = new Set(['Error', 'StrataError', 'SyntaxError']);
async function lerJS(file) {
  const t0 = performance.now();
  try {
    let st;
    if (S.toSafetensors) st = await S.toSafetensors(file, EXIGIR ? { exigirIntegridade: true } : undefined);
    else { // leitor antigo: decode + volta, como os testes faziam
      const dec = await S.decode(file);
      if (!dec.sourceHeaderBytes) throw new Error('sem fonte');
      st = S.rebuildSafetensors(dec.sourceHeaderBytes, dec.tensors);
    }
    return { st, ms: performance.now() - t0 };
  } catch (e) { return { err: e, ms: performance.now() - t0 }; }
}
function classifica(r, shaOrig) {
  if (r.err) {
    if (r.err instanceof Guarda || r.err.name === 'Guarda' || /allocation failed|Invalid typed array length|Array buffer allocation/i.test(r.err.message)) return 'recurso';
    if (ERROS_LEITOR.has(r.err.name) && !(r.err instanceof TypeError) && !(r.err instanceof RangeError)) return 'recusa';
    return 'quebra';
  }
  return sha(r.st) === shaOrig ? 'intacto' : 'silencioso';
}

const rng = S.mulberry32(SEMENTE), sems = await sementes();
const CLASSES = ['recusa', 'intacto', 'silencioso', 'quebra', 'recurso'];
const cont = Object.fromEntries(CLASSES.map((c) => [c, 0])), porTipo = {}, exemplos = {};
const tmp = RUST || SALVAR ? fs.mkdtempSync(path.join(SALVAR || os.tmpdir(), 'fuzz-')) : null;
const casos = [];
const t0 = performance.now();
let msMax = 0;
for (let i = 0; i < CASOS; i++) {
  const sem = sems[i % sems.length], m = await muta(sem, rng);
  const r = await lerJS(m.file), cl = classifica(r, sem.shaSt);
  msMax = Math.max(msMax, r.ms);
  cont[cl]++;
  (porTipo[m.tipo] ||= Object.fromEntries(CLASSES.map((c) => [c, 0])))[cl]++;
  if (['silencioso', 'quebra', 'recurso'].includes(cl)) (exemplos[cl] ||= []).length < 4 && exemplos[cl].push(`${m.tipo}: ${r.err ? r.err.name + ': ' + r.err.message.slice(0, 90) : 'saída difere'}`);
  if (tmp) {
    const nome = `caso-${String(i).padStart(5, '0')}.strata`;
    fs.writeFileSync(path.join(tmp, nome), m.file);
    casos.push({ nome, js: cl, jsSha: r.st ? sha(r.st) : null, orig: sem.shaSt, tipo: m.tipo });
  }
}
const seg = (performance.now() - t0) / 1000;

console.log(`leitor: ${path.relative(process.cwd(), corePath)}${EXIGIR ? ' (exigirIntegridade)' : ''}; ${CASOS} casos, semente ${SEMENTE}, ${seg.toFixed(1)} s, pior caso ${msMax.toFixed(0)} ms`);
console.log('classe       casos   %');
for (const c of CLASSES) console.log(c.padEnd(12), String(cont[c]).padStart(5), (100 * cont[c] / CASOS).toFixed(1).padStart(6));
console.log('\npor mutação: ' + CLASSES.join(' / '));
for (const [k, v] of Object.entries(porTipo).sort()) console.log('  ' + k.padEnd(9), CLASSES.map((c) => v[c]).join(' / '));
for (const [k, v] of Object.entries(exemplos)) { console.log(`\nexemplos de ${k}:`); for (const x of v) console.log('  ' + x); }

let ruim = cont.silencioso + cont.quebra + cont.recurso;
if (RUST) {
  // o leitor em Rust lê a pasta inteira e imprime "nome classe sha" por caso
  const saida = execFileSync(RUST, ['lote', tmp], { maxBuffer: 1 << 28, encoding: 'utf8' });
  const rs = new Map(saida.trim().split('\n').map((l) => { const [n, c, s] = l.split(' '); return [n, { c, s }]; }));
  const rc = Object.fromEntries(CLASSES.map((c) => [c, 0])), diverg = [];
  for (const k of casos) {
    const r = rs.get(k.nome); if (!r) { diverg.push(k.nome + ' sem resposta'); continue; }
    const cl = r.c === 'ok' ? (r.s === k.orig ? 'intacto' : 'silencioso') : r.c;
    rc[cl] = (rc[cl] || 0) + 1;
    const jsAceita = k.jsSha !== null, rsAceita = r.c === 'ok';
    if (jsAceita !== rsAceita || (jsAceita && k.jsSha !== r.s)) diverg.push(`${k.nome} (${k.tipo}): js=${k.js} rust=${cl}`);
  }
  console.log('\nleitor Rust nos mesmos casos:');
  for (const c of CLASSES) console.log(c.padEnd(12), String(rc[c] || 0).padStart(5), (100 * (rc[c] || 0) / CASOS).toFixed(1).padStart(6));
  console.log(`divergências JS × Rust (aceita/recusa ou saída diferente): ${diverg.length}`);
  for (const d of diverg.slice(0, 10)) console.log('  ' + d);
  ruim += (rc.silencioso || 0) + (rc.quebra || 0) + (rc.recurso || 0) + diverg.length;
}
if (tmp && !SALVAR) fs.rmSync(tmp, { recursive: true, force: true });
process.exit(ruim ? 1 : 0);
