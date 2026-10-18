// Modelos em vários .safetensors (shards): dois desenhos de contêiner sobre os MESMOS streams do core v0.2.
//   D1 "por-shard":   um .strata/0.2 por shard original, byte a byte o que o conversor de um arquivo grava,
//                     mais um índice (como o model.safetensors.index.json) com o prefixo de cada estrato.
//   D2 "por-estrato": um arquivo por estrato com todos os tensores do modelo (dividido em partes de até
//                     --cap-gb), mais um índice com a tabela de streams e os cabeçalhos originais dos shards.
// Os dois voltam bit a bit para cada shard original e para o model.safetensors.index.json.
//
// Uso:
//   node tests/test-shards.mjs                                     testes com um modelo sintético em 3 shards
//   node tests/test-shards.mjs encode <pasta> <trabalho> [shard…] [--cru-abaixo 65536]
//                     codifica cada shard em peças por estrato (um processo por shard pode rodar em paralelo);
//                     tensores com menos elementos que --cru-abaixo vão crus num stream só (0 desliga)
//   node tests/test-shards.mjs d1 <pasta> <trabalho> <saída>        monta o desenho 1
//   node tests/test-shards.mjs d2 <pasta> <trabalho> <saída> [--cap-gb 5]
//   node tests/test-shards.mjs verify <saída> [--depth 2] [--json r.json]
//                     volta bit a bit (SHA-256 de cada shard), tempo de leitura numa profundidade e
//                     quantas requisições HTTP Range cada leitura precisa
// A memória é a de um tensor por vez; o disco de trabalho guarda as peças (≈ tamanho final de um desenho).
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { createHash } from 'node:crypto';

const src = fs.readFileSync(new URL('../src/strata-core.js', import.meta.url), 'utf8');
new Function(src)();
const S = globalThis.Strata;
if (!S.hasCodec) { console.error('CompressionStream ausente: use Node 18 ou mais novo'); process.exit(2); }

const te = new TextEncoder(), td = new TextDecoder();
const INDEX = 'model.strata.index.json';
// preâmbulo de um arquivo de estrato do D2: "STRATAE" 0x01, estrato (uint32), parte (uint32)
const MAGIC_E = new Uint8Array([0x53, 0x54, 0x52, 0x41, 0x54, 0x41, 0x45, 0x01]);
const PRE_E = 16;
// tabela de streams binária do D2 (B.9): "STRATAT" 0x01, número de registros (uint32), tamanho do registro (uint32)
const MAGIC_T = new Uint8Array([0x53, 0x54, 0x52, 0x41, 0x54, 0x41, 0x54, 0x01]);
const TABELA = 'model.strata.streams';
const REG = 36, REG_H = 68; // sem e com o hash do stream
const LIMITE_INDICE = 100 * 1024 * 1024; // o mesmo teto do cabeçalho de um .strata (A.12)
const CRU_ABAIXO = 65536;

/* ---------------- E/S ---------------- */
function readAt(fd, pos, len) {
  const out = new Uint8Array(len); let o = 0;
  while (o < len) { const r = fs.readSync(fd, out, o, Math.min(len - o, 1 << 30), pos + o); if (!r) throw new Error('arquivo curto'); o += r; }
  return out;
}
function writeAll(fd, u8) { let o = 0; while (o < u8.length) o += fs.writeSync(fd, u8, o, u8.length - o); }
const COPY = Buffer.allocUnsafe(64 << 20);
function copyRange(fdIn, pos, len, fdOut, hash) {
  while (len > 0) {
    const n = fs.readSync(fdIn, COPY, 0, Math.min(len, COPY.length), pos);
    if (!n) throw new Error('peça curta');
    const v = COPY.subarray(0, n); writeAll(fdOut, v); if (hash) hash.update(v);
    pos += n; len -= n;
  }
}
function sha256File(p) {
  const fd = fs.openSync(p, 'r'), h = createHash('sha256'), size = fs.fstatSync(fd).size;
  for (let o = 0; o < size; o += COPY.length) { const n = fs.readSync(fd, COPY, 0, COPY.length, o); h.update(COPY.subarray(0, n)); }
  fs.closeSync(fd); return h.digest('hex');
}
const b64 = (u8) => Buffer.from(u8).toString('base64');
const unb64 = (s) => new Uint8Array(Buffer.from(s, 'base64'));
const shardList = (dir) => {
  const ix = path.join(dir, 'model.safetensors.index.json');
  if (fs.existsSync(ix)) return [...new Set(Object.values(JSON.parse(fs.readFileSync(ix, 'utf8')).weight_map))].sort();
  return fs.readdirSync(dir).filter((f) => f.endsWith('.safetensors')).sort();
};
const rssGB = () => process.resourceUsage().maxRSS / 1048576;

/* ---------------- fase 1: shard -> peças por estrato ---------------- */
// trabalho/<shard>/s<k>.bin guarda os streams codificados do estrato k, na ordem dos tensores;
// trabalho/<shard>/meta.json guarda tensores, streams (offset dentro da peça), SHA-256 e cabeçalho original.
async function encodeShard(dir, work, name, cruAbaixo = CRU_ABAIXO) {
  const t0 = performance.now(), file = path.join(dir, name);
  const fd = fs.openSync(file, 'r'), size = fs.fstatSync(fd).size;
  const first8 = readAt(fd, 0, 8), N = Number(new DataView(first8.buffer).getBigUint64(0, true));
  const headerBytes = readAt(fd, 8, N);
  const parsed = S.parseSafetensorsHeader(first8, headerBytes);
  const out = path.join(work, name); fs.mkdirSync(out, { recursive: true });
  const h = createHash('sha256'); h.update(first8); h.update(headerBytes);
  let cursor = 0, contiguous = true;
  // teto pelos tipos; o M de verdade sai dos streams (um F32 pequeno guardado cru não abre os estratos 8 a 23)
  const Mtipos = Math.max(0, ...parsed.entries.map((e) => (S.LAYOUTS[e.dtype] ? S.LAYOUTS[e.dtype].m : 0)));
  const fds = [], pos = new Array(Mtipos + 1).fill(0);
  for (let k = 0; k <= Mtipos; k++) fds.push(fs.openSync(path.join(out, `s${k}.bin`), 'w'));
  const tensors = [], streams = [];
  let tpl = null; // cabeçalho do core de um tensor: dá a ordem das chaves e os campos de topo (integridade)
  for (const ent of parsed.entries) {
    if (ent.begin !== cursor) contiguous = false;
    const bytes = readAt(fd, 8 + N + ent.begin, ent.end - ent.begin);
    if (contiguous) h.update(bytes);
    cursor = ent.end;
    const enc = await S.encode([{ name: ent.name, dtype: ent.dtype, shape: ent.shape, bytes }], { cruAbaixo });
    const ti = tensors.length;
    if (!tpl) tpl = enc.header;
    tensors.push(enc.header.tensors[0]);
    enc.streams.forEach((st, i) => {
      writeAll(fds[st.s], st.enc);
      // as chaves da tabela do core (t, s, p, o, l, r, c e o hash h, A.13) na mesma ordem; só t e o mudam
      streams.push({ ...enc.table[i], t: ti, o: pos[st.s] });
      pos[st.s] += st.enc.length;
    });
  }
  fds.forEach((f) => fs.closeSync(f));
  const M = Math.max(0, ...streams.map((e) => e.s));
  for (let k = M + 1; k <= Mtipos; k++) fs.unlinkSync(path.join(out, `s${k}.bin`));
  pos.length = M + 1;
  if (cursor + 8 + N !== size) contiguous = false;
  fs.closeSync(fd);
  const sha256 = contiguous ? h.digest('hex') : sha256File(file);
  const topo = tpl ? Object.fromEntries(Object.entries(tpl).filter(([k]) => !['tensors', 'streams', 'fonte', 'estratos'].includes(k))) : {};
  const meta = { arquivo: name, bytes: size, sha256, header_b64: b64(headerBytes), M, topo, chaves: tpl ? Object.keys(tpl) : null, tensors, streams, pecas: pos,
    tempo_s: (performance.now() - t0) / 1000, rss_gb: rssGB() };
  fs.writeFileSync(path.join(out, 'meta.json'), JSON.stringify(meta));
  return meta;
}

function loadMetas(dir, work) {
  return shardList(dir).map((n) => JSON.parse(fs.readFileSync(path.join(work, n, 'meta.json'), 'utf8')));
}
function sourceIndex(dir) {
  const ix = path.join(dir, 'model.safetensors.index.json');
  if (!fs.existsSync(ix)) return null;
  const bytes = fs.readFileSync(ix);
  return { nome: 'model.safetensors.index.json', sha256: createHash('sha256').update(bytes).digest('hex'), index_b64: b64(bytes),
    metadata: JSON.parse(bytes.toString('utf8')).metadata || null };
}
const byST = (a, b) => a.s - b.s || a.t - b.t; // Array.sort é estável: dentro de (s, t) fica a ordem do core (sign, exp)

/* ---------------- desenho 1: um .strata por shard ---------------- */
// O cabeçalho repete campo a campo o que S.encode grava, então o arquivo é idêntico ao do conversor de um arquivo.
function buildD1(dir, work, out) {
  const t0 = performance.now(), metas = loadMetas(dir, work), src = sourceIndex(dir);
  fs.mkdirSync(out, { recursive: true });
  const arquivos = [], weight_map = {};
  for (const m of metas) {
    const base = [0]; for (let k = 0; k < m.M; k++) base.push(base[k] + m.pecas[k]);
    const table = m.streams.slice().sort(byST).map((e) => ({ ...e, o: base[e.s] + e.o }));
    const campos = { ...m.topo, estratos: m.M + 1, tensors: m.tensors, streams: table,
      fonte: { tipo: 'safetensors', header_b64: m.header_b64, sha256: m.sha256 } };
    const header = {}; for (const k of m.chaves || Object.keys(campos)) header[k] = campos[k];
    const hb = te.encode(JSON.stringify(header));
    const nome = m.arquivo.replace(/\.safetensors$/, '') + '.strata';
    const fd = fs.openSync(path.join(out, nome), 'w'), hash = createHash('sha256');
    const pre = new Uint8Array(12); pre.set(S.MAGIC); new DataView(pre.buffer).setUint32(8, hb.length, true);
    writeAll(fd, pre); hash.update(pre); writeAll(fd, hb); hash.update(hb);
    const id = createHash('sha256').update(pre).update(hb).digest('hex'); // id do arquivo (A.13)
    for (let k = 0; k <= m.M; k++) {
      const pf = fs.openSync(path.join(work, m.arquivo, `s${k}.bin`), 'r');
      copyRange(pf, 0, m.pecas[k], fd, hash); fs.closeSync(pf);
    }
    fs.closeSync(fd);
    const prefixos = []; let acc = 12 + hb.length;
    for (let k = 0; k <= m.M; k++) { acc += m.pecas[k]; prefixos.push(acc); }
    arquivos.push({ nome, fonte: m.arquivo, fonte_bytes: m.bytes, fonte_sha256: m.sha256, bytes: acc, id, sha256: hash.digest('hex'),
      cabecalho: 12 + hb.length, estratos: m.M + 1, prefixos });
    for (const t of m.tensors) weight_map[t.name] = nome;
  }
  const index = { format: 'strata-shards/0.1', desenho: 'por-shard', metadata: src ? src.metadata : null, weight_map, arquivos,
    fonte: src ? { tipo: 'safetensors-index', nome: src.nome, sha256: src.sha256, index_b64: src.index_b64 } : null };
  fs.writeFileSync(path.join(out, INDEX), JSON.stringify(index));
  return { tempo_s: (performance.now() - t0) / 1000, arquivos: arquivos.length };
}

/* ---------------- desenho 2: arquivos por estrato ---------------- */
function buildD2(dir, work, out, cap) {
  const t0 = performance.now(), metas = loadMetas(dir, work), src = sourceIndex(dir);
  fs.mkdirSync(out, { recursive: true });
  const M = Math.max(...metas.map((m) => m.M));
  const tensors = [], all = [];
  metas.forEach((m, si) => {
    const off = tensors.length;
    for (const t of m.tensors) tensors.push({ ...t, shard: si });
    for (const e of m.streams) all.push({ ...e, t: e.t + off, si, po: e.o, f: 0 });
  });
  all.sort(byST);
  // plano das partes: um stream nunca é cortado; parte nova quando passaria do teto
  const partes = [];
  for (let k = 0; k <= M; k++) {
    let cur = null;
    for (const e of all) {
      if (e.s !== k) continue;
      if (!cur || (cur.bytes + e.l > cap && cur.bytes > PRE_E)) { cur = { s: k, bytes: PRE_E, streams: [] }; partes.push(cur); }
      e.o = cur.bytes; cur.streams.push(e); cur.bytes += e.l;
    }
  }
  const perS = {}; for (const p of partes) perS[p.s] = (perS[p.s] || 0) + 1;
  const seen = {}, arquivos = [];
  const pieceFd = new Map();
  const pf = (si, k) => {
    const key = si + ':' + k;
    if (!pieceFd.has(key)) pieceFd.set(key, fs.openSync(path.join(work, metas[si].arquivo, `s${k}.bin`), 'r'));
    return pieceFd.get(key);
  };
  for (const p of partes) {
    const j = seen[p.s] = (seen[p.s] || 0) + 1;
    const nome = `model-e${String(p.s).padStart(2, '0')}-${String(j).padStart(5, '0')}-of-${String(perS[p.s]).padStart(5, '0')}.strata`;
    const fd = fs.openSync(path.join(out, nome), 'w'), hash = createHash('sha256');
    const pre = new Uint8Array(PRE_E); pre.set(MAGIC_E);
    const dv = new DataView(pre.buffer); dv.setUint32(8, p.s, true); dv.setUint32(12, j - 1, true);
    writeAll(fd, pre); hash.update(pre);
    for (const e of p.streams) { copyRange(pf(e.si, e.s), e.po, e.l, fd, hash); e.f = arquivos.length; }
    fs.closeSync(fd);
    arquivos.push({ nome, s: p.s, parte: j - 1, bytes: p.bytes, sha256: hash.digest('hex') });
  }
  for (const f of pieceFd.values()) fs.closeSync(f);
  const integ = metas.every((m) => m.topo && m.topo.integridade);
  const tab = tabelaBin(all, integ);
  fs.writeFileSync(path.join(out, TABELA), tab);
  const index = { format: 'strata-shards/0.1', desenho: 'por-estrato', ordem: 'estrato-maior', estratos: M + 1, tensors, arquivos,
    tabela: { nome: TABELA, registros: all.length, bytes: tab.length, sha256: createHash('sha256').update(tab).digest('hex') },
    fonte: { tipo: 'safetensors-index', nome: src ? src.nome : null, sha256: src ? src.sha256 : null, index_b64: src ? src.index_b64 : null,
      shards: metas.map((m) => ({ nome: m.arquivo, bytes: m.bytes, sha256: m.sha256, header_b64: m.header_b64 })) },
    pai: null };
  if (integ) index.integridade = { alg: 'sha256' };
  fs.writeFileSync(path.join(out, INDEX), JSON.stringify(index));
  return { tempo_s: (performance.now() - t0) / 1000, arquivos: arquivos.length };
}

/* ---------------- tabela de streams binária (D2) ---------------- */
// Registro little-endian, em ordem (estrato, tensor) como no corpo de um .strata:
//   0 t uint32 | 4 s uint8 | 5 parte uint8 (0 raw, 1 sign, 2 exp, 2+j = mj) | 6 codificação uint8 (0 raw, 1 deflate, 2 rans)
//   7 zero | 8 f uint32 (arquivo) | 12 o uint64 | 20 l uint64 | 28 r uint64 | 36 h (32 bytes, só com integridade)
const PARTE = (p) => (p === 'raw' ? 0 : p === 'sign' ? 1 : p === 'exp' ? 2 : 2 + Number(p.slice(1)));
const PARTE_NOME = (k) => (k === 0 ? 'raw' : k === 1 ? 'sign' : k === 2 ? 'exp' : 'm' + (k - 2));
const COD = ['raw', 'deflate', 'rans'];
function tabelaBin(streams, comHash) {
  const rec = comHash ? REG_H : REG, buf = Buffer.alloc(16 + rec * streams.length);
  buf.set(MAGIC_T, 0); buf.writeUInt32LE(streams.length, 8); buf.writeUInt32LE(rec, 12);
  streams.forEach((e, i) => {
    const o = 16 + i * rec;
    buf.writeUInt32LE(e.t, o); buf[o + 4] = e.s; buf[o + 5] = PARTE(e.p); buf[o + 6] = COD.indexOf(e.c);
    buf.writeUInt32LE(e.f, o + 8);
    buf.writeBigUInt64LE(BigInt(e.o), o + 12); buf.writeBigUInt64LE(BigInt(e.l), o + 20); buf.writeBigUInt64LE(BigInt(e.r), o + 28);
    if (comHash) Buffer.from(e.h, 'hex').copy(buf, o + 36);
  });
  return buf;
}
// Lê e valida a tabela contra o índice: o SHA-256 antes de tudo, depois cada registro (tensor, arquivo, parte
// coerente com o tipo do tensor, estrato do arquivo, faixa dentro do arquivo, ordem estritamente crescente).
function lerTabela(out, ix) {
  const tb = ix.tabela, buf = fs.readFileSync(path.join(out, tb.nome));
  if (buf.length !== tb.bytes || createHash('sha256').update(buf).digest('hex') !== tb.sha256) throw new Error('tabela de streams não confere com o índice');
  for (let i = 0; i < 8; i++) if (buf[i] !== MAGIC_T[i]) throw new Error('tabela de streams: assinatura inválida');
  const n = buf.readUInt32LE(8), rec = buf.readUInt32LE(12), comHash = !!ix.integridade;
  if (rec !== (comHash ? REG_H : REG) || n !== tb.registros || buf.length !== 16 + n * rec) throw new Error('tabela de streams: tamanho inválido');
  const out2 = [], nt = ix.tensors.length;
  let ant = -1;
  for (let i = 0; i < n; i++) {
    const o = 16 + i * rec, onde = 'tabela, registro ' + i;
    const t = buf.readUInt32LE(o), s = buf[o + 4], pk = buf[o + 5], c = COD[buf[o + 6]], f = buf.readUInt32LE(o + 8);
    const off = Number(buf.readBigUInt64LE(o + 12)), l = Number(buf.readBigUInt64LE(o + 20)), r = Number(buf.readBigUInt64LE(o + 28));
    if (t >= nt || f >= ix.arquivos.length || !c || buf[o + 7] !== 0) throw new Error(onde + ': tensor, arquivo ou codificação inválidos');
    const T = ix.tensors[t], L = T.n !== null ? S.LAYOUTS[T.dtype] : null, p = PARTE_NOME(pk);
    const okParte = L ? (pk === 1 || pk === 2 ? s === 0 : pk >= 3 && pk - 2 <= L.m && s === pk - 2) : pk === 0 && s === 0;
    if (!okParte || ix.arquivos[f].s !== s) throw new Error(onde + ': parte, estrato ou arquivo incoerentes');
    if (c === 'rans' && p !== 'exp') throw new Error(onde + ': rANS fora do expoente');
    if (off < PRE_E || !Number.isSafeInteger(off + l) || off + l > ix.arquivos[f].bytes) throw new Error(onde + ': faixa fora do arquivo');
    const chave = ((s * nt + t) * 2 + (pk === 2 ? 1 : 0));
    if (chave <= ant) throw new Error(onde + ': fora de ordem ou repetido');
    ant = chave;
    const e = { t, s, p, o: off, l, r, c, f };
    if (comHash) e.h = buf.subarray(o + 36, o + 68).toString('hex');
    out2.push(e);
  }
  return out2;
}

/* ---------------- leitura comum aos dois desenhos ---------------- */
// Visão única: lista de shards de origem, cada um com seus tensores e, por tensor, os streams com (arquivo, offset absoluto).
function openModel(out) {
  if (fs.statSync(path.join(out, INDEX)).size > LIMITE_INDICE) throw new Error('índice maior que 100 MB');
  const ix = JSON.parse(fs.readFileSync(path.join(out, INDEX), 'utf8'));
  const files = [], shards = [];
  if (ix.desenho === 'por-shard') {
    for (const a of ix.arquivos) {
      const fd = fs.openSync(path.join(out, a.nome), 'r');
      const pre = readAt(fd, 0, 12);
      for (let i = 0; i < 7; i++) if (pre[i] !== S.MAGIC[i]) throw new Error(a.nome + ': assinatura .strata inválida');
      const hl = new DataView(pre.buffer).getUint32(8, true), hb = readAt(fd, 12, hl);
      if (a.id && createHash('sha256').update(pre).update(hb).digest('hex') !== a.id) throw new Error(a.nome + ': id do arquivo não confere com o índice');
      const header = JSON.parse(td.decode(hb));
      const fi = files.push({ nome: a.nome, fd, bytes: a.bytes, sha256: a.sha256, head: 12 + hl }) - 1;
      const per = header.tensors.map(() => []);
      for (const e of header.streams) per[e.t].push({ ...e, f: fi, abs: 12 + hl + e.o });
      shards.push({ nome: a.fonte, bytes: a.fonte_bytes, sha256: a.fonte_sha256, headerBytes: unb64(header.fonte.header_b64),
        tensors: header.tensors.map((t, i) => ({ ...t, streams: per[i] })) });
    }
  } else {
    for (const a of ix.arquivos) {
      const fd = fs.openSync(path.join(out, a.nome), 'r'), pre = readAt(fd, 0, PRE_E);
      for (let i = 0; i < 8; i++) if (pre[i] !== MAGIC_E[i]) throw new Error(a.nome + ': assinatura de estrato inválida');
      const dv = new DataView(pre.buffer);
      if (dv.getUint32(8, true) !== a.s || dv.getUint32(12, true) !== a.parte) throw new Error(a.nome + ': estrato/parte não batem com o índice');
      files.push({ nome: a.nome, fd, bytes: a.bytes, sha256: a.sha256 });
    }
    for (const s of ix.fonte.shards) shards.push({ nome: s.nome, bytes: s.bytes, sha256: s.sha256, headerBytes: unb64(s.header_b64), tensors: [] });
    const per = ix.tensors.map(() => []);
    for (const e of lerTabela(out, ix)) per[e.t].push({ ...e, abs: e.o });
    ix.tensors.forEach((t, i) => shards[t.shard].tensors.push({ ...t, streams: per[i] }));
  }
  return { ix, files, shards, close: () => files.forEach((f) => fs.closeSync(f.fd)) };
}

async function decodeStream(e, bytes) {
  if (e.c === 'raw') return bytes;
  if (e.c === 'deflate') return S.inflate(bytes);
  if (e.c === 'rans') return S.ransDecode(bytes, e.r);
  throw new Error('codificação desconhecida: ' + e.c);
}
async function decodeTensor(model, t, depth, counter) {
  const parts = {};
  for (const e of t.streams) {
    if (e.s > depth) continue;
    const b = readAt(model.files[e.f].fd, e.abs, e.l);
    if (counter) counter.bytes += e.l;
    // hash do stream (A.13) antes de descomprimir
    if (e.h && createHash('sha256').update(b).digest('hex') !== e.h) throw new Error(`hash do stream não confere: ${t.name} ${e.p}`);
    parts[e.p] = await decodeStream(e, b);
  }
  const L = S.LAYOUTS[t.dtype];
  let out;
  if (!L || !parts.exp) out = parts.raw;
  else {
    const d = Math.min(depth, L.m), st = { sign: parts.sign, exp: parts.exp, planes: [] };
    for (let j = 0; j < d; j++) st.planes.push(parts['m' + (j + 1)]);
    const vals = S.joinStrata(st, L, t.n, d, d < L.m);
    out = new Uint8Array(vals.buffer, vals.byteOffset, vals.byteLength);
    if (d < L.m) return out;
  }
  if (t.sha256 && createHash('sha256').update(out).digest('hex') !== t.sha256) throw new Error('hash do tensor não confere: ' + t.name);
  return out;
}

// Volta bit a bit: remonta cada shard em fluxo (cabeçalho original + tensores na ordem de data_offsets) e confere o SHA-256.
async function roundtrip(model) {
  const res = [];
  for (const sh of model.shards) {
    const t0 = performance.now(), h = createHash('sha256');
    const len8 = new Uint8Array(8); new DataView(len8.buffer).setBigUint64(0, BigInt(sh.headerBytes.length), true);
    h.update(len8); h.update(sh.headerBytes);
    const H = JSON.parse(td.decode(sh.headerBytes)), byName = new Map(sh.tensors.map((t) => [t.name, t]));
    const order = Object.entries(H).filter(([k]) => k !== '__metadata__').sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);
    let cursor = 0, total = 8 + sh.headerBytes.length;
    for (const [name, v] of order) {
      if (v.data_offsets[0] > cursor) { h.update(new Uint8Array(v.data_offsets[0] - cursor)); total += v.data_offsets[0] - cursor; }
      const b = await decodeTensor(model, byName.get(name), Infinity);
      h.update(b); total += b.length; cursor = v.data_offsets[1];
    }
    const sha = h.digest('hex');
    res.push({ shard: sh.nome, bytes: total, sha256: sha, igual: sha === sh.sha256 && total === sh.bytes, tempo_s: (performance.now() - t0) / 1000 });
  }
  if (model.ix.fonte && model.ix.fonte.index_b64) {
    const sha = createHash('sha256').update(unb64(model.ix.fonte.index_b64)).digest('hex');
    res.push({ shard: model.ix.fonte.nome, sha256: sha, igual: sha === model.ix.fonte.sha256 });
  }
  return res;
}

// Requisições HTTP Range para um conjunto de streams: faixas vizinhas no mesmo arquivo viram uma só.
// No D1 o leitor precisa também do cabeçalho de cada arquivo tocado (a tabela de streams mora nele);
// no D2 a tabela está no índice e o preâmbulo de 16 bytes não precisa ser lido.
function ranges(model, pick) {
  const per = new Map();
  for (const sh of model.shards) for (const t of sh.tensors) for (const e of t.streams) if (pick(t, e)) {
    if (!per.has(e.f)) per.set(e.f, model.files[e.f].head ? [[0, model.files[e.f].head]] : []);
    per.get(e.f).push([e.abs, e.abs + e.l]);
  }
  let req = 0, bytes = 0;
  for (const rs of per.values()) {
    rs.sort((a, b) => a[0] - b[0]);
    let end = -1;
    for (const [a, b] of rs) { if (a !== end) req++; end = b; bytes += b - a; }
  }
  return { arquivos: per.size, requisicoes: req, bytes };
}

async function verify(out, depth) {
  const model = openModel(out), ix = model.ix, r = { desenho: ix.desenho, arquivos: model.files.length };
  r.bytes_total = model.files.reduce((a, f) => a + f.bytes, 0);
  r.indice_bytes = fs.statSync(path.join(out, INDEX)).size;
  r.indice_sha256 = sha256File(path.join(out, INDEX)); // a raiz: publicado, autentica todos os arquivos
  r.integridade = !!(ix.integridade || (ix.desenho === 'por-shard' && ix.arquivos.every((a) => a.id)));
  r.cabecalhos_bytes = ix.desenho === 'por-shard' ? ix.arquivos.reduce((a, f) => a + f.cabecalho, 0) : PRE_E * model.files.length;
  r.maior_arquivo = Math.max(...model.files.map((f) => f.bytes));
  let t0 = performance.now(), badFiles = 0;
  for (const f of model.files) if (sha256File(path.join(out, f.nome)) !== f.sha256) badFiles++;
  r.sha_arquivos = { conferidos: model.files.length, errados: badFiles, tempo_s: (performance.now() - t0) / 1000 };
  t0 = performance.now();
  try { r.volta = await roundtrip(model); } catch (e) { r.volta = []; r.erro = e.message; model.close(); return r; }
  r.volta_tempo_s = (performance.now() - t0) / 1000;
  // leitura do modelo inteiro numa profundidade (decodifica tudo, sem gravar)
  t0 = performance.now(); const counter = { bytes: 0 };
  for (const sh of model.shards) for (const t of sh.tensors) await decodeTensor(model, t, depth, counter);
  r.leitura = { profundidade: depth, bytes_lidos: counter.bytes, tempo_s: (performance.now() - t0) / 1000 };
  // requisições HTTP: modelo inteiro em cada profundidade, uma camada, e subir de profundidade
  const M = Math.max(...model.shards.flatMap((s) => s.tensors.map((t) => Math.max(...t.streams.map((e) => e.s)))));
  r.http = { modelo: [], camada: [], subir: [] };
  for (let d = 0; d <= Math.min(M, 7); d++) r.http.modelo.push({ d, ...ranges(model, (t, e) => e.s <= d) });
  const layers = [...new Set(model.shards.flatMap((s) => s.tensors.map((t) => (t.name.match(/\.layers\.(\d+)\./) || [])[1]).filter(Boolean)))];
  const L = layers.length ? layers[Math.floor(layers.length / 2) - 1] || layers[0] : null; // uma camada do meio
  if (L !== null) for (const d of [0, 2, 7]) r.http.camada.push({ camada: +L, d, ...ranges(model, (t, e) => e.s <= d && t.name.includes(`.layers.${L}.`)) });
  for (const [a, b] of [[0, 2], [2, 7]]) r.http.subir.push({ de: a, para: b, ...ranges(model, (t, e) => e.s > a && e.s <= b) });
  r.rss_gb = rssGB();
  model.close();
  return r;
}

/* ---------------- testes com modelo sintético ---------------- */
function writeSafetensors(p, tensors, metadata) {
  const { file, headerBytes } = S.buildSafetensors(tensors, metadata);
  fs.writeFileSync(p, file); return headerBytes;
}
async function unit() {
  let fails = 0, checks = 0;
  const check = (ok, msg) => { checks++; if (!ok) { fails++; console.log('  FALHOU:', msg); } return ok; };
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'strata-shards-')), dir = path.join(tmp, 'modelo');
  fs.mkdirSync(dir);
  const shards = [S.synthModel(3e5, 1), S.synthModel(2e5, 2), S.synthModel(1e5, 3)].map((ts, i) => ts.map((t) => ({ ...t, name: `s${i}.` + t.name })));
  shards[1].push({ name: 'pos_ids', dtype: 'I64', shape: [3], bytes: new Uint8Array(new BigInt64Array([7n, 42n, -1n]).buffer) });
  const f32 = new Float32Array(1000); for (let i = 0; i < f32.length; i++) f32[i] = Math.sin(i) * 1e-3;
  shards[2].push({ name: 'norm.f32', dtype: 'F32', shape: [1000], bytes: new Uint8Array(f32.buffer) }); // shard com M = 23
  const names = shards.map((_, i) => `model-0000${i + 1}-of-00003.safetensors`), weight_map = {};
  names.forEach((n, i) => { writeSafetensors(path.join(dir, n), shards[i], i === 0 ? { format: 'pt' } : null); for (const t of shards[i]) weight_map[t.name] = n; });
  fs.writeFileSync(path.join(dir, 'model.safetensors.index.json'), JSON.stringify({ metadata: { total_size: 123 }, weight_map }, null, 2) + '\n');
  const work = path.join(tmp, 'trabalho');
  for (const n of names) await encodeShard(dir, work, n, 2000); // a norma (256 a 512 pesos) e o F32 de 1000 vão crus
  buildD1(dir, work, path.join(tmp, 'd1'));
  buildD2(dir, work, path.join(tmp, 'd2'), 40000); // teto pequeno para forçar várias partes por estrato
  // D1 = exatamente o arquivo do conversor de um arquivo só
  for (let i = 0; i < names.length; i++) {
    const raw = fs.readFileSync(path.join(dir, names[i])), st = S.readSafetensors(new Uint8Array(raw));
    const ref = await S.encode(st.tensors, { sourceHeaderBytes: st.headerBytes, sourceSha256: createHash('sha256').update(raw).digest('hex'), cruAbaixo: 2000 });
    const got = new Uint8Array(fs.readFileSync(path.join(tmp, 'd1', names[i].replace('.safetensors', '.strata'))));
    const same = ref.file.length === got.length && ref.file.every((v, j) => v === got[j]);
    console.log(`${check(same, names[i] + ': D1 difere do S.encode') ? 'ok' : '--'} D1 ${names[i]} idêntico ao S.encode (${got.length} B)`);
    const ix = JSON.parse(fs.readFileSync(path.join(tmp, 'd1', INDEX), 'utf8'));
    check(JSON.stringify(ix.arquivos[i].prefixos) === JSON.stringify(ref.prefix), names[i] + ': prefixos do índice diferem dos do core');
    // prefixo do estrato 2 lido pelo core = mesmo valor que a leitura deste script
    const dec = await S.decode(got.subarray(0, ref.prefix[2]), 2);
    const m = openModel(path.join(tmp, 'd1')), t0 = m.shards[i].tensors[0];
    const mine = await decodeTensor(m, t0, 2); m.close();
    check(Buffer.compare(Buffer.from(dec.tensors[0].bytes), Buffer.from(mine)) === 0, names[i] + ': leitura no estrato 2 difere do core');
  }
  for (const d of ['d1', 'd2']) {
    const r = await verify(path.join(tmp, d), 2);
    const ok = r.volta.every((x) => x.igual) && r.sha_arquivos.errados === 0;
    console.log(`${check(ok, d + ': volta bit a bit falhou') ? 'ok' : '--'} ${d} (${r.desenho}): ${r.arquivos} arquivos, ${r.bytes_total} B, ` +
      `volta ${r.volta.filter((x) => x.igual).length}/${r.volta.length} iguais`);
    if (d === 'd2') check(r.arquivos > 8, 'd2: o teto pequeno devia gerar várias partes');
  }
  // os dois desenhos guardam os mesmos streams: o corpo é o mesmo, só cabeçalhos mudam
  const v1 = await verify(path.join(tmp, 'd1'), 7), v2 = await verify(path.join(tmp, 'd2'), 7);
  check(v1.leitura.bytes_lidos === v2.leitura.bytes_lidos, 'corpo diferente entre D1 e D2');
  // arquivo de estrato corrompido: o SHA do índice acusa, e o preâmbulo trocado é recusado
  const ix2 = JSON.parse(fs.readFileSync(path.join(tmp, 'd2', INDEX), 'utf8')), victim = path.join(tmp, 'd2', ix2.arquivos[3].nome);
  const buf = fs.readFileSync(victim), orig = Buffer.from(buf); buf[buf.length - 1] ^= 1; fs.writeFileSync(victim, buf);
  // a leitura também recusa: o hash do stream (A.13) é conferido antes de descomprimir
  const bad = await verify(path.join(tmp, 'd2'), 0);
  console.log(`${check(bad.sha_arquivos.errados === 1, 'd2: bit trocado não foi acusado') ? 'ok' : '--'} d2: bit trocado num arquivo de estrato acusado pelo SHA-256 do índice`);
  console.log(`${check(/hash do stream/.test(bad.erro || ''), 'd2: leitura de stream adulterado passou') ? 'ok' : '--'} d2: leitura do stream adulterado recusada antes de descomprimir (${bad.erro})`);
  fs.copyFileSync(path.join(tmp, 'd2', ix2.arquivos[0].nome), victim);
  let err = null; try { openModel(path.join(tmp, 'd2')).close(); } catch (e) { err = e; }
  console.log(`${check(err && /estrato\/parte/.test(err.message), 'd2: arquivo trocado não foi recusado') ? 'ok' : '--'} d2: arquivo de outro estrato no lugar é recusado`);
  // D1: um byte do cabeçalho mudado muda o id e o arquivo é recusado antes do JSON
  const v1f = path.join(tmp, 'd1', names[0].replace('.safetensors', '.strata')), b1 = fs.readFileSync(v1f);
  const at = b1.indexOf('estrato-maior'); b1[at] = 'E'.charCodeAt(0); fs.writeFileSync(v1f, b1);
  err = null; try { openModel(path.join(tmp, 'd1')).close(); } catch (e) { err = e; }
  console.log(`${check(err && /id do arquivo/.test(err.message), 'd1: cabeçalho adulterado não foi recusado') ? 'ok' : '--'} d1: cabeçalho adulterado recusado pelo id do índice`);
  // tensores pequenos crus: um registro só, n = null, e a tabela binária encolhe na mesma conta
  const crus = ix2.tensors.filter((t) => t.n === null && S.LAYOUTS[t.dtype]);
  const esperado = ix2.tensors.reduce((a, t) => a + (t.n === null ? 1 : 2 + S.LAYOUTS[t.dtype].m), 0);
  console.log(`${check(crus.length === 4 && ix2.tabela.registros === esperado && ix2.tabela.bytes === 16 + REG_H * esperado, 'd2: tensores crus ou tamanho da tabela errados') ? 'ok' : '--'} d2: ${crus.length} tensores pequenos crus; tabela binária de ${ix2.tabela.registros} registros, ${ix2.tabela.bytes} B`);
  // tabela adulterada: o SHA-256 do índice acusa; tabela coerente com o hash mas com registro inválido é recusada
  const tf = path.join(tmp, 'd2', TABELA), tab = fs.readFileSync(tf);
  fs.writeFileSync(victim, orig); // desfaz as adulterações acima
  tab[16 + 4] = 3; fs.writeFileSync(tf, tab); // registro 0 (sinal) passa a dizer estrato 3
  err = null; try { openModel(path.join(tmp, 'd2')).close(); } catch (e) { err = e; }
  console.log(`${check(err && /não confere com o índice/.test(err.message), 'd2: tabela adulterada não foi recusada') ? 'ok' : '--'} d2: tabela adulterada recusada pelo SHA-256 do índice`);
  const ixB = { ...ix2, tabela: { ...ix2.tabela, sha256: createHash('sha256').update(tab).digest('hex') } };
  fs.writeFileSync(path.join(tmp, 'd2', INDEX), JSON.stringify(ixB)); // quem escreve o índice "assina" a tabela errada
  err = null; try { openModel(path.join(tmp, 'd2')).close(); } catch (e) { err = e; }
  console.log(`${check(err && /incoerentes/.test(err.message), 'd2: registro incoerente passou') ? 'ok' : '--'} d2: registro com parte e estrato incoerentes recusado mesmo com o hash da tabela certo (${err && err.message})`);
  fs.rmSync(tmp, { recursive: true, force: true });
  console.log(`\n${checks} conferências, ${fails} falhas`);
  return fails;
}

/* ---------------- linha de comando ---------------- */
const args = process.argv.slice(2);
const opt = (k, def) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : def; };
const cmd = args.shift();
if (!cmd) process.exit(await unit() ? 1 : 0);
if (cmd === 'encode') {
  const cru = Number(opt('--cru-abaixo', String(CRU_ABAIXO)));
  const [dir, work, ...only] = args;
  for (const n of only.length ? only : shardList(dir)) {
    const m = await encodeShard(dir, work, n, cru);
    const body = m.pecas.reduce((a, b) => a + b, 0);
    console.log(`${n}: ${m.tensors.length} tensores, ${(m.bytes / 1e9).toFixed(2)} GB -> ${(body / 1e9).toFixed(2)} GB em ${m.tempo_s.toFixed(0)} s, RSS ${m.rss_gb.toFixed(2)} GB`);
  }
} else if (cmd === 'd1' || cmd === 'd2') {
  const [dir, work, out] = args, cap = Number(opt('--cap-gb', '5')) * 1e9;
  const r = cmd === 'd1' ? buildD1(dir, work, out) : buildD2(dir, work, out, cap);
  console.log(`${cmd}: ${r.arquivos} arquivos em ${r.tempo_s.toFixed(0)} s`);
} else if (cmd === 'verify') {
  const depth = Number(opt('--depth', '2')), json = opt('--json', null);
  const r = await verify(args[0], depth);
  console.log(JSON.stringify(r, null, 1));
  if (json) fs.writeFileSync(json, JSON.stringify(r, null, 1) + '\n');
  process.exit(!r.erro && r.volta.every((x) => x.igual) && r.sha_arquivos.errados === 0 ? 0 : 1);
} else { console.error('comando desconhecido: ' + cmd); process.exit(2); }
