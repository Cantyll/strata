// Segurança do leitor .strata: as regras de A.12 e os hashes de A.13 (docs/02-especificacao-formato.md).
// Confere, sem sorte envolvida:
//   1. o inflate próprio dá o mesmo que o zlib em entradas de todo tipo e recusa o que o zlib recusa;
//   2. o SHA-256 em JS bate com o node:crypto;
//   3. cada ataque conhecido (offset negativo, stream repetido, bomba deflate e rANS, __proto__,
//      dados depois do fim, fonte adulterada, hash trocado...) é recusado com StrataError;
//   4. o que é legítimo continua lendo: arquivo sem hashes, prefixo com hashes, id do cabeçalho.
// Uso: node tests/test-seguranca.mjs     (sai com código 1 se qualquer conferência falhar)
import fs from 'node:fs';
import zlib from 'node:zlib';
import { createHash } from 'node:crypto';

new Function(fs.readFileSync(new URL('../src/strata-core.js', import.meta.url), 'utf8'))();
const S = globalThis.Strata;
const sha = (u8) => createHash('sha256').update(u8).digest('hex');
const te = new TextEncoder(), td = new TextDecoder();
let checks = 0, fails = 0;
function check(ok, msg) { checks++; if (!ok) { fails++; console.log('  FALHOU:', msg); } return ok; }
async function erro(fn) { try { await fn(); return null; } catch (e) { return e; } }
async function recusa(nome, fn) {
  const e = await erro(fn);
  check(e && e.name === 'StrataError', `${nome}: esperava StrataError, veio ${e ? e.name + ': ' + e.message : 'leitura sem erro'}`);
  return e;
}

/* ---- 1. inflate ---- */
console.log('=== inflate estrito contra o zlib ===');
const rng = S.mulberry32(5);
const amostras = [new Uint8Array(0), new Uint8Array(1), new Uint8Array(70000), te.encode('abc'.repeat(5000))];
for (let k = 0; k < 40; k++) {
  const n = (rng() * 200000) | 0, u = new Uint8Array(n), alfabeto = 1 + ((rng() * 255) | 0);
  for (let i = 0; i < n; i++) u[i] = rng() < 0.3 ? u[Math.max(0, i - 1 - ((rng() * 300) | 0))] : (rng() * alfabeto) | 0;
  amostras.push(u);
}
let ok = 0;
for (const u of amostras) {
  for (const o of [{ level: 0 }, { level: 1 }, { level: 9 }, { strategy: zlib.constants.Z_FIXED }, { strategy: zlib.constants.Z_HUFFMAN_ONLY }, { strategy: zlib.constants.Z_RLE }]) {
    const d = new Uint8Array(zlib.deflateRawSync(u, o)), x = S.inflateEstrito(d, u.length);
    if (check(Buffer.compare(Buffer.from(x), Buffer.from(u)) === 0, `inflate difere do zlib (n=${u.length}, ${JSON.stringify(o)})`)) ok++;
  }
}
console.log(`${ok} streams iguais ao zlib`);
const d0 = new Uint8Array(zlib.deflateRawSync(te.encode('estrato '.repeat(1000))));
await recusa('inflate: byte a mais no fim', () => S.inflateEstrito(Uint8Array.of(...d0, 0), 8000));
await recusa('inflate: cortado', () => S.inflateEstrito(d0.subarray(0, d0.length - 1), 8000));
await recusa('inflate: saída maior que a declarada', () => S.inflateEstrito(d0, 7999));
await recusa('inflate: saída menor que a declarada', () => S.inflateEstrito(d0, 8001));
await recusa('inflate: tipo de bloco 3', () => S.inflateEstrito(Uint8Array.of(0x07), 0));
await recusa('inflate: bloco guardado com complemento errado', () => S.inflateEstrito(Uint8Array.of(0x01, 0x01, 0x00, 0x00, 0x00, 0x41), 1));
// bits trocados: o inflate próprio nunca pode aceitar o que o zlib recusa, nem divergir quando os dois aceitam
let concordam = 0;
for (let k = 0; k < 3000; k++) {
  const u = amostras[4 + (k % 40)], d = new Uint8Array(zlib.deflateRawSync(u, { level: 1 + (k % 9) }));
  for (let j = 0; j < 1 + (k % 3); j++) d[(rng() * d.length) | 0] ^= 1 << ((rng() * 8) | 0);
  let z = null, j = null;
  try { z = zlib.inflateRawSync(d, { maxOutputLength: u.length + 1 }); } catch (e) { z = null; }
  try { j = S.inflateEstrito(d, u.length); } catch (e) { if (e.name !== 'StrataError') check(false, 'inflate: erro que não é StrataError: ' + e.message); }
  // o zlib aceita bytes depois do fim; o inflate estrito não, então só conta quando ele aceita
  if (j) { check(z && z.length === u.length && Buffer.compare(z, Buffer.from(j)) === 0, 'inflate aceitou algo que o zlib lê diferente'); concordam++; }
}
console.log(`bits trocados: ${concordam} aceitos, todos iguais ao zlib; o resto recusado`);

/* ---- 2. SHA-256 ---- */
let shaOk = 0;
for (const n of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 1 << 20]) {
  const u = new Uint8Array(n); for (let i = 0; i < n; i++) u[i] = (i * 31 + 7) & 255;
  if (check(S.sha256Js(u) === sha(u) && (await S.sha256Hex(u)) === sha(u), 'sha256 difere em n=' + n)) shaOk++;
}
console.log(`\nSHA-256 em JS igual ao node:crypto em ${shaOk} tamanhos`);

/* ---- 3. ataques ---- */
console.log('\n=== ataques: todos têm que ser recusados ===');
const bf = (vals) => { const u = new Uint16Array(vals.length); vals.forEach((v, i) => { u[i] = S.f32ToBf16(v); }); return new Uint8Array(u.buffer); };
const tensores = [
  { name: 'w', dtype: 'BF16', shape: [4, 50], bytes: bf(Array.from({ length: 200 }, (_, i) => Math.sin(i) * 0.1)) },
  { name: 'ids', dtype: 'I64', shape: [3], bytes: new Uint8Array(new BigInt64Array([1n, 2n, 3n]).buffer) },
];
const { file: st, headerBytes } = S.buildSafetensors(tensores, { origem: 'teste' });
const enc = await S.encode(tensores, { sourceHeaderBytes: headerBytes, sourceSha256: sha(st) });
const F = enc.file;
const partes = (f) => { const hl = new DataView(f.buffer, f.byteOffset).getUint32(8, true); return { h: JSON.parse(td.decode(f.subarray(12, 12 + hl))), corpo: f.subarray(12 + hl) }; };
const monta = (h, corpo) => { const hb = te.encode(JSON.stringify(h)), f = new Uint8Array(12 + hb.length + corpo.length); f.set(S.MAGIC); new DataView(f.buffer).setUint32(8, hb.length, true); f.set(hb, 12); f.set(corpo, 12 + hb.length); return f; };
const muda = (fn) => { const { h, corpo } = partes(F); const r = fn(h, corpo); return r instanceof Uint8Array ? r : monta(h, corpo); };
const ler = (f, o) => S.toSafetensors(f, o);
const exp = (h) => h.streams.find((e) => e.p === 'exp');

check(sha(await ler(F)) === sha(st), 'o arquivo legítimo não voltou idêntico');
await recusa('offset negativo', () => ler(muda((h) => { h.streams[1].o = -12; })));
await recusa('offset que pula para o cabeçalho', () => ler(muda((h) => { h.streams[0].o = 0; h.streams[1].o = 0; })));
await recusa('tamanho maior que o arquivo', () => ler(muda((h) => { h.streams.at(-1).l += 1000; })));
await recusa('stream repetido', () => ler(muda((h) => { h.streams.splice(1, 0, structuredClone(h.streams[1])); })));
await recusa('stream faltando', () => ler(muda((h) => { h.streams.pop(); })));
await recusa('parte __proto__', () => ler(muda((h) => { h.streams[0].p = '__proto__'; })));
await recusa('chave __proto__ no stream', () => ler(muda((h, c) => { const hb = te.encode(JSON.stringify(h).replace('"t":0,', '"__proto__":{"x":1},"t":0,')); const f = new Uint8Array(12 + hb.length + c.length); f.set(S.MAGIC); new DataView(f.buffer).setUint32(8, hb.length, true); f.set(hb, 12); f.set(c, 12 + hb.length); return f; })));
await recusa('tensor fora da tabela', () => ler(muda((h) => { h.streams[0].t = 9; })));
await recusa('estrato trocado', () => ler(muda((h) => { h.streams.at(-1).s = 0; })));
await recusa('codificação desconhecida', () => ler(muda((h) => { h.streams[0].c = 'gzip'; })));
await recusa('dtype que não bate com os bytes', () => ler(muda((h) => { h.tensors[0].dtype = 'F32'; })));
await recusa('shape gigante', () => ler(muda((h) => { h.tensors[0].shape = [2 ** 32, 2 ** 32]; })));
await recusa('nome de tensor repetido', () => ler(muda((h) => { h.tensors[1].name = 'w'; })));
await recusa('dados depois do fim', () => ler(Uint8Array.of(...F, 0)));
await recusa('cabeçalho de 4 GB', () => { const f = F.slice(); new DataView(f.buffer).setUint32(8, 0xffffffff, true); return ler(f); });
await recusa('JSON com 100 níveis', () => ler(muda((h) => { h.pai = JSON.parse('['.repeat(100) + ']'.repeat(100)); })));
await recusa('UTF-8 inválido no cabeçalho', () => { const f = F.slice(), i = 12 + td.decode(F.subarray(12)).indexOf('"w"') + 1; f[i] = 0xff; return ler(f); });
await recusa('bomba rANS: 2^31 expoentes de 9 bytes', () => ler(muda((h, c) => {
  const e = exp(h), b = Uint8Array.of(1, 0, 0, 0, 0x80, 0, 0, 0x80, 0), n = 2 ** 31;
  Object.assign(h.tensors[0], { shape: [n], n, bytes: 2 * n }); Object.assign(e, { c: 'rans', r: n, l: b.length });
  return monta(h, c);
})));
await recusa('bomba deflate: 128 MB de zeros num stream de 4 elementos', async () => {
  const bomba = new Uint8Array(zlib.deflateRawSync(Buffer.alloc(128 << 20))), { h, corpo } = partes(F), e = exp(h);
  const c = new Uint8Array(corpo.length - e.l + bomba.length);
  c.set(corpo.subarray(0, e.o)); c.set(bomba, e.o); c.set(corpo.subarray(e.o + e.l), e.o + bomba.length);
  for (const x of h.streams) if (x.o > e.o) x.o += bomba.length - e.l;
  Object.assign(e, { c: 'deflate', l: bomba.length, h: sha(bomba) });
  return ler(monta(h, c));
});
await recusa('limiteSaida', () => ler(F, { limiteSaida: 100 }));
await recusa('fonte com data_offsets sobrepostos', () => ler(muda((h) => {
  const s = JSON.parse(atob(h.fonte.header_b64)); s.ids.data_offsets = [0, 24]; h.fonte.header_b64 = btoa(JSON.stringify(s));
})));
await recusa('fonte com base64 não canônico', () => ler(muda((h) => { h.fonte.header_b64 = h.fonte.header_b64.replace(/=*$/, ''); })));
await recusa('fonte que não confere com o SHA-256 do original', () => ler(muda((h) => {
  h.fonte.header_b64 = btoa(atob(h.fonte.header_b64).trimEnd() + '        ');
  h.fonte.sha256 = '0'.repeat(64);
})));
// hashes: um byte trocado em qualquer stream é pego antes de chegar ao descompressor
let pegos = 0;
for (const e of partes(F).h.streams) {
  const f = F.slice(), base = F.length - partes(F).corpo.length;
  if (!e.l) continue;
  f[base + e.o + (e.l >> 1)] ^= 1;
  const r = await erro(() => ler(f));
  if (check(r && r.name === 'StrataError' && /hash/.test(r.message), `byte trocado em ${e.p} passou`)) pegos++;
}
console.log(`byte trocado no corpo: pego pelo hash em ${pegos} de ${partes(F).h.streams.filter((e) => e.l).length} streams`);
await recusa('stream trocado com hash refeito (o hash do tensor pega)', () => ler(muda((h, c) => {
  const e = h.streams.find((x) => x.p === 'm7'), c2 = c.slice(); c2[e.o] ^= 0x80; e.h = sha(c2.subarray(e.o, e.o + e.l)); return monta(h, c2);
})));
await recusa('id do cabeçalho errado', () => ler(F, { idEsperado: '0'.repeat(64) }));
await recusa('exigirIntegridade num arquivo sem hashes', async () => {
  const sem = await S.encode(tensores, { sourceHeaderBytes: headerBytes, integridade: false });
  return ler(sem.file, { exigirIntegridade: true });
});
await recusa('hashes apagados de um arquivo com id publicado', () => ler(muda((h) => {
  delete h.integridade; for (const e of h.streams) delete e.h; for (const t of h.tensors) delete t.sha256;
}), { idEsperado: '0'.repeat(64) }));
await recusa('volta com tensor maior que o espaço dele', () => {
  const t2 = tensores.map((t) => ({ ...t })); t2[1] = { ...t2[1], bytes: new Uint8Array(32) };
  return S.rebuildSafetensors(headerBytes, t2);
});

/* ---- 4. o que é legítimo continua lendo ---- */
console.log('\n=== legítimo ===');
const id = await S.cabecalhoId(F);
check(sha(await ler(F, { idEsperado: id, exigirIntegridade: true })) === sha(st), 'leitura com id e exigirIntegridade');
const semHash = await S.encode(tensores, { sourceHeaderBytes: headerBytes, integridade: false });
check(sha(await ler(semHash.file)) === sha(st), 'arquivo sem hashes (como os da 0.2 antes desta mudança) ainda lê');
for (let d = 0; d <= enc.M; d++) {
  const p = await S.decode(F.subarray(0, enc.prefix[d]), d, { exigirIntegridade: true });
  check(p.tensors.length === 2, 'prefixo ' + d);
}
const ext = enc.header.streams.filter((e) => e.h).length, hb = enc.headerLen, hbSem = semHash.headerLen;
console.log(`id do cabeçalho ${id.slice(0, 16)}…; prefixos 0..${enc.M} leem com hashes conferidos`);
console.log(`custo dos hashes neste arquivo: cabeçalho ${hbSem} -> ${hb} bytes (${ext} streams, ${tensores.length} tensores)`);

/* ---- 5. vetores públicos (os mesmos que o leitor Rust confere em leitor-rs/tests/vetores.rs) ---- */
const dirV = new URL('./vetores/', import.meta.url), man = JSON.parse(fs.readFileSync(new URL('vetores.json', dirV), 'utf8'));
let vOk = 0;
for (const v of man.vetores) {
  const f = new Uint8Array(fs.readFileSync(new URL(v.arquivo, dirV)));
  if (v.espera === 'recusa') { if ((await recusa(v.arquivo, () => ler(f)))?.name === 'StrataError') vOk++; continue; }
  let bom = check(sha(await ler(f)) === v.sha256_safetensors, v.arquivo + ': volta');
  for (const p of v.prefixos || []) {
    const d = await S.decode(f.subarray(0, p.bytes), p.profundidade), cat = Buffer.concat(d.tensors.map((t) => Buffer.from(t.bytes.buffer, t.bytes.byteOffset, t.bytes.length)));
    bom = check(sha(cat) === p.sha256_tensores, `${v.arquivo}: prefixo ${p.profundidade}`) && bom;
  }
  if (bom) vOk++;
}
console.log(`vetores públicos: ${vOk} de ${man.vetores.length} conferem`);

console.log(`\n${checks} conferências, ${fails} falhas`);
process.exit(fails ? 1 : 0);
