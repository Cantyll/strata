// Gera os vetores de teste públicos do leitor (.strata válidos e hostis) e o manifesto vetores.json.
// Qualquer leitor do formato (o JS de src/, o Rust de leitor-rs/, um futuro em Python ou C++) tem
// que dar exatamente o resultado do manifesto: "ok" com o SHA-256 da saída, ou "recusa".
// Uso: node tests/vetores/gera.mjs   (determinístico: rodar de novo não muda nenhum byte)
import fs from 'node:fs';
import zlib from 'node:zlib';
import { createHash } from 'node:crypto';

new Function(fs.readFileSync(new URL('../../src/strata-core.js', import.meta.url), 'utf8'))();
const S = globalThis.Strata;
const dir = new URL('./', import.meta.url).pathname;
const sha = (u8) => createHash('sha256').update(u8).digest('hex');
const te = new TextEncoder(), td = new TextDecoder();
const cat = (us) => { const o = new Uint8Array(us.reduce((a, u) => a + u.length, 0)); let p = 0; for (const u of us) { o.set(u, p); p += u.length; } return o; };

const bf = (vals) => { const u = new Uint16Array(vals.length); vals.forEach((v, i) => { u[i] = S.f32ToBf16(v); }); return new Uint8Array(u.buffer); };
const rng = S.mulberry32(2026), aleat = (n) => Uint8Array.from({ length: n }, () => (rng() * 256) | 0);
const exemplo = [
  { name: 'exemplo.weight', dtype: 'BF16', shape: [2, 4], bytes: bf([1, -0.5, 0.199, 0, 3.14, -123.5, 0.00995, 0.000999]) },
  { name: 'exemplo.ids', dtype: 'I64', shape: [2], bytes: new Uint8Array(new BigInt64Array([7n, 42n]).buffer) },
];
const mistos = [
  { name: 'a.f16', dtype: 'F16', shape: [64, 3], bytes: aleat(384) }, { name: 'b.f32', dtype: 'F32', shape: [50], bytes: aleat(200) },
  { name: 'c.e4m3', dtype: 'F8_E4M3', shape: [33], bytes: aleat(33) }, { name: 'd.e5m2', dtype: 'F8_E5M2', shape: [17], bytes: aleat(17) },
  { name: 'e.u8', dtype: 'U8', shape: [40], bytes: aleat(40) }, { name: 'f.bool', dtype: 'BOOL', shape: [9], bytes: new Uint8Array(9).fill(1) },
  { name: 'g.vazio', dtype: 'BF16', shape: [0], bytes: new Uint8Array(0) },
  { name: 'h.bf16', dtype: 'BF16', shape: [3000], bytes: bf(Array.from({ length: 3000 }, (_, i) => Math.sin(i * 0.37) * 0.02)) },
];
async function strata(ts, opts) {
  const { file: st, headerBytes } = S.buildSafetensors(ts, { origem: 'vetores' });
  const enc = await S.encode(ts, { sourceHeaderBytes: headerBytes, sourceSha256: sha(st), ...opts });
  return { st, enc };
}
const partes = (f) => { const hl = new DataView(f.buffer, f.byteOffset).getUint32(8, true); return { h: JSON.parse(td.decode(f.subarray(12, 12 + hl))), corpo: f.subarray(12 + hl) }; };
const monta = (h, corpo, txt) => { const hb = te.encode(txt || JSON.stringify(h)), f = new Uint8Array(12 + hb.length + corpo.length); f.set(S.MAGIC); new DataView(f.buffer).setUint32(8, hb.length, true); f.set(hb, 12); f.set(corpo, 12 + hb.length); return f; };

const vetores = [], grava = (arquivo, bytes, esperado) => { fs.writeFileSync(dir + arquivo, bytes); vetores.push({ arquivo, ...esperado }); };
const ex = await strata(exemplo), mx = await strata(mistos), semHash = await strata(exemplo, { integridade: false });
const E = ex.enc.file, muda = (fn) => { const { h, corpo } = partes(E); const r = fn(h, corpo); return r instanceof Uint8Array ? r : monta(h, corpo); };
const tensoresSha = async (f, d) => sha(cat((await S.decode(f, d)).tensors.map((t) => t.bytes)));

// válidos: a volta para .safetensors e a leitura parcial (ponto médio) têm que dar estes bytes
grava('ok-exemplo.strata', E, { espera: 'ok', sha256_safetensors: sha(ex.st), id: await S.cabecalhoId(E),
  prefixos: await Promise.all([0, 1, 2, 7].map(async (d) => ({ profundidade: d, bytes: ex.enc.prefix[d], sha256_tensores: await tensoresSha(E.subarray(0, ex.enc.prefix[d]), d) }))) });
grava('ok-tipos-mistos.strata', mx.enc.file, { espera: 'ok', sha256_safetensors: sha(mx.st),
  prefixos: await Promise.all([0, 3, 9, 12, 23].map(async (d) => ({ profundidade: d, bytes: mx.enc.prefix[d], sha256_tensores: await tensoresSha(mx.enc.file.subarray(0, mx.enc.prefix[d]), d) }))) });
grava('ok-sem-hashes.strata', semHash.enc.file, { espera: 'ok', sha256_safetensors: sha(ex.st), nota: 'sem integridade: lê, mas recusa com exigirIntegridade' });

// hostis: todos têm que ser recusados
const exp = (h) => h.streams.find((e) => e.p === 'exp');
const recusa = (arquivo, bytes, nota) => grava(arquivo, bytes, { espera: 'recusa', nota });
recusa('ruim-offset-negativo.strata', muda((h) => { h.streams[1].o = -12; }), 'offset negativo aponta para dentro do cabeçalho');
recusa('ruim-stream-repetido.strata', muda((h) => { h.streams.splice(1, 0, structuredClone(h.streams[1])); }), 'o mesmo stream duas vezes na tabela');
recusa('ruim-stream-faltando.strata', muda((h) => { h.streams.pop(); }), 'falta o último plano');
recusa('ruim-parte-proto.strata', muda((h) => { h.streams[0].p = '__proto__'; }), 'parte "__proto__" (poluição de protótipo num leitor ingênuo)');
recusa('ruim-dtype.strata', muda((h) => { h.tensors[0].dtype = 'F32'; }), 'dtype que não bate com os bytes');
recusa('ruim-shape-gigante.strata', muda((h) => { h.tensors[0].shape = [2 ** 32, 2 ** 32]; }), 'shape com 2^64 elementos');
recusa('ruim-dados-depois-do-fim.strata', cat([E, Uint8Array.of(0)]), 'um byte depois do fim do corpo');
recusa('ruim-cabecalho-4gb.strata', (() => { const f = E.slice(); new DataView(f.buffer).setUint32(8, 0xffffffff, true); return f; })(), 'HL = 2^32 − 1');
recusa('ruim-json-fundo.strata', muda((h) => { h.pai = JSON.parse('['.repeat(100) + ']'.repeat(100)); }), 'JSON com 100 níveis');
recusa('ruim-utf8.strata', (() => { const { h, corpo } = partes(E); const f = monta(h, corpo), i = 12 + td.decode(f.subarray(12)).indexOf('exemplo'); f[i] = 0xff; return f; })(), 'byte 0xFF no meio de um nome');
recusa('ruim-bomba-rans.strata', muda((h, c) => {
  const e = exp(h), b = Uint8Array.of(1, 0, 0, 0, 0x80, 0, 0, 0x80, 0), n = 2 ** 31;
  Object.assign(h.tensors[0], { shape: [n], n, bytes: 2 * n }); Object.assign(e, { c: 'rans', r: n, l: b.length });
  return monta(h, c);
}), '9 bytes de rANS que declaram 2^31 expoentes (4 GiB de saída)');
recusa('ruim-bomba-deflate.strata', (() => {
  const bomba = new Uint8Array(zlib.deflateRawSync(Buffer.alloc(16 << 20), { level: 9 })), { h, corpo } = partes(E), e = exp(h);
  const c = cat([corpo.subarray(0, e.o), bomba, corpo.subarray(e.o + e.l)]);
  for (const x of h.streams) if (x.o > e.o) x.o += bomba.length - e.l;
  Object.assign(e, { c: 'deflate', l: bomba.length, h: sha(bomba) });
  return monta(h, c);
})(), '16 KB de deflate que viram 16 MB, num stream de 8 expoentes');
recusa('ruim-deflate-sobra.strata', (() => {
  const { h, corpo } = partes(E), e = h.streams.find((x) => x.p === 'raw');
  const d = cat([new Uint8Array(zlib.deflateRawSync(ex.enc.items[1].bytes)), Uint8Array.of(0x55)]);
  const c = cat([corpo.subarray(0, e.o), d, corpo.subarray(e.o + e.l)]);
  for (const x of h.streams) if (x.o > e.o) x.o += d.length - e.l;
  Object.assign(e, { c: 'deflate', l: d.length, h: sha(d) });
  return monta(h, c);
})(), 'deflate válido com um byte sobrando depois do último bloco');
recusa('ruim-rans-fora-de-ordem.strata', muda((h, c) => {
  const e = exp(h), b = Uint8Array.of(2, 0, 9, 0, 0x40, 3, 0, 0x40, 0, 0, 0x80, 0), c2 = cat([c.subarray(0, e.o), b, c.subarray(e.o + e.l)]);
  for (const x of h.streams) if (x.o > e.o) x.o += b.length - e.l;
  Object.assign(e, { c: 'rans', l: b.length, h: sha(b) });
  return monta(h, c2);
}), 'tabela do rANS com símbolos fora de ordem crescente');
recusa('ruim-hash-stream.strata', (() => { const f = E.slice(); f[f.length - 1] ^= 1; return f; })(), 'um bit trocado no último stream');
recusa('ruim-hash-tensor.strata', muda((h, c) => {
  const e = h.streams.find((x) => x.p === 'm7'), c2 = c.slice(); c2[e.o] ^= 0x80; e.h = sha(c2.subarray(e.o, e.o + e.l)); return monta(h, c2);
}), 'bit trocado num plano com o hash do stream refeito: o hash do tensor pega');
recusa('ruim-fonte-sobreposta.strata', muda((h) => {
  const s = JSON.parse(atob(h.fonte.header_b64)); s['exemplo.ids'].data_offsets = [0, 16]; h.fonte.header_b64 = btoa(JSON.stringify(s));
}), 'safetensors de origem com data_offsets sobrepostos');
recusa('ruim-base64.strata', muda((h) => { h.fonte.header_b64 = h.fonte.header_b64.replace(/=*$/, ''); }), 'base64 sem o enchimento');
recusa('ruim-sha-original.strata', muda((h) => { h.fonte.sha256 = '0'.repeat(64); }), 'o .safetensors reconstruído não bate com o SHA-256 do original');

fs.writeFileSync(dir + 'vetores.json', JSON.stringify({
  sobre: 'Vetores de teste do leitor .strata (docs/02, A.12 e A.13). "ok": a volta para .safetensors tem o SHA-256 indicado e cada prefixo, lido na profundidade indicada, dá tensores cuja concatenação tem o SHA-256 indicado. "recusa": o leitor tem que recusar o arquivo sem travar nem alocar além do limite.',
  vetores,
}, null, 1) + '\n');
const tot = vetores.reduce((a, v) => a + fs.statSync(dir + v.arquivo).size, 0);
console.log(`${vetores.length} vetores (${vetores.filter((v) => v.espera === 'ok').length} ok), ${tot} bytes`);
