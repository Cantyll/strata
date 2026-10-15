/* STRATA v0.1 — núcleo do codec do protótipo.
   Converte tensores (BF16 / FP16 / FP32 / FP8) em estratos de bits comprimidos, monta um
   contêiner .strata em ordem estrato-maior (truncar o arquivo = ler menos precisão) e faz o
   caminho de volta bit-exato até o .safetensors original.
   Roda no navegador (CompressionStream) e no Node 18+. */
(function (root) {
  'use strict';

  const LAYOUTS = {
    BF16:    { bits: 16, e: 8, m: 7,  bytes: 2, label: 'BF16' },
    F16:     { bits: 16, e: 5, m: 10, bytes: 2, label: 'FP16' },
    F32:     { bits: 32, e: 8, m: 23, bytes: 4, label: 'FP32' },
    F8_E4M3: { bits: 8,  e: 4, m: 3,  bytes: 1, label: 'FP8 E4M3' },
    F8_E5M2: { bits: 8,  e: 5, m: 2,  bytes: 1, label: 'FP8 E5M2' },
  };
  const DTYPE_BYTES = { BF16: 2, F16: 2, F32: 4, F64: 8, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1,
    U16: 2, U32: 4, U64: 8, BOOL: 1, F8_E4M3: 1, F8_E5M2: 1 };
  const MAGIC = new Uint8Array([0x53, 0x54, 0x52, 0x41, 0x54, 0x41, 0x00, 0x01]); // "STRATA" 0 1

  const te = new TextEncoder();
  const td = new TextDecoder();

  /* ---------------- números ---------------- */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0; while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }

  const _f = new Float32Array(1), _u = new Uint32Array(_f.buffer);
  function f32ToBf16(x) {
    _f[0] = x; let u = _u[0];
    if ((u & 0x7f800000) === 0x7f800000 && (u & 0x007fffff)) return ((u >>> 16) | 0x40) & 0xffff;
    u = (u + 0x7fff + ((u >>> 16) & 1)) >>> 0;
    return (u >>> 16) & 0xffff;
  }
  function f32ToF16(val) {
    _f[0] = val; const x = _u[0];
    const sign = (x >>> 16) & 0x8000, E = (x >>> 23) & 0xff;
    let e = E - 127 + 15, m = x & 0x7fffff;
    if (E === 0xff) return sign | 0x7c00 | (m ? 0x200 : 0);
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) {
      if (e < -10) return sign;
      m |= 0x800000;
      const shift = 14 - e, half = 1 << (shift - 1);
      let hm = m >>> shift; const rem = m & ((1 << shift) - 1);
      if (rem > half || (rem === half && (hm & 1))) hm++;
      return sign | hm;
    }
    let hm = m >>> 13; const rem = m & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (hm & 1))) {
      hm++; if (hm === 0x400) { hm = 0; e++; if (e >= 31) return sign | 0x7c00; }
    }
    return sign | (e << 10) | hm;
  }

  const _tables = {};
  function tableFor(name) {
    if (_tables[name]) return _tables[name];
    let t;
    if (name === 'BF16') {
      t = new Float32Array(65536); const u = new Uint32Array(t.buffer);
      for (let h = 0; h < 65536; h++) u[h] = h << 16;
    } else if (name === 'F16') {
      t = new Float32Array(65536);
      for (let h = 0; h < 65536; h++) {
        const s = (h & 0x8000) ? -1 : 1, e = (h >>> 10) & 0x1f, m = h & 0x3ff;
        t[h] = e === 0 ? s * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : s * Infinity)
          : s * (1 + m / 1024) * Math.pow(2, e - 15);
      }
    } else if (name === 'F8_E4M3') {
      t = new Float32Array(256);
      for (let h = 0; h < 256; h++) {
        const s = (h & 0x80) ? -1 : 1, e = (h >>> 3) & 0xf, m = h & 7;
        t[h] = (e === 15 && m === 7) ? NaN : e === 0 ? s * m * Math.pow(2, -9) : s * (1 + m / 8) * Math.pow(2, e - 7);
      }
    } else if (name === 'F8_E5M2') {
      t = new Float32Array(256);
      for (let h = 0; h < 256; h++) {
        const s = (h & 0x80) ? -1 : 1, e = (h >>> 2) & 0x1f, m = h & 3;
        t[h] = e === 0 ? s * m * Math.pow(2, -16) : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 4) * Math.pow(2, e - 15);
      }
    }
    _tables[name] = t; return t;
  }
  function makeToFloat(name) {
    if (name === 'F32') return (u) => { _u[0] = u; return _f[0]; };
    const t = tableFor(name); return (u) => t[u];
  }

  /* --------- truncamento com reconstrução no ponto médio --------- */
  function truncMid(u, L, d) {
    if (d >= L.m) return u;
    const drop = L.m - d;
    const e = (u >>> L.m) & ((1 << L.e) - 1);
    let v = (u & ~((1 << drop) - 1)) >>> 0;
    if (e !== 0 && e !== (1 << L.e) - 1) v = (v | (1 << (drop - 1))) >>> 0;
    return v;
  }

  /* ---------------- estratos ---------------- */
  function splitStrata(vals, L) {
    const n = vals.length, nb = (n + 7) >>> 3;
    const sign = new Uint8Array(nb), exp = new Uint8Array(n);
    const planes = [];
    for (let j = 0; j < L.m; j++) planes.push(new Uint8Array(nb));
    const sShift = L.bits - 1, eMask = (1 << L.e) - 1, m = L.m;
    for (let i = 0; i < n; i++) {
      const u = vals[i] >>> 0, byte = i >>> 3, bit = 1 << (7 - (i & 7));
      if ((u >>> sShift) & 1) sign[byte] |= bit;
      exp[i] = (u >>> m) & eMask;
      for (let j = 0; j < m; j++) if ((u >>> (m - 1 - j)) & 1) planes[j][byte] |= bit;
    }
    return { sign, exp, planes };
  }
  function joinStrata(st, L, n, depth, mid) {
    const out = L.bits === 32 ? new Uint32Array(n) : L.bits === 16 ? new Uint16Array(n) : new Uint8Array(n);
    const m = L.m, d = Math.min(depth, m), signBit = L.bits === 32 ? 0x80000000 : 1 << (L.bits - 1);
    const eMax = (1 << L.e) - 1, drop = m - d, midBit = drop > 0 ? 1 << (drop - 1) : 0;
    for (let i = 0; i < n; i++) {
      const byte = i >>> 3, bit = 7 - (i & 7);
      const e = st.exp[i];
      let u = (st.sign[byte] >>> bit) & 1 ? signBit : 0;
      u = (u | (e << m)) >>> 0;
      for (let j = 0; j < d; j++) if ((st.planes[j][byte] >>> bit) & 1) u = (u | (1 << (m - 1 - j))) >>> 0;
      if (mid && midBit && e !== 0 && e !== eMax) u = (u | midBit) >>> 0;
      out[i] = u;
    }
    return out;
  }

  /* ---------------- entropia ---------------- */
  const POP = new Uint8Array(256);
  for (let i = 0; i < 256; i++) POP[i] = (i & 1) + POP[i >>> 1];
  function binEnt(p) { return p <= 0 || p >= 1 ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p)); }
  function byteEntropy(u8) {
    const h = new Float64Array(256), n = u8.length; if (!n) return 0;
    for (let i = 0; i < n; i++) h[u8[i]]++;
    let H = 0; for (let k = 0; k < 256; k++) if (h[k]) { const p = h[k] / n; H -= p * Math.log2(p); }
    return H;
  }
  function planeOnes(packed) { let c = 0; for (let i = 0; i < packed.length; i++) c += POP[packed[i]]; return c; }
  function bitPositionEntropy(vals, L, stride) {
    const cnt = new Float64Array(L.bits); let n = 0;
    for (let i = 0; i < vals.length; i += stride) {
      const u = vals[i] >>> 0; n++;
      for (let b = 0; b < L.bits; b++) if ((u >>> b) & 1) cnt[b]++;
    }
    const H = []; for (let b = L.bits - 1; b >= 0; b--) H.push(binEnt(cnt[b] / n));
    return H; // MSB primeiro
  }

  /* ---------------- compressão ---------------- */
  async function pipe(u8, Ctor) {
    const cs = new Ctor('deflate-raw');
    const w = cs.writable.getWriter();
    w.write(u8).catch(() => {}); w.close().catch(() => {});
    return new Uint8Array(await new Response(cs.readable).arrayBuffer());
  }
  const hasCodec = typeof CompressionStream !== 'undefined' && typeof DecompressionStream !== 'undefined';
  const deflate = (u8) => pipe(u8, CompressionStream);
  const inflate = (u8) => pipe(u8, DecompressionStream);

  async function sha256Hex(u8) {
    const c = (root.crypto && root.crypto.subtle) ? root.crypto.subtle : null;
    if (c) {
      const h = new Uint8Array(await c.digest('SHA-256', u8));
      return Array.from(h, (b) => b.toString(16).padStart(2, '0')).join('');
    }
    let h1 = 0x811c9dc5; // fallback FNV-1a (só se SubtleCrypto faltar)
    for (let i = 0; i < u8.length; i++) { h1 ^= u8[i]; h1 = Math.imul(h1, 16777619); }
    return 'fnv1a:' + (h1 >>> 0).toString(16);
  }
  function b64(u8) {
    let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function unb64(s) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }

  /* ---------------- safetensors ---------------- */
  function parseSafetensorsHeader(first8, headerBytes) {
    const N = Number(new DataView(first8.buffer, first8.byteOffset, 8).getBigUint64(0, true));
    const H = JSON.parse(td.decode(headerBytes));
    const entries = Object.entries(H).filter(([k]) => k !== '__metadata__')
      .map(([name, v]) => ({ name, dtype: v.dtype, shape: v.shape, begin: v.data_offsets[0], end: v.data_offsets[1] }))
      .sort((a, b) => a.begin - b.begin);
    return { N, header: H, entries, metadata: H.__metadata__ || null };
  }
  function buildSafetensors(tensors, metadata) {
    const H = {}; if (metadata) H.__metadata__ = metadata;
    let off = 0;
    for (const t of tensors) { H[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [off, off + t.bytes.length] }; off += t.bytes.length; }
    let hb = te.encode(JSON.stringify(H));
    const pad = (8 - (hb.length % 8)) % 8;
    if (pad) { const p = new Uint8Array(hb.length + pad); p.set(hb); p.fill(0x20, hb.length); hb = p; }
    const out = new Uint8Array(8 + hb.length + off);
    new DataView(out.buffer).setBigUint64(0, BigInt(hb.length), true);
    out.set(hb, 8);
    let o = 8 + hb.length; for (const t of tensors) { out.set(t.bytes, o); o += t.bytes.length; }
    return { file: out, headerBytes: hb };
  }
  function rebuildSafetensors(sourceHeaderBytes, tensors) {
    const H = JSON.parse(td.decode(sourceHeaderBytes));
    let end = 0;
    for (const [k, v] of Object.entries(H)) if (k !== '__metadata__') end = Math.max(end, v.data_offsets[1]);
    const out = new Uint8Array(8 + sourceHeaderBytes.length + end);
    new DataView(out.buffer).setBigUint64(0, BigInt(sourceHeaderBytes.length), true);
    out.set(sourceHeaderBytes, 8);
    const base = 8 + sourceHeaderBytes.length;
    for (const t of tensors) out.set(t.bytes, base + H[t.name].data_offsets[0]);
    return out;
  }

  /* ---------------- tensores ---------------- */
  function prep(t) {
    const L = LAYOUTS[t.dtype];
    if (!L || t.bytes.length % L.bytes) return { name: t.name, dtype: t.dtype, shape: t.shape, bytes: t.bytes, kind: 'raw' };
    const copy = t.bytes.slice();
    const vals = L.bytes === 2 ? new Uint16Array(copy.buffer) : L.bytes === 4 ? new Uint32Array(copy.buffer) : copy;
    return { name: t.name, dtype: t.dtype, shape: t.shape, bytes: t.bytes, kind: 'float', L, n: vals.length, vals };
  }

  /* ---------------- contêiner .strata ---------------- */
  async function encode(tensors, opts) {
    opts = opts || {};
    const stage = opts.onStage || function () {};
    const items = tensors.map(prep);
    const floats = items.filter((x) => x.kind === 'float');
    const M = floats.length ? Math.max.apply(null, floats.map((x) => x.L.m)) : 0;
    const streams = [];
    items.forEach((it, ti) => {
      if (it.kind === 'raw') { streams.push({ t: ti, s: 0, p: 'raw', data: it.bytes }); return; }
      it.st = splitStrata(it.vals, it.L);
      streams.push({ t: ti, s: 0, p: 'sign', data: it.st.sign });
      streams.push({ t: ti, s: 0, p: 'exp', data: it.st.exp });
      for (let j = 0; j < it.L.m; j++) streams.push({ t: ti, s: j + 1, p: 'm' + (j + 1), data: it.st.planes[j] });
    });
    stage('separar');
    for (const s of streams) {
      const d = hasCodec ? await deflate(s.data) : null;
      if (d && d.length < s.data.length) { s.enc = d; s.c = 'deflate'; } else { s.enc = s.data; s.c = 'raw'; }
    }
    stage('comprimir');
    streams.sort((a, b) => a.s - b.s || a.t - b.t);
    let off = 0;
    const table = streams.map((s) => { const e = { t: s.t, s: s.s, p: s.p, o: off, l: s.enc.length, r: s.data.length, c: s.c }; off += s.enc.length; return e; });
    const header = {
      format: 'strata/0.1',
      ordem: 'estrato-maior',
      estratos: M + 1,
      tensors: items.map((it) => ({ name: it.name, dtype: it.dtype, shape: it.shape, n: it.n || null, bytes: it.bytes.length })),
      streams: table,
      fonte: opts.sourceHeaderBytes ? { tipo: 'safetensors', header_b64: b64(opts.sourceHeaderBytes) } : null,
      pai: opts.parent || null,
    };
    const hb = te.encode(JSON.stringify(header));
    const file = new Uint8Array(12 + hb.length + off);
    file.set(MAGIC, 0);
    new DataView(file.buffer).setUint32(8, hb.length, true);
    file.set(hb, 12);
    let o = 12 + hb.length; for (const s of streams) { file.set(s.enc, o); o += s.enc.length; }
    // prefixo de bytes necessário para cada profundidade
    const prefix = [];
    for (let d = 0; d <= M; d++) prefix.push(12 + hb.length + table.filter((e) => e.s <= d).reduce((a, e) => a + e.l, 0));
    return { file, header, headerLen: hb.length, items, streams, table, M, prefix };
  }

  async function decode(file, depth) {
    if (depth === undefined) depth = Infinity;
    for (let i = 0; i < 8; i++) if (file[i] !== MAGIC[i]) throw new Error('assinatura .strata inválida');
    const hl = new DataView(file.buffer, file.byteOffset, file.byteLength).getUint32(8, true);
    const header = JSON.parse(td.decode(file.subarray(12, 12 + hl)));
    const base = 12 + hl, parts = header.tensors.map(() => ({}));
    for (const e of header.streams) {
      if (e.s > depth) continue;
      if (base + e.o + e.l > file.length) throw new Error('estrato fora do prefixo lido: peça profundidade menor');
      const bytes = file.subarray(base + e.o, base + e.o + e.l);
      parts[e.t][e.p] = e.c === 'deflate' ? await inflate(bytes) : bytes;
    }
    const tensors = header.tensors.map((t, ti) => {
      const L = LAYOUTS[t.dtype];
      if (!L || !parts[ti].exp) return { name: t.name, dtype: t.dtype, shape: t.shape, bytes: parts[ti].raw };
      const d = Math.min(depth, L.m), st = { sign: parts[ti].sign, exp: parts[ti].exp, planes: [] };
      for (let j = 0; j < d; j++) st.planes.push(parts[ti]['m' + (j + 1)]);
      const vals = joinStrata(st, L, t.n, d, d < L.m);
      return { name: t.name, dtype: t.dtype, shape: t.shape, vals, bytes: new Uint8Array(vals.buffer) };
    });
    return { header, tensors, sourceHeaderBytes: header.fonte ? unb64(header.fonte.header_b64) : null };
  }

  // Lê UM tensor numa profundidade, usando só o prefixo do arquivo até aquele estrato.
  // `limit` = quantos bytes do arquivo o leitor tem; falha se precisar de algo além.
  async function decodeTensorAt(file, name, depth, limit) {
    for (let i = 0; i < 8; i++) if (file[i] !== MAGIC[i]) throw new Error('assinatura .strata inválida');
    const hl = new DataView(file.buffer, file.byteOffset, file.byteLength).getUint32(8, true);
    const header = JSON.parse(td.decode(file.subarray(12, 12 + hl)));
    const ti = header.tensors.findIndex((t) => t.name === name);
    if (ti < 0) throw new Error('tensor ausente: ' + name);
    const t = header.tensors[ti], L = LAYOUTS[t.dtype], base = 12 + hl, parts = {};
    const cap = limit === undefined ? file.length : limit;
    for (const e of header.streams) {
      if (e.t !== ti || e.s > depth) continue;
      if (base + e.o + e.l > cap) throw new Error('estrato fora do prefixo lido');
      const bytes = file.subarray(base + e.o, base + e.o + e.l);
      parts[e.p] = e.c === 'deflate' ? await inflate(bytes) : bytes;
    }
    if (!L || !parts.exp) return { meta: t, bytes: parts.raw };
    const d = Math.min(depth, L.m), st = { sign: parts.sign, exp: parts.exp, planes: [] };
    for (let j = 0; j < d; j++) st.planes.push(parts['m' + (j + 1)]);
    return { meta: t, vals: joinStrata(st, L, t.n, d, d < L.m) };
  }

  function readSafetensors(file) {
    const N = Number(new DataView(file.buffer, file.byteOffset, 8).getBigUint64(0, true));
    const parsed = parseSafetensorsHeader(file.subarray(0, 8), file.subarray(8, 8 + N));
    const base = 8 + N;
    return {
      headerBytes: file.subarray(8, 8 + N),
      metadata: parsed.metadata,
      tensors: parsed.entries.map((e) => ({ name: e.name, dtype: e.dtype, shape: e.shape, bytes: file.slice(base + e.begin, base + e.end) })),
    };
  }

  /* ---------------- métricas ---------------- */
  // amostra = segmentos contíguos de 4096 (preserva blocos para as referências INT4/INT8)
  function segments(n, budget) {
    const SEG = 4096;
    if (n <= budget) return [[0, n]];
    const nseg = Math.max(1, Math.floor(budget / SEG)), step = Math.floor(n / nseg), out = [];
    for (let k = 0; k < nseg; k++) { const a = k * step; out.push([a, Math.min(n, a + SEG)]); }
    return out;
  }
  function blockQuantErr(x, a, b, levels, blk) { // simétrico, escala = max|x|/levels por bloco
    let err = 0;
    for (let s = a; s < b; s += blk) {
      const e = Math.min(b, s + blk); let mx = 0;
      for (let i = s; i < e; i++) { const v = Math.abs(x[i]); if (v > mx) mx = v; }
      const d = mx / levels || 1;
      for (let i = s; i < e; i++) { const q = Math.max(-levels, Math.min(levels, Math.round(x[i] / d))) * d; err += (q - x[i]) * (q - x[i]); }
    }
    return err;
  }
  // erro relativo (RMS) por tensor, médio ponderado pelo número de pesos de cada tensor:
  // um tensor pequeno de valores grandes (ex.: normalização ≈ 1,0) não domina a conta.
  function depthMetrics(items, M, budget) {
    const floats = items.filter((x) => x.kind === 'float');
    const totalN = floats.reduce((s, x) => s + x.n, 0) || 1;
    const acc = new Float64Array(M + 1); let a4 = 0, a8 = 0, wsum = 0, used = 0;
    for (const it of floats) {
      const toF = makeToFloat(it.dtype), L = it.L;
      const per = Math.max(4096, Math.round((budget || 2e6) * it.n / totalN));
      const err = new Float64Array(M + 1); let sig = 0, e4 = 0, e8 = 0;
      for (const [a, b] of segments(it.n, per)) {
        const x = new Float64Array(b - a);
        for (let i = a; i < b; i++) x[i - a] = toF(it.vals[i]);
        let ok = true; for (let i = 0; i < x.length; i++) if (!isFinite(x[i])) { ok = false; break; }
        if (!ok) continue;
        for (let i = 0; i < x.length; i++) sig += x[i] * x[i];
        for (let d = 0; d <= M; d++) {
          let s = 0;
          for (let i = a; i < b; i++) { const y = toF(truncMid(it.vals[i] >>> 0, L, d)) - x[i - a]; s += y * y; }
          err[d] += s;
        }
        e4 += blockQuantErr(x, 0, x.length, 7, 32);
        e8 += blockQuantErr(x, 0, x.length, 127, 32);
        used += x.length;
      }
      if (sig <= 0) continue;
      const w = it.n;
      for (let d = 0; d <= M; d++) acc[d] += w * Math.sqrt(err[d] / sig);
      a4 += w * Math.sqrt(e4 / sig); a8 += w * Math.sqrt(e8 / sig); wsum += w;
    }
    const avg = (v) => (wsum ? v / wsum : 0);
    return { depth: Array.from(acc, avg), int4: avg(a4), int8: avg(a8), sampled: used };
  }

  /* ---------------- genealogia ---------------- */
  async function lineage(parentItems, childItems, depth, budget) {
    const rows = []; let standalone = 0, delta = 0;
    for (let k = 0; k < childItems.length; k++) {
      const p = parentItems[k], c = childItems[k];
      if (c.kind !== 'float') { rows.push({ name: c.name, mode: 'HERDA', bytes: 32, standalone: c.bytes.length }); delta += 32; standalone += c.bytes.length; continue; }
      const L = c.L, n = c.n;
      const x = L.bits === 32 ? new Uint32Array(n) : L.bits === 16 ? new Uint16Array(n) : new Uint8Array(n);
      let changed = 0;
      for (let i = 0; i < n; i++) { const v = (p.vals[i] ^ c.vals[i]) >>> 0; x[i] = v; if (v) changed++; }
      // custo isolado do filho
      const sc = splitStrata(c.vals, L);
      let alone = 0;
      for (const s of [sc.sign, sc.exp].concat(sc.planes)) { const d = await deflate(s); alone += Math.min(d.length, s.length); }
      standalone += alone;
      if (!changed) { rows.push({ name: c.name, mode: 'HERDA', bytes: 32, standalone: alone, changed: 0 }); delta += 32; continue; }
      const sx = splitStrata(x, L), per = [];
      let tot = 0;
      for (const s of [sx.sign, sx.exp].concat(sx.planes)) { const d = await deflate(s); const b = Math.min(d.length, s.length); per.push(b); tot += b; }
      delta += tot;
      // o quanto do Δ sobrevive: pesos truncados vs Δ-estrato de 1 bit
      const toF = makeToFloat(c.dtype); let ab = 0, aa = 0, bb = 0, ab1 = 0, bb1 = 0;
      for (const [a0, b0] of segments(n, budget || 262144)) {
        for (let s = a0; s < b0; s += 64) {
          const e = Math.min(b0, s + 64); let mean = 0;
          for (let i = s; i < e; i++) mean += Math.abs(toF(c.vals[i]) - toF(p.vals[i]));
          mean /= (e - s);
          for (let i = s; i < e; i++) {
            const dt = toF(c.vals[i]) - toF(p.vals[i]);
            const dk = toF(truncMid(c.vals[i] >>> 0, L, depth)) - toF(truncMid(p.vals[i] >>> 0, L, depth));
            const d1 = dt > 0 ? mean : dt < 0 ? -mean : 0;
            aa += dt * dt; ab += dt * dk; bb += dk * dk; ab1 += dt * d1; bb1 += d1 * d1;
          }
        }
      }
      rows.push({ name: c.name, mode: 'DELTA', bytes: tot, standalone: alone, changed: changed / n, perStratum: per,
        cosTrunc: aa && bb ? ab / Math.sqrt(aa * bb) : 0, cosBit: aa && bb1 ? ab1 / Math.sqrt(aa * bb1) : 0 });
    }
    return { rows, standalone, delta };
  }

  /* ---------------- modelos sintéticos ---------------- */
  const SPECS = [
    ['model.embed_tokens.weight', 0.30, 0.020, 'congelado'],
    ['model.layers.0.self_attn.q_proj.weight', 0.12, 0.018, 'LoRA r=8'],
    ['model.layers.0.self_attn.v_proj.weight', 0.12, 0.012, 'LoRA r=8'],
    ['model.layers.0.mlp.up_proj.weight', 0.23, 0.016, 'FT completo'],
    ['model.layers.0.mlp.down_proj.weight', 0.23, 0.014, 'congelado'],
  ];
  function synthModel(params, seed) {
    const rng = mulberry32(seed || 7), cols = params >= 2e6 ? 512 : 256, tensors = [];
    for (const [name, frac, sigma] of SPECS) {
      const rows = Math.max(8, Math.round((params * frac) / cols)), n = rows * cols;
      const outCols = [Math.floor(rng() * cols), Math.floor(rng() * cols)];
      const v = new Uint16Array(n);
      for (let r = 0; r < rows; r++) {
        const sr = sigma * Math.exp(0.35 * gauss(rng));
        for (let c = 0; c < cols; c++) {
          let x = gauss(rng) * sr;
          if (rng() < 0.004) x *= 6;                         // caudas pesadas
          if (c === outCols[0] || c === outCols[1]) x *= 8;   // canais atípicos (outliers)
          v[r * cols + c] = f32ToBf16(x);
        }
      }
      tensors.push({ name, dtype: 'BF16', shape: [rows, cols], bytes: new Uint8Array(v.buffer) });
    }
    const ln = new Uint16Array(cols);
    for (let c = 0; c < cols; c++) ln[c] = f32ToBf16(1 + 0.05 * gauss(rng));
    tensors.push({ name: 'model.layers.0.input_layernorm.weight', dtype: 'BF16', shape: [cols], bytes: new Uint8Array(ln.buffer) });
    return tensors;
  }
  // fine-tune simulado: congela uns tensores, aplica LoRA (posto 8) ou FT completo em outros
  function synthChild(tensors, seed) {
    const rng = mulberry32((seed || 7) * 7919 + 13), kinds = [];
    const out = tensors.map((t, idx) => {
      const L = LAYOUTS[t.dtype];
      const spec = SPECS.find((s) => s[0] === t.name);
      let kind = spec ? spec[3] : (t.shape.length >= 2 ? ['congelado', 'LoRA r=8', 'FT completo'][idx % 3] : 'FT completo');
      if (!L || L.bytes === 1) kind = 'congelado';
      kinds.push(kind);
      if (kind === 'congelado') return { name: t.name, dtype: t.dtype, shape: t.shape, bytes: t.bytes.slice() };
      const it = prep(t), toF = makeToFloat(t.dtype), n = it.n;
      const back = t.dtype === 'BF16' ? f32ToBf16 : t.dtype === 'F16' ? f32ToF16 : (x) => { _f[0] = x; return _u[0]; };
      const x = new Float32Array(n); let ss = 0, cnt = 0;
      for (let i = 0; i < n; i++) { x[i] = toF(it.vals[i]); if (isFinite(x[i])) { ss += x[i] * x[i]; cnt++; } }
      const std = Math.sqrt(ss / Math.max(1, cnt)) || 1e-3;
      const dlt = new Float32Array(n);
      const rows = t.shape.length >= 2 ? t.shape[0] : 1, cols = Math.floor(n / rows);
      if (kind === 'LoRA r=8' && rows > 1) {
        const r = 8, B = new Float32Array(rows * r), A = new Float32Array(r * cols);
        for (let i = 0; i < B.length; i++) B[i] = gauss(rng);
        for (let i = 0; i < A.length; i++) A[i] = gauss(rng);
        let s2 = 0;
        for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
          let s = 0; for (let k = 0; k < r; k++) s += B[i * r + k] * A[k * cols + j];
          dlt[i * cols + j] = s; s2 += s * s;
        }
        const k = (0.01 * std) / Math.sqrt(s2 / (rows * cols));
        for (let i = 0; i < n; i++) dlt[i] *= k;
      } else {
        for (let i = 0; i < n; i++) dlt[i] = gauss(rng) * 0.004 * (Math.abs(x[i]) + 0.25 * std);
      }
      const nv = L.bytes === 2 ? new Uint16Array(n) : new Uint32Array(n);
      for (let i = 0; i < n; i++) nv[i] = isFinite(x[i]) ? back(x[i] + dlt[i]) : it.vals[i];
      return { name: t.name, dtype: t.dtype, shape: t.shape, bytes: new Uint8Array(nv.buffer) };
    });
    return { tensors: out, kinds };
  }

  root.Strata = {
    LAYOUTS, DTYPE_BYTES, MAGIC, hasCodec,
    f32ToBf16, f32ToF16, makeToFloat, truncMid, splitStrata, joinStrata,
    byteEntropy, planeOnes, binEnt, bitPositionEntropy,
    deflate, inflate, sha256Hex,
    parseSafetensorsHeader, buildSafetensors, rebuildSafetensors,
    prep, encode, decode, decodeTensorAt, readSafetensors, depthMetrics, lineage, synthModel, synthChild, segments,
    mulberry32, gauss,
  };
})(typeof window !== 'undefined' ? window : globalThis);
