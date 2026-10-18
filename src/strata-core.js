/* STRATA v0.2 — núcleo do codec do protótipo.
   Converte tensores (BF16 / FP16 / FP32 / FP8) em estratos de bits comprimidos, monta um
   contêiner .strata em ordem estrato-maior (truncar o arquivo = ler menos precisão) e faz o
   caminho de volta bit-exato até o .safetensors original. O leitor trata o arquivo como hostil
   (docs/02, A.12) e confere o SHA-256 de cada stream e de cada tensor (A.13).
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
  // "STRATA" 0 2. A versão 2 introduz o rANS no expoente; um leitor da 0.1 recusa o arquivo em vez de
  // tratar o stream rANS como cru. Este leitor aceita as versões 1 e 2.
  const MAGIC = new Uint8Array([0x53, 0x54, 0x52, 0x41, 0x54, 0x41, 0x00, 0x02]);

  const te = new TextEncoder();

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

  /* ---------------- rANS de ordem 0 (expoente) ----------------
     rANS com estado de 32 bits e saída em bytes (o "rans_byte" de F. Giesen), só com inteiros:
     o mesmo arquivo sai igual em qualquer máquina. Probabilidades em 2^15, uma tabela por stream.
     Stream: k (u16) | k × [símbolo u8, freq u16] | estado final (u32) | bytes do corpo.
     Todos os inteiros em little-endian. O número de símbolos vem de `r` na tabela de streams. */
  const RANS_BITS = 15, RANS_M = 1 << RANS_BITS, RANS_L = 1 << 23;

  // Quantiza contagens para frequências que somam 2^15, cada símbolo presente com freq >= 1.
  // Ajusta a soma tirando ou pondo 1 onde isso custa ou rende menos bits (guloso, determinístico).
  function ransFreqs(counts, n) {
    const f = new Uint32Array(256);
    if (!n) return f;
    let sum = 0;
    for (let s = 0; s < 256; s++) if (counts[s]) { f[s] = Math.max(1, Math.round(counts[s] * RANS_M / n)); sum += f[s]; }
    while (sum !== RANS_M) {
      let best = -1, bestCost = Infinity;
      for (let s = 0; s < 256; s++) {
        if (!counts[s]) continue;
        let c;
        if (sum > RANS_M) { if (f[s] <= 1) continue; c = counts[s] * Math.log2(f[s] / (f[s] - 1)); }
        else c = -counts[s] * Math.log2((f[s] + 1) / f[s]);
        if (c < bestCost) { bestCost = c; best = s; }
      }
      if (sum > RANS_M) { f[best]--; sum--; } else { f[best]++; sum++; }
    }
    return f;
  }

  function ransEncode(sym) {
    const n = sym.length, counts = new Float64Array(256);
    for (let i = 0; i < n; i++) counts[sym[i]]++;
    const freq = ransFreqs(counts, n), cum = new Uint32Array(257);
    let k = 0;
    for (let s = 0; s < 256; s++) { cum[s + 1] = cum[s] + freq[s]; if (freq[s]) k++; }
    // bits esperados -> tamanho inicial do buffer (cresce se precisar)
    let bits = 0; for (let s = 0; s < 256; s++) if (counts[s]) bits += counts[s] * (RANS_BITS - Math.log2(freq[s]));
    let buf = new Uint8Array(Math.ceil(bits / 8) + 64), p = 0;
    const put = (b) => { if (p === buf.length) { const g = new Uint8Array(buf.length * 2); g.set(buf); buf = g; } buf[p++] = b; };
    let x = RANS_L;
    for (let i = n - 1; i >= 0; i--) {
      const s = sym[i], fs = freq[s], xmax = ((RANS_L >>> RANS_BITS) << 8) * fs;
      while (x >= xmax) { put(x & 255); x >>>= 8; }
      x = ((x / fs) | 0) * RANS_M + (x % fs) + cum[s];
    }
    put(x >>> 24); put((x >>> 16) & 255); put((x >>> 8) & 255); put(x & 255);
    const head = 2 + 3 * k, out = new Uint8Array(head + p);
    out[0] = k & 255; out[1] = k >>> 8;
    let o = 2;
    for (let s = 0; s < 256; s++) if (freq[s]) { out[o] = s; out[o + 1] = freq[s] & 255; out[o + 2] = freq[s] >>> 8; o += 3; }
    for (let i = 0; i < p; i++) out[head + i] = buf[p - 1 - i]; // o decodificador lê na ordem inversa da escrita
    return out;
  }

  function ransDecode(src, n) {
    if (src.length < 2) falha('rANS: stream curto demais');
    const k = src[0] | (src[1] << 8), freq = new Uint32Array(256), cum = new Uint32Array(256);
    if (k > 256 || (n > 0 && k === 0) || src.length < 2 + 3 * k + 4) falha('rANS: tabela de frequências inválida');
    const out = new Uint8Array(n);
    let o = 2, tot = 0, ant = -1;
    const slot = new Uint8Array(RANS_M);
    for (let j = 0; j < k; j++, o += 3) {
      const s = src[o], fs = src[o + 1] | (src[o + 2] << 8);
      if (s <= ant || !fs || tot + fs > RANS_M) falha('rANS: tabela de frequências inválida');
      ant = s; freq[s] = fs; cum[s] = tot; slot.fill(s, tot, tot + fs); tot += fs;
    }
    if (k && tot !== RANS_M) falha('rANS: frequências não somam 2^15');
    let x = (src[o] | (src[o + 1] << 8) | (src[o + 2] << 16) | (src[o + 3] << 24)) >>> 0, p = o + 4;
    if (x < RANS_L || x >= 0x80000000) falha('rANS: estado inicial inválido');
    const end = src.length, mask = RANS_M - 1;
    for (let i = 0; i < n; i++) {
      const sl = x & mask, s = slot[sl];
      out[i] = s;
      x = freq[s] * (x >>> RANS_BITS) + sl - cum[s];
      while (x < RANS_L) { if (p >= end) falha('rANS: stream truncado'); x = (x << 8) | src[p++]; }
    }
    // o codificador começa em RANS_L; voltar a ele e gastar todos os bytes confere o stream inteiro
    if (x !== RANS_L || p !== end) falha('rANS: stream corrompido');
    return out;
  }

  /* ---------------- inflate estrito (RFC 1951) ----------------
     Um inflate próprio, e não o DecompressionStream, por três motivos: ele para assim que a saída
     passaria do tamanho declarado (sem bomba de descompressão), recusa bytes sobrando depois do
     último bloco (o DecompressionStream do Node aceita, o dos navegadores não) e dá o mesmo
     resultado em qualquer plataforma. Segue o puff.c de Mark Adler, com as mesmas recusas do zlib:
     código de Huffman com excesso, código incompleto (fora o caso de um só código), distância antes
     do início da saída, comprimento de bloco guardado que não confere com o complemento. */
  const INF_LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
  const INF_LEXT = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
  const INF_DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
  const INF_DEXT = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
  const INF_ORDEM = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
  // Tabela canônica: count[len] e símbolos em ordem de código. Devolve o que sobra do espaço de
  // códigos: 0 = completo, > 0 = incompleto, < 0 = excesso.
  function infHuff(lens, ini, n) {
    const count = new Uint16Array(16), symbol = new Uint16Array(n), offs = new Uint16Array(16);
    for (let s = 0; s < n; s++) count[lens[ini + s]]++;
    const h = { count, symbol, left: 0 };
    if (count[0] === n) return h;
    let left = 1;
    for (let len = 1; len < 16; len++) { left = left * 2 - count[len]; if (left < 0) { h.left = left; return h; } }
    for (let len = 1; len < 15; len++) offs[len + 1] = offs[len] + count[len];
    for (let s = 0; s < n; s++) if (lens[ini + s]) symbol[offs[lens[ini + s]]++] = s;
    h.left = left;
    return h;
  }
  let INF_FIXO = null;
  function inflateEstrito(src, outLen) {
    const out = new Uint8Array(outLen), end = src.length;
    let ip = 0, bitbuf = 0, bitcnt = 0, op = 0;
    const bits = (need) => {
      let v = bitbuf;
      while (bitcnt < need) { if (ip >= end) falha('deflate: stream truncado'); v |= src[ip++] << bitcnt; bitcnt += 8; }
      bitbuf = v >>> need; bitcnt -= need;
      return v & ((1 << need) - 1);
    };
    const decodifica = (h) => {
      let code = 0, first = 0, index = 0;
      for (let len = 1; len < 16; len++) {
        if (bitcnt === 0) { if (ip >= end) falha('deflate: stream truncado'); bitbuf = src[ip++]; bitcnt = 8; }
        code |= bitbuf & 1; bitbuf >>>= 1; bitcnt--;
        const count = h.count[len];
        if (code - count < first) return h.symbol[index + (code - first)];
        index += count; first = (first + count) << 1; code <<= 1;
      }
      falha('deflate: código de Huffman inválido');
    };
    const blocoHuff = (lc, dc) => {
      for (;;) {
        let s = decodifica(lc);
        if (s < 256) { if (op >= outLen) falha('deflate: saída maior que o tamanho declarado'); out[op++] = s; continue; }
        if (s === 256) return;
        s -= 257;
        if (s >= 29) falha('deflate: comprimento inválido');
        const len = INF_LBASE[s] + bits(INF_LEXT[s]);
        const ds = decodifica(dc);
        if (ds >= 30) falha('deflate: distância inválida');
        const dist = INF_DBASE[ds] + bits(INF_DEXT[ds]);
        if (dist > op) falha('deflate: distância antes do início');
        if (op + len > outLen) falha('deflate: saída maior que o tamanho declarado');
        for (let k = 0; k < len; k++, op++) out[op] = out[op - dist];
      }
    };
    let ultimo;
    do {
      ultimo = bits(1);
      const tipo = bits(2);
      if (tipo === 0) {
        bitbuf = 0; bitcnt = 0;
        if (ip + 4 > end) falha('deflate: stream truncado');
        const len = src[ip] | (src[ip + 1] << 8), nlen = src[ip + 2] | (src[ip + 3] << 8); ip += 4;
        if (len !== (~nlen & 0xffff)) falha('deflate: bloco guardado inválido');
        if (ip + len > end) falha('deflate: stream truncado');
        if (op + len > outLen) falha('deflate: saída maior que o tamanho declarado');
        out.set(src.subarray(ip, ip + len), op); ip += len; op += len;
      } else if (tipo === 1) {
        if (!INF_FIXO) {
          const l = new Uint8Array(288 + 30);
          for (let s = 0; s < 288; s++) l[s] = s < 144 ? 8 : s < 256 ? 9 : s < 280 ? 7 : 8;
          l.fill(5, 288);
          INF_FIXO = [infHuff(l, 0, 288), infHuff(l, 288, 30)];
        }
        blocoHuff(INF_FIXO[0], INF_FIXO[1]);
      } else if (tipo === 2) {
        const nlen = bits(5) + 257, ndist = bits(5) + 1, ncode = bits(4) + 4;
        if (nlen > 286 || ndist > 30) falha('deflate: contagens inválidas');
        const lens = new Uint8Array(19);
        for (let k = 0; k < ncode; k++) lens[INF_ORDEM[k]] = bits(3);
        const cl = infHuff(lens, 0, 19);
        if (cl.left !== 0) falha('deflate: código dos comprimentos incompleto');
        const ll = new Uint8Array(nlen + ndist);
        for (let k = 0; k < nlen + ndist;) {
          let s = decodifica(cl);
          if (s < 16) { ll[k++] = s; continue; }
          let v = 0;
          if (s === 16) { if (k === 0) falha('deflate: repetição sem anterior'); v = ll[k - 1]; s = 3 + bits(2); }
          else if (s === 17) s = 3 + bits(3);
          else s = 11 + bits(7);
          if (k + s > nlen + ndist) falha('deflate: comprimentos demais');
          ll.fill(v, k, k + s); k += s;
        }
        if (ll[256] === 0) falha('deflate: sem código de fim de bloco');
        const lc = infHuff(ll, 0, nlen), dc = infHuff(ll, nlen, ndist);
        if (lc.left < 0 || (lc.left > 0 && nlen - lc.count[0] !== 1)) falha('deflate: código de literais inválido');
        if (dc.left < 0 || (dc.left > 0 && ndist - dc.count[0] !== 1)) falha('deflate: código de distâncias inválido');
        blocoHuff(lc, dc);
      } else falha('deflate: tipo de bloco inválido');
    } while (!ultimo);
    if (ip !== end) falha('deflate: bytes depois do fim do stream');
    if (op !== outLen) falha('deflate: saída menor que o tamanho declarado');
    return out;
  }

  /* ---------------- SHA-256 ----------------
     Usa o SubtleCrypto quando existe; senão, esta implementação (FIPS 180-4), para que o hash
     confira igual mesmo fora de contexto seguro. */
  const SHA_K = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);
  function sha256Js(u8) {
    const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    const W = new Uint32Array(64), n = u8.length, tot = ((n + 9 + 63) >>> 6) << 6, last = new Uint8Array(tot - (n & ~63));
    last.set(u8.subarray(n & ~63)); last[n & 63] = 0x80;
    const dv = new DataView(last.buffer);
    dv.setUint32(last.length - 8, Math.floor(n / 0x20000000), false); dv.setUint32(last.length - 4, (n * 8) >>> 0, false);
    const bloco = (b, o) => {
      for (let i = 0; i < 16; i++) W[i] = (b[o + 4 * i] << 24) | (b[o + 4 * i + 1] << 16) | (b[o + 4 * i + 2] << 8) | b[o + 4 * i + 3];
      for (let i = 16; i < 64; i++) {
        const a = W[i - 15], c = W[i - 2];
        W[i] = (((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3)) + (((c >>> 17) | (c << 15)) ^ ((c >>> 19) | (c << 13)) ^ (c >>> 10)) + W[i - 7] + W[i - 16];
      }
      let a = H[0], b1 = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
      for (let i = 0; i < 64; i++) {
        const t1 = (h + (((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7))) + ((e & f) ^ (~e & g)) + SHA_K[i] + W[i]) | 0;
        const t2 = ((((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10))) + ((a & b1) ^ (a & c) ^ (b1 & c))) | 0;
        h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b1; b1 = a; a = (t1 + t2) | 0;
      }
      H[0] += a; H[1] += b1; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    };
    for (let o = 0; o + 64 <= n; o += 64) bloco(u8, o);
    for (let o = 0; o < last.length; o += 64) bloco(last, o);
    return Array.from(H, (x) => x.toString(16).padStart(8, '0')).join('');
  }
  // No Node, o node:crypto (rápido e sem limite de tamanho); no navegador, o SubtleCrypto, que no Node
  // recusa mais de 2^31 bits de uma vez, então acima de 256 MB fica a implementação em JS.
  const nodeCrypto = (() => {
    try { return typeof process !== 'undefined' && process.getBuiltinModule ? process.getBuiltinModule('node:crypto') : null; } catch (e) { return null; }
  })();
  async function sha256Hex(u8) {
    if (nodeCrypto) {
      const h = nodeCrypto.createHash('sha256');
      for (let o = 0; o < u8.length; o += 1 << 30) h.update(u8.subarray(o, o + (1 << 30)));
      return h.digest('hex');
    }
    const c = (root.crypto && root.crypto.subtle) ? root.crypto.subtle : null;
    if (!c || u8.length >= 1 << 28) return sha256Js(u8);
    const h = new Uint8Array(await c.digest('SHA-256', u8));
    return Array.from(h, (b) => b.toString(16).padStart(2, '0')).join('');
  }
  function b64(u8) {
    let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function unb64(s) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }

  /* ---------------- validação ----------------
     O leitor trata todo arquivo como hostil (regras em docs/02, A.12): confere cada campo do
     cabeçalho antes de tocar no corpo, só aloca o que o cabeçalho declara e o total até
     `limiteSaida`, confere o hash de cada stream ANTES de descomprimir e o de cada tensor depois.
     Toda recusa é um StrataError; qualquer outro erro que escape daqui é um defeito do leitor. */
  class StrataError extends Error { constructor(m) { super(m); this.name = 'StrataError'; } }
  function falha(m) { throw new StrataError(m); }
  const LIMITE_CABECALHO = 100 * 1024 * 1024; // o mesmo teto do safetensors
  const PROF_JSON = 64;
  const tdEstrito = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
  const HEX64 = /^[0-9a-f]{64}$/, DTYPE_RE = /^[A-Z][A-Z0-9_]{0,31}$/, B64_RE = /^[A-Za-z0-9+/]*={0,2}$/, PLANO_RE = /^m([1-9][0-9]*)$/;
  const SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;
  const ehObj = (x) => x !== null && typeof x === 'object' && !Array.isArray(x);
  const tem = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  function inteiro(x, onde) { if (!Number.isSafeInteger(x) || x < 0) falha(onde + ': inteiro inválido'); return x; }
  function soChaves(o, exigidas, onde) {
    for (const k of Object.keys(o)) if (!exigidas.includes(k)) falha(onde + ': chave desconhecida ' + JSON.stringify(k));
    for (const k of exigidas) if (!tem(o, k)) falha(onde + ': falta a chave ' + k);
  }
  function jsonEstrito(bytes, onde) {
    let v;
    try { v = JSON.parse(tdEstrito.decode(bytes)); } catch (e) { falha(onde + ': JSON ou UTF-8 inválido'); }
    (function anda(x, d) {
      if (d > PROF_JSON) falha(onde + ': JSON com mais de ' + PROF_JSON + ' níveis');
      if (typeof x === 'string') { if (SURROGATE.test(x)) falha(onde + ': texto com surrogate solto'); }
      else if (typeof x === 'number') { if (!Number.isFinite(x)) falha(onde + ': número fora do alcance'); }
      else if (Array.isArray(x)) { for (const y of x) anda(y, d + 1); }
      else if (x !== null && typeof x === 'object') for (const k of Object.keys(x)) { if (SURROGATE.test(k)) falha(onde + ': chave com surrogate solto'); anda(x[k], d + 1); }
    })(v, 0);
    return v;
  }
  function elementos(shape, onde) {
    if (!Array.isArray(shape)) falha(onde + ': shape inválido');
    let el = 1;
    for (const d of shape) { el *= inteiro(d, onde + '.shape'); if (el > Number.MAX_SAFE_INTEGER) falha(onde + ': shape grande demais'); }
    return el;
  }
  function b64Estrito(s, onde) {
    if (typeof s !== 'string' || s.length % 4 || !B64_RE.test(s)) falha(onde + ': base64 inválido');
    const u = unb64(s);
    if (b64(u) !== s) falha(onde + ': base64 não canônico');
    return u;
  }

  /* ---------------- safetensors ---------------- */
  // Cabeçalho do safetensors com as regras do próprio safetensors: começa com '{', no máximo 100 MB,
  // dtype e shape de cada tensor batendo com o tamanho, data_offsets contíguos a partir de 0.
  // `dados` = tamanho da área de dados quando conhecido (null quando só o cabeçalho existe).
  function validaSafetensors(hb, dados) {
    if (hb.length > LIMITE_CABECALHO) falha('safetensors: cabeçalho grande demais');
    if (hb[0] !== 0x7b) falha('safetensors: cabeçalho não começa com {');
    const H = jsonEstrito(hb, 'safetensors');
    if (!ehObj(H)) falha('safetensors: cabeçalho não é um objeto');
    const entries = [];
    for (const name of Object.keys(H)) {
      const v = H[name], onde = 'safetensors ' + JSON.stringify(name);
      if (name === '__metadata__') {
        if (!ehObj(v) || Object.keys(v).some((k) => typeof v[k] !== 'string')) falha('safetensors: __metadata__ inválido');
        continue;
      }
      if (!ehObj(v)) falha(onde + ' inválido');
      soChaves(v, ['dtype', 'shape', 'data_offsets'], onde);
      if (typeof v.dtype !== 'string' || !DTYPE_RE.test(v.dtype)) falha(onde + ': dtype inválido');
      const el = elementos(v.shape, onde), off = v.data_offsets;
      if (!Array.isArray(off) || off.length !== 2) falha(onde + ': data_offsets inválido');
      const begin = inteiro(off[0], onde), end = inteiro(off[1], onde);
      if (end < begin) falha(onde + ': data_offsets invertido');
      const sz = tem(DTYPE_BYTES, v.dtype) ? DTYPE_BYTES[v.dtype] : 0;
      if (sz && end - begin !== el * sz) falha(onde + ': tamanho não bate com shape × dtype');
      entries.push({ name, dtype: v.dtype, shape: v.shape, begin, end });
    }
    entries.sort((a, b) => a.begin - b.begin || a.end - b.end);
    let pos = 0;
    for (const e of entries) { if (e.begin !== pos) falha('safetensors: data_offsets com buraco ou sobreposição'); pos = e.end; }
    if (dados !== null && pos !== dados) falha('safetensors: área de dados não bate com os data_offsets');
    return { header: H, entries, metadata: H.__metadata__ || null, total: pos };
  }
  function parseSafetensorsHeader(first8, headerBytes) {
    const N = Number(new DataView(first8.buffer, first8.byteOffset, 8).getBigUint64(0, true));
    return Object.assign({ N }, validaSafetensors(headerBytes, null));
  }
  function readSafetensors(file) {
    if (file.length < 8) falha('safetensors: arquivo curto demais');
    const N64 = new DataView(file.buffer, file.byteOffset, 8).getBigUint64(0, true);
    if (N64 > BigInt(LIMITE_CABECALHO) || 8 + Number(N64) > file.length) falha('safetensors: tamanho do cabeçalho inválido');
    const N = Number(N64), v = validaSafetensors(file.subarray(8, 8 + N), file.length - 8 - N), base = 8 + N;
    return {
      headerBytes: file.subarray(8, 8 + N),
      metadata: v.metadata,
      tensors: v.entries.map((e) => ({ name: e.name, dtype: e.dtype, shape: e.shape, bytes: file.slice(base + e.begin, base + e.end) })),
    };
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
  // Volta para .safetensors: o cabeçalho original e cada tensor no seu data_offsets. Confere tudo
  // antes de alocar: mesmos nomes dos dois lados e cada tensor do tamanho exato do seu espaço.
  function rebuildSafetensors(sourceHeaderBytes, tensors) {
    const st = validaSafetensors(sourceHeaderBytes, null), porNome = new Map();
    for (const t of tensors) porNome.set(t.name, t);
    if (porNome.size !== tensors.length || tensors.length !== st.entries.length) falha('volta: tensores não batem com o cabeçalho');
    for (const e of st.entries) {
      const t = porNome.get(e.name);
      if (!t || !t.bytes || t.bytes.length !== e.end - e.begin) falha('volta: tensor ' + JSON.stringify(e.name) + ' não bate com o cabeçalho');
    }
    const out = new Uint8Array(8 + sourceHeaderBytes.length + st.total);
    new DataView(out.buffer).setBigUint64(0, BigInt(sourceHeaderBytes.length), true);
    out.set(sourceHeaderBytes, 8);
    const base = 8 + sourceHeaderBytes.length;
    for (const e of st.entries) out.set(porNome.get(e.name).bytes, base + e.begin);
    return out;
  }
  // A fonte guardada no .strata tem que descrever exatamente os mesmos tensores.
  function validaFonte(f, tensors) {
    if (f === null) return null;
    if (!ehObj(f)) falha('fonte inválida');
    soChaves(f, tem(f, 'sha256') ? ['tipo', 'header_b64', 'sha256'] : ['tipo', 'header_b64'], 'fonte');
    if (f.tipo !== 'safetensors') falha('fonte: tipo desconhecido');
    if (tem(f, 'sha256') && (typeof f.sha256 !== 'string' || !HEX64.test(f.sha256))) falha('fonte: sha256 inválido');
    const hb = b64Estrito(f.header_b64, 'fonte.header_b64'), st = validaSafetensors(hb, null);
    if (st.entries.length !== tensors.length) falha('fonte: tensores não batem com o .strata');
    const porNome = new Map(tensors.map((t) => [t.name, t]));
    for (const e of st.entries) {
      const t = porNome.get(e.name);
      if (!t || t.dtype !== e.dtype || t.shape.length !== e.shape.length || t.shape.some((d, i) => d !== e.shape[i]) || e.end - e.begin !== t.bytes)
        falha('fonte: tensor ' + JSON.stringify(e.name) + ' não bate com o .strata');
    }
    return { bytes: hb, sha256: f.sha256 || null };
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
    // o escritor recusa o que o leitor recusaria: nome repetido e tamanho que não bate com shape × dtype
    const nomes = new Set();
    for (const t of tensors) {
      if (typeof t.name !== 'string' || nomes.has(t.name)) throw new Error('encode: nome de tensor inválido ou repetido: ' + t.name);
      nomes.add(t.name);
      if (typeof t.dtype !== 'string' || !DTYPE_RE.test(t.dtype)) throw new Error('encode: dtype inválido em ' + t.name);
      const sz = tem(DTYPE_BYTES, t.dtype) ? DTYPE_BYTES[t.dtype] : 0;
      if (sz && t.bytes.length !== elementos(t.shape, t.name) * sz) throw new Error('encode: ' + t.name + ' tem bytes que não batem com shape × dtype');
    }
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
    // cada stream fica com a menor entre: cru, deflate e (só no expoente) rANS
    for (const s of streams) {
      s.enc = s.data; s.c = 'raw';
      const d = hasCodec ? await deflate(s.data) : null;
      if (d && d.length < s.enc.length) { s.enc = d; s.c = 'deflate'; }
      if (s.p === 'exp' && opts.rans !== false) {
        const r = ransEncode(s.data);
        if (r.length < s.enc.length) { s.enc = r; s.c = 'rans'; }
      }
    }
    stage('comprimir');
    // integridade (A.13): SHA-256 dos bytes gravados de cada stream e dos bytes originais de cada tensor
    const integ = opts.integridade !== false;
    if (integ) {
      for (const s of streams) s.h = await sha256Hex(s.enc);
      for (const it of items) it.sha256 = await sha256Hex(it.bytes);
    }
    streams.sort((a, b) => a.s - b.s || a.t - b.t);
    let off = 0;
    const table = streams.map((s) => {
      const e = { t: s.t, s: s.s, p: s.p, o: off, l: s.enc.length, r: s.data.length, c: s.c };
      if (integ) e.h = s.h;
      off += s.enc.length; return e;
    });
    const fonte = opts.sourceHeaderBytes ? { tipo: 'safetensors', header_b64: b64(opts.sourceHeaderBytes) } : null;
    if (fonte && opts.sourceSha256) fonte.sha256 = opts.sourceSha256;
    const header = {
      format: 'strata/0.2',
      ordem: 'estrato-maior',
      estratos: M + 1,
      tensors: items.map((it) => {
        const t = { name: it.name, dtype: it.dtype, shape: it.shape, n: it.n || (it.kind === 'float' ? 0 : null), bytes: it.bytes.length };
        if (integ) t.sha256 = it.sha256;
        return t;
      }),
      streams: table,
      fonte,
      pai: opts.parent || null,
    };
    if (integ) header.integridade = { alg: 'sha256' };
    validaFonte(fonte, header.tensors);
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

  /* ---------------- leitura ---------------- */
  // Abre e valida o cabeçalho (A.12). Não toca no corpo além de conferir que ele não passa do fim.
  async function abre(file, opts) {
    opts = opts || {};
    if (!(file instanceof Uint8Array)) falha('o arquivo tem que ser um Uint8Array');
    if (file.length < 12) falha('arquivo curto demais');
    for (let i = 0; i < 7; i++) if (file[i] !== MAGIC[i]) falha('assinatura .strata inválida');
    const versao = file[7];
    if (versao !== 1 && versao !== 2) falha('versão .strata desconhecida: ' + versao);
    const hl = new DataView(file.buffer, file.byteOffset, file.byteLength).getUint32(8, true);
    if (hl < 2 || hl > LIMITE_CABECALHO) falha('tamanho do cabeçalho inválido: ' + hl);
    if (12 + hl > file.length) falha('cabeçalho incompleto');
    // com o id publicado por fora (índice, hub), nem o JSON de um cabeçalho adulterado é lido
    if (opts.idEsperado !== undefined && (await sha256Hex(file.subarray(0, 12 + hl))) !== opts.idEsperado) falha('id do cabeçalho não confere');
    const h = jsonEstrito(file.subarray(12, 12 + hl), 'cabeçalho');
    if (!ehObj(h)) falha('cabeçalho não é um objeto');
    for (const k of ['format', 'ordem', 'estratos', 'tensors', 'streams', 'fonte', 'pai']) if (!tem(h, k)) falha('cabeçalho: falta a chave ' + k);
    if (h.format !== (versao === 1 ? 'strata/0.1' : 'strata/0.2')) falha('format não bate com a versão do arquivo');
    if (h.ordem !== 'estrato-maior') falha('ordem desconhecida');
    const integ = tem(h, 'integridade') && h.integridade !== null;
    if (integ) {
      if (!ehObj(h.integridade)) falha('integridade inválida');
      soChaves(h.integridade, ['alg'], 'integridade');
      if (h.integridade.alg !== 'sha256') falha('integridade: algoritmo desconhecido');
    }
    if (opts.exigirIntegridade && !integ) falha('arquivo sem hashes de integridade');
    if (h.pai !== null && !ehObj(h.pai)) falha('pai inválido');
    if (!Array.isArray(h.tensors)) falha('tensors inválido');
    const limite = opts.limiteSaida !== undefined ? opts.limiteSaida : Math.max(1 << 30, 64 * file.length);
    const nomes = new Set(), chT = integ ? ['name', 'dtype', 'shape', 'n', 'bytes', 'sha256'] : ['name', 'dtype', 'shape', 'n', 'bytes'];
    let M = 0, saida = 0;
    const ts = h.tensors.map((t, i) => {
      const onde = 'tensor ' + i;
      if (!ehObj(t)) falha(onde + ' inválido');
      soChaves(t, chT, onde);
      if (typeof t.name !== 'string' || nomes.has(t.name)) falha(onde + ': nome inválido ou repetido');
      nomes.add(t.name);
      if (typeof t.dtype !== 'string' || !DTYPE_RE.test(t.dtype)) falha(onde + ': dtype inválido');
      const el = elementos(t.shape, onde);
      inteiro(t.bytes, onde + '.bytes');
      const L = tem(LAYOUTS, t.dtype) ? LAYOUTS[t.dtype] : null, sz = tem(DTYPE_BYTES, t.dtype) ? DTYPE_BYTES[t.dtype] : 0;
      if (sz && t.bytes !== el * sz) falha(onde + ': bytes não bate com shape × dtype');
      if (L ? t.n !== el : t.n !== null) falha(onde + ': n inválido');
      if (integ && (typeof t.sha256 !== 'string' || !HEX64.test(t.sha256))) falha(onde + ': sha256 inválido');
      if (L) M = Math.max(M, L.m);
      saida += t.bytes;
      if (saida > limite) falha('o cabeçalho declara mais de ' + limite + ' bytes de saída (limiteSaida)');
      return { meta: t, L, n: el, streams: {} };
    });
    if (h.estratos !== M + 1) falha('estratos não bate com os tipos dos tensores');
    if (!Array.isArray(h.streams)) falha('streams inválido');
    const chS = integ ? ['t', 's', 'p', 'o', 'l', 'r', 'c', 'h'] : ['t', 's', 'p', 'o', 'l', 'r', 'c'];
    let off = 0, ant = -1, esperado = 0;
    for (const t of ts) esperado += t.L ? 2 + t.L.m : 1;
    h.streams.forEach((e, i) => {
      const onde = 'stream ' + i;
      if (!ehObj(e)) falha(onde + ' inválido');
      soChaves(e, chS, onde);
      if (inteiro(e.t, onde + '.t') >= ts.length) falha(onde + ': tensor fora da tabela');
      inteiro(e.s, onde + '.s');
      const t = ts[e.t], L = t.L;
      if (typeof e.p !== 'string') falha(onde + ': parte inválida');
      let s, ordem, r;
      if (!L) { if (e.p !== 'raw') falha(onde + ': parte inválida para um tensor cru'); s = 0; ordem = 0; r = t.meta.bytes; }
      else if (e.p === 'sign') { s = 0; ordem = 0; r = Math.ceil(t.n / 8); }
      else if (e.p === 'exp') { s = 0; ordem = 1; r = t.n; }
      else {
        const m = PLANO_RE.exec(e.p);
        if (!m || +m[1] > L.m) falha(onde + ': parte inválida: ' + JSON.stringify(e.p));
        s = +m[1]; ordem = 0; r = Math.ceil(t.n / 8);
      }
      if (e.s !== s) falha(onde + ': estrato não bate com a parte');
      // ordem (s, t, sign antes de exp) estritamente crescente: sem repetição e prefixos corretos
      const chave = (s * ts.length + e.t) * 2 + ordem;
      if (chave <= ant) falha(onde + ': fora de ordem ou repetido');
      ant = chave;
      if (inteiro(e.o, onde + '.o') !== off) falha(onde + ': offset fora de sequência');
      inteiro(e.l, onde + '.l');
      if (inteiro(e.r, onde + '.r') !== r) falha(onde + ': tamanho cru não bate com o tensor');
      if (e.c === 'raw') { if (e.l !== e.r) falha(onde + ': stream cru com tamanho errado'); }
      else if (e.c === 'rans') { if (e.p !== 'exp' || versao < 2) falha(onde + ': rANS fora do expoente'); }
      else if (e.c !== 'deflate') falha(onde + ': codificação desconhecida: ' + JSON.stringify(e.c));
      if (integ && (typeof e.h !== 'string' || !HEX64.test(e.h))) falha(onde + ': hash inválido');
      off += e.l;
      if (off > Number.MAX_SAFE_INTEGER) falha('corpo grande demais');
      t.streams[e.p] = e;
    });
    if (h.streams.length !== esperado) falha('faltam streams na tabela');
    const base = 12 + hl;
    if (file.length > base + off) falha('dados depois do fim do corpo');
    const fonte = validaFonte(h.fonte, h.tensors);
    return { header: h, versao, hl, base, ts, M, integ, fonte };
  }
  // Um stream: dentro do que foi lido, hash conferido antes de decodificar, tamanho exato depois,
  // e a forma canônica (bits de enchimento zerados, expoente dentro do campo do dtype).
  async function leStream(file, A, t, p, cap) {
    const e = t.streams[p], ini = A.base + e.o, fim = ini + e.l;
    if (fim > cap) falha('estrato fora do prefixo lido: peça profundidade menor');
    const bytes = file.subarray(ini, fim);
    if (A.integ && (await sha256Hex(bytes)) !== e.h) falha('hash não confere: ' + t.meta.name + ' ' + p);
    const out = e.c === 'raw' ? bytes : e.c === 'deflate' ? inflateEstrito(bytes, e.r) : ransDecode(bytes, e.r);
    if (p === 'exp') {
      if (t.L.e < 8) { const lim = 1 << t.L.e; for (let i = 0; i < out.length; i++) if (out[i] >= lim) falha('expoente fora do campo do dtype: ' + t.meta.name); }
    } else if (p !== 'raw' && (t.n & 7) && (out[out.length - 1] & (0xff >>> (t.n & 7)))) falha('bits de enchimento não nulos: ' + t.meta.name + ' ' + p);
    return out;
  }
  async function montaTensor(file, A, t, depth, cap) {
    const m = t.meta;
    let vals = null, bytes;
    if (!t.L) bytes = await leStream(file, A, t, 'raw', cap);
    else {
      const d = Math.min(depth, t.L.m), st = { sign: await leStream(file, A, t, 'sign', cap), exp: await leStream(file, A, t, 'exp', cap), planes: [] };
      for (let j = 1; j <= d; j++) st.planes.push(await leStream(file, A, t, 'm' + j, cap));
      vals = joinStrata(st, t.L, t.n, d, d < t.L.m);
      bytes = new Uint8Array(vals.buffer, vals.byteOffset, vals.byteLength);
      if (d < t.L.m) return { name: m.name, dtype: m.dtype, shape: m.shape, vals, bytes };
    }
    if (A.integ && (await sha256Hex(bytes)) !== m.sha256) falha('hash do tensor não confere: ' + m.name);
    return vals ? { name: m.name, dtype: m.dtype, shape: m.shape, vals, bytes } : { name: m.name, dtype: m.dtype, shape: m.shape, bytes };
  }
  function profundidade(depth) {
    if (depth === undefined || depth === Infinity) return Infinity;
    return inteiro(depth, 'profundidade');
  }

  // opts: { exigirIntegridade, idEsperado (SHA-256 hex do cabeçalho), limiteSaida (bytes) }
  async function decode(file, depth, opts) {
    depth = profundidade(depth);
    const A = await abre(file, opts), tensors = [];
    for (const t of A.ts) tensors.push(await montaTensor(file, A, t, depth, file.length));
    return { header: A.header, tensors, sourceHeaderBytes: A.fonte ? A.fonte.bytes : null, integridade: A.integ };
  }

  // Lê UM tensor numa profundidade, usando só o prefixo do arquivo até aquele estrato.
  // `limit` = quantos bytes do arquivo o leitor tem; falha se precisar de algo além.
  async function decodeTensorAt(file, name, depth, limit, opts) {
    depth = profundidade(depth);
    const A = await abre(file, opts);
    const cap = limit === undefined ? file.length : Math.min(inteiro(limit, 'limite'), file.length);
    if (A.base > cap) falha('estrato fora do prefixo lido');
    const t = A.ts.find((x) => x.meta.name === name);
    if (!t) falha('tensor ausente: ' + name);
    const r = await montaTensor(file, A, t, depth, cap);
    return r.vals ? { meta: t.meta, vals: r.vals } : { meta: t.meta, bytes: r.bytes };
  }

  // .strata -> .safetensors byte a byte, conferindo o SHA-256 do original quando o arquivo o guarda.
  async function toSafetensors(file, opts) {
    const d = await decode(file, Infinity, opts);
    if (!d.sourceHeaderBytes) falha('o arquivo não guarda o cabeçalho do safetensors original');
    const out = rebuildSafetensors(d.sourceHeaderBytes, d.tensors), esperado = d.header.fonte.sha256;
    if (esperado && (await sha256Hex(out)) !== esperado) falha('o .safetensors reconstruído não confere com o SHA-256 do original');
    return out;
  }
  // Id do arquivo: SHA-256 de MAGIC + HL + cabeçalho. O cabeçalho guarda o hash de cada stream e de
  // cada tensor, então este id sozinho autentica o arquivo inteiro.
  async function cabecalhoId(file) {
    if (!(file instanceof Uint8Array) || file.length < 12) falha('arquivo curto demais');
    const hl = new DataView(file.buffer, file.byteOffset, file.byteLength).getUint32(8, true);
    if (12 + hl > file.length) falha('cabeçalho incompleto');
    return sha256Hex(file.subarray(0, 12 + hl));
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
    deflate, inflate, inflateEstrito, sha256Hex, sha256Js, ransEncode, ransDecode, ransFreqs,
    parseSafetensorsHeader, buildSafetensors, rebuildSafetensors, StrataError,
    prep, encode, decode, decodeTensorAt, readSafetensors, toSafetensors, cabecalhoId, depthMetrics, lineage, synthModel, synthChild, segments,
    mulberry32, gauss,
  };
})(typeof window !== 'undefined' ? window : globalThis);
