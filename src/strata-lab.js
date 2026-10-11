/* STRATA — pipeline da demonstração. Roda num Web Worker (ou na página, se o worker falhar).
   prepare(): gera o modelo sintético + o tensor-desenho e monta o .safetensors original.
   convert(): .safetensors → .strata → .safetensors, confere SHA-256, mede demãos e família. */
(function (root) {
  'use strict';
  const S = root.Strata;
  // desenho → pesos: intensidade 0..1 vira 0,01..0,04 (três oitavas de expoente),
  // então cada demão de mantissa dobra os tons visíveis: 3, 5, 10, 20…
  const PIC_BASE = 0.01, PIC_SPAN = 0.03;
  const PIC_NAME = 'model.desenho.weight';
  const now = () => (root.performance ? root.performance.now() : Date.now());
  let st = null;

  async function prepare(opts) {
    const t0 = now();
    const seed = (opts.seed >>> 0) || 7;
    const tensors = S.synthModel(opts.params || 1000000, seed);
    if (opts.picture) {
      const rng = S.mulberry32((Math.imul(seed, 2654435761) >>> 0) || 1);
      const pic = opts.picture, v = new Uint16Array(pic.length);
      for (let i = 0; i < pic.length; i++) v[i] = S.f32ToBf16(PIC_BASE + pic[i] * PIC_SPAN + S.gauss(rng) * 0.0004);
      tensors.push({ name: PIC_NAME, dtype: 'BF16', shape: [opts.h, opts.w], bytes: new Uint8Array(v.buffer) });
    }
    const built = S.buildSafetensors(tensors, { format: 'pt', origem: 'sintetico', semente: String(seed) });
    const sha = await S.sha256Hex(built.file);
    let nParams = 0;
    for (const t of tensors) nParams += t.bytes.length / (S.LAYOUTS[t.dtype] ? S.LAYOUTS[t.dtype].bytes : 1);
    st = { seed, file: built.file, sha, nParams, hasPicture: !!opts.picture };
    return { seed, bytes: built.file.length, nParams, sha, ms: Math.round(now() - t0) };
  }

  async function convert(progress) {
    if (!st) throw new Error('modelo não preparado');
    const tick = progress || function () {};
    const T = {}; let t = now();
    const lap = (k) => { const n = now(); T[k] = Math.round(n - t); t = n; };

    const src = S.readSafetensors(st.file);
    lap('ler'); tick('lido');
    const enc = await S.encode(src.tensors, {
      sourceHeaderBytes: src.headerBytes,
      onStage: (s) => { if (s === 'separar') { lap('separar'); tick('separado'); } else { lap('comprimir'); tick('comprimido'); } },
    });
    // as demãos do desenho, lidas de prefixos do arquivo recém-escrito: a página pinta enquanto o resto segue
    const byStratum = new Array(enc.M + 1).fill(0);
    for (const e of enc.table) byStratum[e.s] += e.l;
    if (st.hasPicture) {
      const toF = S.makeToFloat('BF16'), pics = [];
      for (let d = 0; d <= enc.M; d++) {
        const r = await S.decodeTensorAt(enc.file, PIC_NAME, d, enc.prefix[d]);
        const out = new Float32Array(r.vals.length);
        for (let i = 0; i < out.length; i++) { const x = (toF(r.vals[i]) - PIC_BASE) / PIC_SPAN; out[i] = x < 0 ? 0 : x > 1 ? 1 : x; }
        pics.push(out);
      }
      lap('camadas'); tick('camadas', { pics, byStratum, M: enc.M, prefix: enc.prefix, strataBytes: enc.file.length });
    }

    const dec = await S.decode(enc.file);
    const rebuilt = S.rebuildSafetensors(dec.sourceHeaderBytes, dec.tensors);
    lap('remontar'); tick('remontado');
    const shaBack = await S.sha256Hex(rebuilt);
    let same = rebuilt.length === st.file.length;
    if (same) for (let i = 0; i < rebuilt.length; i++) if (rebuilt[i] !== st.file[i]) { same = false; break; }
    lap('conferir'); tick('conferido');

    // métricas só nos pesos "de LLM": o tensor-desenho é um recurso visual e ficaria fora da curva
    const llm = enc.items.filter((it) => it.name !== PIC_NAME);
    // orçamento de 2 M pesos: no modelo de teste (1 M) a conta usa todos os pesos, sem amostragem
    const metrics = S.depthMetrics(llm, enc.M, 2000000);
    let nFloat = 0, eh = 0, nLlm = 0;
    for (const it of enc.items) if (it.kind === 'float') nFloat += it.n;
    for (const it of llm) if (it.kind === 'float') { nLlm += it.n; eh += S.byteEntropy(it.st.exp) * it.n; }
    lap('medir'); tick('medido');

    const child = S.synthChild(src.tensors, st.seed);
    const lin = await S.lineage(enc.items, child.tensors.map(S.prep), 3, 131072);
    lap('familia');

    return {
      seed: st.seed,
      fileBytes: st.file.length,
      strataBytes: enc.file.length,
      headerLen: enc.headerLen,
      M: enc.M,
      prefix: enc.prefix,
      byStratum,
      nFloat,
      nParams: st.nParams,
      metrics,
      expEntropy: nLlm ? eh / nLlm : 0,
      shaOrig: st.sha,
      shaBack,
      identical: same && st.sha === shaBack,
      lineage: {
        standalone: lin.standalone, delta: lin.delta,
        rows: lin.rows.map((r, i) => ({ name: r.name, mode: r.mode, bytes: r.bytes, standalone: r.standalone,
          changed: r.changed === undefined ? 0 : r.changed, cosTrunc: r.cosTrunc || 0, cosBit: r.cosBit || 0, kind: child.kinds[i] })),
      },
      timings: T,
    };
  }

  root.StrataLab = { prepare, convert, PIC_NAME };

  const isWorker = typeof root.WorkerGlobalScope !== 'undefined' && root instanceof root.WorkerGlobalScope;
  if (isWorker) {
    root.onmessage = async (ev) => {
      const m = ev.data || {};
      try {
        if (m.type === 'prepare') {
          const r = await prepare(m);
          root.postMessage({ type: 'prepared', id: m.id, r });
        } else if (m.type === 'convert') {
          const r = await convert((stage, data) => root.postMessage({ type: 'progress', id: m.id, stage, data },
            data && data.pics ? data.pics.map((p) => p.buffer) : []));
          root.postMessage({ type: 'done', id: m.id, r });
        }
      } catch (e) {
        root.postMessage({ type: 'error', id: m.id, message: String((e && e.message) || e) });
      }
    };
  }
})(typeof self !== 'undefined' ? self : globalThis);
