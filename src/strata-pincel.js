/* STRATA — pincel. Pinta, quando a página abre, os ornamentos do painel: os azulejos do fundo,
   os frisos, as molduras, as cantoneiras, as cartelas e as placas. Nada disso é imagem pronta
   nem sombra de CSS: sai de traços de pincel (engordam depois do toque, afinam ao levantar),
   aguadas de densidade irregular e grão, no tamanho real do azulejo e nas cores do tema.
   Cada ornamento tem dois modos: "pinta" (cobalto e mostarda) e "risco" (o contorno pontilhado
   do estresido, antes da pintura). */
(function (root) {
  'use strict';
  const TAU = Math.PI * 2;

  function rng(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const css = (c, a) => (a == null ? 'rgb(' : 'rgba(') + Math.round(c[0]) + ',' + Math.round(c[1]) + ',' + Math.round(c[2]) + (a == null ? ')' : ',' + a.toFixed(3) + ')');
  function tela(w, h) {
    const c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h));
    return c;
  }

  /* ---------- curvas ---------- */
  // bézier cúbicas encadeadas: [x0,y0, c1x,c1y, c2x,c2y, x1,y1, c3x,c3y, c4x,c4y, x2,y2, ...]
  function bez(p, n) {
    const segs = Math.floor((p.length - 2) / 6), out = [];
    const m = Math.max(8, Math.round((n || 36) / segs));
    for (let s = 0; s < segs; s++) {
      const o = s * 6;
      const x0 = p[o], y0 = p[o + 1], x1 = p[o + 2], y1 = p[o + 3], x2 = p[o + 4], y2 = p[o + 5], x3 = p[o + 6], y3 = p[o + 7];
      for (let i = s ? 1 : 0; i <= m; i++) {
        const t = i / m, u = 1 - t;
        out.push([u * u * u * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t * t * t * x3,
          u * u * u * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y3]);
      }
    }
    return out;
  }
  // volta de voluta: raio de r0 a r1, ângulo de a0 a a1 (radianos)
  function espiral(cx, cy, r0, r1, a0, a1, n) {
    const out = []; n = n || 36;
    for (let i = 0; i <= n; i++) {
      const t = i / n, r = r0 + (r1 - r0) * Math.pow(t, .9), a = a0 + (a1 - a0) * t;
      out.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }
    return out;
  }
  // gavinha que sai de (x,y) e termina enrolada: o centro da espiral fica a "raio" do ponto de saída
  function gavinha(x, y, ang, comp, raio, voltas, lado) {
    const c = Math.cos(ang), s = Math.sin(ang);
    const ex = x + c * comp, ey = y + s * comp;               // fim do trecho reto-curvo
    const nx = -s * lado, ny = c * lado;                       // para que lado enrola
    const cx = ex + nx * raio, cy = ey + ny * raio;
    const a0 = Math.atan2(ey - cy, ex - cx);
    const reta = bez([x, y, x + c * comp * .35 - nx * comp * .08, y + s * comp * .35 - ny * comp * .08,
      x + c * comp * .75 - nx * comp * .06, y + s * comp * .75 - ny * comp * .06, ex, ey], 16);
    const volta = espiral(cx, cy, raio, raio * .22, a0, a0 + lado * voltas * TAU, 34);
    return { pts: reta.concat(volta.slice(1)), olho: [cx + Math.cos(a0 + lado * voltas * TAU) * raio * .22, cy + Math.sin(a0 + lado * voltas * TAU) * raio * .22], centro: [cx, cy] };
  }
  function perfil(t, a, b, base) {
    let w = 1;
    if (a > 0 && t < a) w *= Math.pow(Math.sin(t / a * Math.PI / 2), .6);
    if (b > 0 && t > 1 - b) w *= Math.pow(Math.sin((1 - t) / b * Math.PI / 2), .8);
    return base + (1 - base) * w;
  }

  /* ---------- os dois pincéis: o que pinta e o que risca ---------- */
  function pincel(ctx, cores, semente) {
    const R = rng(semente || 1);
    const cor = (k) => (k === 'm' ? cores.mostarda : cores.cobalto);
    const P = {
      R,
      // traço: polígono que segue o caminho com largura variável (toque, corpo, levantada)
      traco(pts, w, o) {
        o = o || {};
        const n = pts.length; if (n < 2) return;
        const ph = R() * TAU, fr = 1.2 + R() * 2.2, tremor = o.tremor == null ? .1 : o.tremor;
        const A = o.a == null ? .12 : o.a, B = o.b == null ? .55 : o.b, base = o.base == null ? .1 : o.base;
        const s = [0];
        for (let i = 1; i < n; i++) s.push(s[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
        const L = s[n - 1] || 1, lf = [], rt = [];
        for (let i = 0; i < n; i++) {
          const p = pts[Math.max(0, i - 1)], q = pts[Math.min(n - 1, i + 1)];
          let dx = q[0] - p[0], dy = q[1] - p[1]; const d = Math.hypot(dx, dy) || 1; dx /= d; dy /= d;
          const t = s[i] / L;
          const h = .5 * w * perfil(t, A, B, base) * (1 + tremor * Math.sin(fr * TAU * t + ph));
          lf.push(pts[i][0] - dy * h, pts[i][1] + dx * h); rt.push(pts[i][0] + dy * h, pts[i][1] - dx * h);
        }
        const path = new Path2D();
        path.moveTo(lf[0], lf[1]);
        for (let i = 2; i < lf.length; i += 2) path.lineTo(lf[i], lf[i + 1]);
        for (let i = rt.length - 2; i >= 0; i -= 2) path.lineTo(rt[i], rt[i + 1]);
        path.closePath();
        const a = o.alfa == null ? .86 + R() * .1 : o.alfa;
        ctx.fillStyle = css(cor(o.cor), a);
        ctx.fill(path);
        // a carga de tinta: mais escura no toque, mais rala no fim
        if (o.carga !== false && w > 0) {
          const g = ctx.createLinearGradient(pts[0][0], pts[0][1], pts[n - 1][0], pts[n - 1][1]);
          g.addColorStop(0, css(cor(o.cor), .22)); g.addColorStop(.5, css(cor(o.cor), .06)); g.addColorStop(1, css(cor(o.cor), 0));
          ctx.fillStyle = g; ctx.fill(path);
        }
      },
      // aguada: preenchimento ralo, com manchas e a tinta acumulada na borda ao secar
      aguada(path, caixa, o) {
        o = o || {};
        const c = cor(o.cor), a = o.alfa == null ? .36 : o.alfa;
        ctx.save();
        ctx.fillStyle = css(c, a); ctx.fill(path);
        ctx.clip(path);
        ctx.strokeStyle = css(c, Math.min(1, a * .9)); ctx.lineWidth = o.borda || .022; ctx.stroke(path);
        const x = caixa[0], y = caixa[1], w = caixa[2], h = caixa[3], k = o.manchas == null ? 4 : o.manchas;
        for (let i = 0; i < k; i++) {
          const cx = x + R() * w, cy = y + R() * h, r = (.25 + R() * .45) * Math.max(w, h);
          const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
          g.addColorStop(0, css(c, a * (.35 + R() * .45))); g.addColorStop(1, css(c, 0));
          ctx.fillStyle = g; ctx.fillRect(x, y, w, h);
        }
        ctx.restore();
      },
      ponto(x, y, r, o) {
        o = o || {};
        const p = new Path2D(), k = 11, ph = R() * TAU;
        for (let i = 0; i <= k; i++) {
          const a = i / k * TAU, rr = r * (1 + .09 * Math.sin(a * 2 + ph) + .05 * (R() - .5));
          if (i) p.lineTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); else p.moveTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr);
        }
        p.closePath();
        ctx.fillStyle = css(cor(o.cor), o.alfa == null ? .95 : o.alfa); ctx.fill(p);
      },
      // filete: faixa horizontal com ondulação periódica (emenda sem costura no azulejo vizinho)
      banda(y, w, x0, x1, o) {
        o = o || {};
        const per = o.periodo || 1, ph1 = R() * TAU, ph2 = R() * TAU, ph3 = R() * TAU, amp = o.amp == null ? .07 : o.amp;
        const top = [], bot = [], N = 48;
        for (let i = 0; i <= N; i++) {
          const x = x0 + (x1 - x0) * i / N, u = x / per * TAU;
          const wob = Math.sin(u + ph1) * .55 + Math.sin(2 * u + ph2) * .3 + Math.sin(3 * u + ph3) * .15;
          const yy = y + w * amp * Math.sin(u + ph3);
          top.push([x, yy - w / 2 * (1 + amp * 2.2 * wob)]); bot.push([x, yy + w / 2 * (1 - amp * 1.6 * wob)]);
        }
        const p = new Path2D(); p.moveTo(top[0][0], top[0][1]);
        for (const q of top) p.lineTo(q[0], q[1]);
        for (let i = bot.length - 1; i >= 0; i--) p.lineTo(bot[i][0], bot[i][1]);
        p.closePath();
        ctx.fillStyle = css(cor(o.cor), o.alfa == null ? .9 : o.alfa); ctx.fill(p);
      },
      // filete vertical (para as laterais das molduras)
      bandaV(x, w, y0, y1, o) {
        ctx.save(); ctx.transform(0, 1, 1, 0, 0, 0); P.banda(x, w, y0, y1, o); ctx.restore();
      },
    };
    return P;
  }

  function riscador(ctx, cores, px) {
    const pontilha = (path) => {
      ctx.save();
      ctx.setLineDash([.001, 3.3 * px]); ctx.lineCap = 'round'; ctx.lineWidth = 1.35 * px;
      ctx.strokeStyle = css(cores.risco); ctx.stroke(path);
      ctx.restore();
    };
    const linha = (pts) => { const p = new Path2D(); p.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) p.lineTo(pts[i][0], pts[i][1]); return p; };
    return {
      R: rng(9),
      risco: true,
      traco(pts) { pontilha(linha(pts)); },
      aguada(path) { pontilha(path); },
      ponto(x, y, r) { if (r < 2.4 * px) return; const p = new Path2D(); p.arc(x, y, r, 0, TAU); pontilha(p); },
      banda(y, w, x0, x1) { pontilha(linha([[x0, y], [x1, y]])); },
      bandaV(x, w, y0, y1) { pontilha(linha([[x, y0], [x, y1]])); },
    };
  }

  /* ---------- motivos (em unidades de azulejo: 1 = um azulejo) ---------- */
  // pétala amendoada com aguada, contorno nos dois lados e nervura
  function petala(pen, cx, cy, ang, len, wid, o) {
    o = o || {};
    const c = Math.cos(ang), s = Math.sin(ang), nx = -s, ny = c, b0 = o.base0 == null ? .16 : o.base0;
    const bx = cx + c * len * b0, by = cy + s * len * b0, tx = cx + c * len, ty = cy + s * len;
    const curva = o.curva || 0; // ponta virada
    const tx2 = tx + nx * len * curva, ty2 = ty + ny * len * curva;
    const L1 = [bx, by, bx + c * len * .2 + nx * wid, by + s * len * .2 + ny * wid, tx2 - c * len * .3 + nx * wid * .8, ty2 - s * len * .3 + ny * wid * .8, tx2, ty2];
    const L2 = [bx, by, bx + c * len * .2 - nx * wid, by + s * len * .2 - ny * wid, tx2 - c * len * .3 - nx * wid * .55, ty2 - s * len * .3 - ny * wid * .55, tx2, ty2];
    const path = new Path2D();
    path.moveTo(bx, by);
    path.bezierCurveTo(L1[2], L1[3], L1[4], L1[5], tx2, ty2);
    path.bezierCurveTo(L2[4], L2[5], L2[2], L2[3], bx, by);
    path.closePath();
    const xs = [bx, tx2, L1[2], L1[4], L2[2], L2[4]], ys = [by, ty2, L1[3], L1[5], L2[3], L2[5]];
    const x0 = Math.min.apply(null, xs), y0 = Math.min.apply(null, ys);
    pen.aguada(path, [x0, y0, Math.max.apply(null, xs) - x0, Math.max.apply(null, ys) - y0], { alfa: o.leve ? .32 : .5, manchas: 3 });
    const lw = o.contorno == null ? Math.max(.016, wid * .2) : o.contorno;
    if (!o.leve || o.contornar) {
      pen.traco(bez(L1, 18), lw, { a: .08, b: .6, base: .15 });
      pen.traco(bez(L2, 18), lw * .8, { a: .08, b: .6, base: .15 });
    }
    if (o.nervura !== false) {
      const mx = bx + c * len * .72 + nx * len * curva * .5, my = by + s * len * .72 + ny * len * curva * .5;
      pen.traco(bez([bx, by, bx + c * len * .25, by + s * len * .25, mx - c * len * .2, my - s * len * .2, mx, my], 12), lw * .75, { a: .05, b: .8, base: .05 });
    }
  }
  function lobo(pen, cx, cy, r, o) {
    const p = new Path2D(); p.arc(cx, cy, r, 0, TAU);
    pen.aguada(p, [cx - r, cy - r, 2 * r, 2 * r], { alfa: (o && o.alfa) || .4, manchas: 2 });
    pen.traco(espiral(cx, cy, r * .98, r * .98, -2.3, 1.2, 20), r * .32, { a: .1, b: .5, base: .2 });
  }
  function voluta(pen, g, w, o) {
    pen.traco(g.pts, w, Object.assign({ a: .1, b: .62, base: .1 }, o || {}));
    pen.ponto(g.olho[0], g.olho[1], w * .62);
  }

  // friso: roseta no meio, quadrifólias partidas nas emendas, volutas entre elas, filetes em cima e embaixo
  function motivoFriso(pen) {
    pen.banda(.065, .058, -.1, 1.1, { amp: .09 });
    pen.banda(.935, .058, -.1, 1.1, { amp: .09 });
    pen.banda(.138, .024, -.1, 1.1, { cor: 'm', amp: .12, alfa: .95 });
    pen.banda(.862, .024, -.1, 1.1, { cor: 'm', amp: .12, alfa: .95 });
    for (const x of [0, 1]) {
      for (let k = 0; k < 4; k++) { const a = k * TAU / 4; lobo(pen, x + Math.cos(a) * .082, .5 + Math.sin(a) * .082, .056); }
      pen.ponto(x, .5, .036); pen.ponto(x, .5, .018, { cor: 'm' });
    }
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
      const g = gavinha(.5 + sx * .1, .5 + sy * .16, Math.atan2(sy * .14, sx * .1), .15, .07, .95, sx * sy > 0 ? 1 : -1);
      voluta(pen, g, .048);
      petala(pen, .5 + sx * .2, .5 + sy * .3, Math.atan2(sy * .5, sx * 1), .13, .045, { leve: true, contornar: true, nervura: false });
    }
    for (let k = 0; k < 4; k++) petala(pen, .5, .5, k * TAU / 4 - TAU / 8, .2, .07, { leve: true, nervura: false });
    for (let k = 0; k < 4; k++) petala(pen, .5, .5, k * TAU / 4 - TAU / 4, .31, .1, { curva: .04 });
    pen.ponto(.5, .5, .066); pen.ponto(.5, .5, .03, { cor: 'm' });
  }

  // cantoneira (canto de dentro da moldura): voluta ao longo de cada lado, folha de acanto na diagonal
  function motivoCanto(pen) {
    const gTopo = gavinha(.16, .13, -.04, .52, .085, 1.0, 1);
    const gLado = gavinha(.13, .16, Math.PI / 2 + .04, .52, .085, 1.0, -1);
    voluta(pen, gTopo, .075);
    voluta(pen, gLado, .075);
    petala(pen, .16, .16, TAU / 8, .62, .15, { curva: .06 });
    petala(pen, .2, .2, TAU / 8 - .55, .34, .07, { leve: true, contornar: true });
    petala(pen, .2, .2, TAU / 8 + .55, .34, .07, { leve: true, contornar: true });
    pen.ponto(.13, .13, .085); pen.ponto(.13, .13, .04, { cor: 'm' });
    pen.ponto(gTopo.olho[0], gTopo.olho[1], .03, { cor: 'm' });
    pen.ponto(gLado.olho[0], gLado.olho[1], .03, { cor: 'm' });
  }

  // florão: leque de acanto com volutas para os lados (caixa 1,4 × 0,6)
  function motivoFlorao(pen) {
    const bx = .7, by = .52;
    petala(pen, bx, by, -Math.PI / 2, .44, .13, { base0: .05, curva: .03 });
    petala(pen, bx, by, -Math.PI / 2 - .62, .36, .1, { base0: .05, curva: -.08 });
    petala(pen, bx, by, -Math.PI / 2 + .62, .36, .1, { base0: .05, curva: .08 });
    petala(pen, bx, by, -Math.PI / 2 - 1.15, .26, .07, { base0: .05, leve: true, contornar: true, curva: -.1 });
    petala(pen, bx, by, -Math.PI / 2 + 1.15, .26, .07, { base0: .05, leve: true, contornar: true, curva: .1 });
    const gE = gavinha(bx - .06, by, Math.PI + .1, .38, .07, .9, 1);
    const gD = gavinha(bx + .06, by, -.1, .38, .07, .9, -1);
    voluta(pen, gE, .05); voluta(pen, gD, .05);
    pen.ponto(gE.olho[0], gE.olho[1], .02, { cor: 'm' }); pen.ponto(gD.olho[0], gD.olho[1], .02, { cor: 'm' });
    pen.ponto(bx, by, .05); pen.ponto(bx, by - .005, .024, { cor: 'm' });
  }

  // voluta lateral da cartela (caixa 0,5 × 0,9), aberta para a direita
  function motivoVoluta(pen) {
    const cima = gavinha(.46, .06, Math.PI - .5, .2, .11, .85, -1);
    const baixo = gavinha(.46, .84, Math.PI + .5, .2, .11, .85, 1);
    voluta(pen, cima, .075); voluta(pen, baixo, .075);
    pen.traco(bez([.4, .1, .12, .25, .12, .65, .4, .8], 24), .05, { a: .15, b: .15, base: .3 });
    petala(pen, .36, .45, Math.PI, .26, .08, { leve: true, contornar: true });
    pen.ponto(.2, .45, .05); pen.ponto(.2, .45, .024, { cor: 'm' });
    pen.ponto(cima.olho[0], cima.olho[1], .028, { cor: 'm' });
    pen.ponto(baixo.olho[0], baixo.olho[1], .028, { cor: 'm' });
  }

  /* ---------- molduras em nove partes (border-image) ---------- */
  /* ---------- grão ---------- */
  function grao(c, forca, seed, periodico) {
    const ctx = c.getContext('2d'), w = c.width, h = c.height;
    const img = ctx.getImageData(0, 0, w, h), d = img.data, R = rng(seed || 5);
    const G = 6, grid = new Float32Array(G * G);
    for (let i = 0; i < grid.length; i++) grid[i] = R();
    for (let y = 0; y < h; y++) {
      const gy = y / h * G, iy = Math.floor(gy), fy = gy - iy;
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4; if (!d[i + 3]) continue;
        const gx = x / w * G, ix = Math.floor(gx), fx = gx - ix;
        const x1 = (ix + 1) % G, y1 = (iy + 1) % G, x0 = ix % G, y0 = iy % G;
        const lo = (grid[y0 * G + x0] * (1 - fx) + grid[y0 * G + x1] * fx) * (1 - fy) + (grid[y1 * G + x0] * (1 - fx) + grid[y1 * G + x1] * fx) * fy;
        const k = 1 - forca * (.5 * lo + .5 * R());
        d[i + 3] = Math.max(0, Math.min(255, d[i + 3] * k));
      }
    }
    ctx.putImageData(img, 0, 0);
    void periodico;
  }

  /* ---------- azulejos do fundo: vidrado, almofada, brilho, craquelê, rejunte ---------- */
  function azulejos(T, cores, dpr, seed) {
    const n = 4, c = tela(T * n, T * n), ctx = c.getContext('2d'), R = rng(seed || 11);
    const v = cores.vidrado, sb = cores.sombra;
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const x = i * T, y = j * T, d = (R() - .5) * 5, q = (R() - .5) * 3;
      ctx.fillStyle = css([v[0] + d + q, v[1] + d, v[2] + d - q * .6]);
      ctx.fillRect(x, y, T + 1, T + 1);
      // almofada: o vidrado engrossa e escurece perto das bordas do azulejo
      const g = ctx.createRadialGradient(x + T * .5, y + T * .48, T * .3, x + T * .5, y + T * .5, T * .78);
      g.addColorStop(0, css(sb, 0)); g.addColorStop(1, css(sb, cores.almofada));
      ctx.fillStyle = g; ctx.fillRect(x, y, T, T);
      // brilho do vidrado
      const bx = x + T * (.26 + R() * .14), by = y + T * (.2 + R() * .12);
      const s = ctx.createRadialGradient(bx, by, 0, bx, by, T * .62);
      s.addColorStop(0, 'rgba(255,255,255,' + cores.brilho + ')'); s.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = s; ctx.fillRect(x, y, T, T);
      // craquelê: fissuras finas no vidrado
      const nc = R() < .6 ? 1 + Math.floor(R() * 3) : 0;
      ctx.lineWidth = Math.max(.6, dpr * .45); ctx.strokeStyle = css(sb, cores.craquele);
      for (let k = 0; k < nc; k++) {
        let px = x + R() * T, py = y + R() * T, a = R() * TAU;
        ctx.beginPath(); ctx.moveTo(px, py);
        const steps = 4 + Math.floor(R() * 6);
        for (let m = 0; m < steps; m++) { a += (R() - .5) * 1.1; px += Math.cos(a) * T * .07; py += Math.sin(a) * T * .07; ctx.lineTo(px, py); }
        ctx.stroke();
      }
      // pintas de ferro
      const np = Math.floor(R() * 3);
      for (let k = 0; k < np; k++) { ctx.fillStyle = css(sb, .18 + R() * .15); ctx.beginPath(); ctx.arc(x + R() * T, y + R() * T, Math.max(.5, dpr * (.3 + R() * .5)), 0, TAU); ctx.fill(); }
    }
    // rejunte e o boleado da borda
    const gw = Math.max(1, Math.round(dpr * 1.25));
    for (let k = 0; k < n; k++) {
      const p = Math.round(k * T);
      ctx.fillStyle = 'rgba(255,255,255,' + (cores.brilho * .9).toFixed(3) + ')';
      ctx.fillRect(p + gw, 0, Math.max(1, dpr * .6), T * n); ctx.fillRect(0, p + gw, T * n, Math.max(1, dpr * .6));
      ctx.fillStyle = css(sb, .07);
      ctx.fillRect(p - Math.max(1, dpr * .8), 0, Math.max(1, dpr * .8), T * n); ctx.fillRect(0, p - Math.max(1, dpr * .8), T * n, Math.max(1, dpr * .8));
      ctx.fillStyle = css(cores.rejunte);
      ctx.fillRect(p, 0, gw, T * n); ctx.fillRect(0, p, T * n, gw);
    }
    return c;
  }

  /* ---------- o jogo de ornamentos ---------- */
  function desenha(S, w, h, motivo, cores, modo, seed, px) {
    const c = tela(S * w, S * h), ctx = c.getContext('2d');
    ctx.setTransform(S, 0, 0, S, 0, 0);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    const pen = modo === 'risco' ? riscador(ctx, cores, 1 / S * (px || 1)) : pincel(ctx, cores, seed);
    motivo(pen, ctx);
    if (modo !== 'risco') grao(c, .2, seed + 1);
    return c;
  }

  // moldura de painel: filete grosso de cobalto, mostarda, filete fino; quina de bloco com botão de mostarda
  function moldura(S, cores, seed) {
    const cw = .34, W = 2 * cw + 1, c = tela(S * W, S * W), ctx = c.getContext('2d');
    ctx.setTransform(S, 0, 0, S, 0, 0);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    const pen = pincel(ctx, cores, seed);
    const q = .3;  // lado do bloco da quina
    const lista = [{ d: .08, w: .082 }, { d: .172, w: .03, cor: 'm', amp: .1 }, { d: .238, w: .024 }];
    for (const f of lista) {
      const o = { cor: f.cor, amp: f.amp == null ? .045 : f.amp, periodo: 1 };
      pen.banda(f.d, f.w, q - .01, W - q + .01, o);
      pen.banda(W - f.d, f.w, q - .01, W - q + .01, o);
      pen.bandaV(f.d, f.w, q - .01, W - q + .01, o);
      pen.bandaV(W - f.d, f.w, q - .01, W - q + .01, o);
    }
    for (const [x, y] of [[0, 0], [W, 0], [0, W], [W, W]]) {
      const qx = x ? x - q - .03 : .03, qy = y ? y - q - .03 : .03;
      const p = new Path2D(); p.rect(qx, qy, q, q);
      pen.aguada(p, [qx, qy, q, q], { alfa: .62, manchas: 2, borda: .04 });
      const cx = qx + q / 2, cy = qy + q / 2;
      for (let k = 0; k < 4; k++) { const a = k * TAU / 4 + TAU / 8; pen.ponto(cx + Math.cos(a) * q * .3, cy + Math.sin(a) * q * .3, .026, { alfa: .9 }); }
      pen.ponto(cx, cy, .062, { cor: 'm' });
    }
    grao(c, .16, seed + 1);
    return { canvas: c, corte: Math.round(cw * S), largura: cw };
  }

  // cartela: moldura com quinas recortadas (arcos côncavos concêntricos) e um trevo em cada recorte
  function cartela(S, cores, seed, opt) {
    opt = opt || {};
    const k = .5, m = .5, W = 2 * k + m, c = tela(S * W, S * W), ctx = c.getContext('2d');
    ctx.setTransform(S, 0, 0, S, 0, 0);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    const pen = pincel(ctx, cores, seed);
    const lin = [{ d: .07, w: .078 }, { d: .155, w: .03, cor: 'm', amp: .1 }, { d: .212, w: .021 }];
    const R0 = .2;
    for (const f of lin) {
      const o = { cor: f.cor, amp: f.amp == null ? .05 : f.amp, periodo: m * 3 };
      const r = R0 + f.d, x0 = Math.sqrt(r * r - f.d * f.d), th = Math.asin(f.d / r);
      pen.banda(f.d, f.w, x0 - .005, W - x0 + .005, o);
      pen.banda(W - f.d, f.w, x0 - .005, W - x0 + .005, o);
      pen.bandaV(f.d, f.w, x0 - .005, W - x0 + .005, o);
      pen.bandaV(W - f.d, f.w, x0 - .005, W - x0 + .005, o);
      for (const [cx, cy, a0] of [[0, 0, 0], [W, 0, Math.PI / 2], [W, W, Math.PI], [0, W, -Math.PI / 2]]) {
        pen.traco(espiral(cx, cy, r, r, a0 + th, a0 + Math.PI / 2 - th, 18), f.w, { a: 0, b: 0, base: 1, tremor: .05, cor: f.cor, carga: false, alfa: f.cor === 'm' ? .95 : .9 });
      }
    }
    for (const [cx, cy, a] of [[0, 0, TAU / 8], [W, 0, 3 * TAU / 8], [W, W, 5 * TAU / 8], [0, W, 7 * TAU / 8]]) {
      const bx = cx + Math.cos(a) * (R0 + .045), by = cy + Math.sin(a) * (R0 + .045), out = a + Math.PI;
      petala(pen, bx, by, out, .17, .045, { base0: .02, nervura: false, contorno: .012 });
      petala(pen, bx, by, out - .75, .11, .032, { base0: .02, leve: true, contornar: true, nervura: false, contorno: .01 });
      petala(pen, bx, by, out + .75, .11, .032, { base0: .02, leve: true, contornar: true, nervura: false, contorno: .01 });
      pen.ponto(bx, by, .028, { cor: 'm' });
    }
    grao(c, .14, seed + 1);
    return { canvas: c, corte: Math.round(k * S), quina: k };
  }

  // placa (botão): corpo em aguada densa de cobalto, cantos recortados, filete de reserva e mostarda
  function placa(S, cores, seed, modo) {
    const k = .32, m = .4, W = 2 * k + m, H = W, c = tela(S * W, S * H), ctx = c.getContext('2d');
    ctx.setTransform(S, 0, 0, S, 0, 0);
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    const pen = pincel(ctx, cores, seed);
    const rq = .11, i0 = .025;
    const corpo = new Path2D();
    corpo.moveTo(i0 + rq, i0);
    corpo.lineTo(W - i0 - rq, i0); corpo.arc(W - i0, i0, rq, Math.PI, Math.PI / 2, true);
    corpo.lineTo(W - i0, H - i0 - rq); corpo.arc(W - i0, H - i0, rq, -Math.PI / 2, -Math.PI, true);
    corpo.lineTo(i0 + rq, H - i0); corpo.arc(i0, H - i0, rq, 0, -Math.PI / 2, true);
    corpo.lineTo(i0, i0 + rq); corpo.arc(i0, i0, rq, Math.PI / 2, 0, true);
    corpo.closePath();
    if (modo === 'clara') {
      ctx.save(); ctx.lineWidth = .05; ctx.strokeStyle = css(cores.cobalto, .9); ctx.stroke(corpo); ctx.restore();
      const dentro = new Path2D(); dentro.rect(.13, .13, W - .26, H - .26);
      ctx.save(); ctx.lineWidth = .022; ctx.strokeStyle = css(cores.mostarda, .95); ctx.stroke(dentro); ctx.restore();
    } else {
      const forte = modo === 'escura';
      ctx.fillStyle = css(forte ? cores.cobaltoFundo : cores.cobalto, forte ? 1 : .96); ctx.fill(corpo);
      ctx.save(); ctx.clip(corpo);
      const R = pen.R;
      for (let i = 0; i < 5; i++) {
        const cx = R() * W, cy = R() * H, r = (.3 + R() * .5) * W;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        g.addColorStop(0, css(cores.cobaltoFundo, .12 + R() * .12)); g.addColorStop(1, css(cores.cobaltoFundo, 0));
        ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
      }
      ctx.lineWidth = .03; ctx.strokeStyle = css(cores.cobaltoFundo, .55); ctx.stroke(corpo);
      ctx.restore();
      // filete de reserva: vidrado sem tinta (apaga o cobalto)
      const res = new Path2D(); res.rect(.1, .1, W - .2, H - .2);
      ctx.save(); ctx.globalCompositeOperation = 'destination-out'; ctx.lineWidth = .026; ctx.strokeStyle = 'rgba(0,0,0,.92)'; ctx.stroke(res); ctx.restore();
      const mus = new Path2D(); mus.rect(.15, .15, W - .3, H - .3);
      ctx.save(); ctx.lineWidth = .016; ctx.strokeStyle = css(cores.mostarda, .9); ctx.stroke(mus); ctx.restore();
    }
    grao(c, modo === 'clara' ? .2 : .07, seed + 1);
    return { canvas: c, corte: Math.round(k * S) };
  }

  // seletor (botão do controle de demãos): quadrado de quatro pinceladas e filete de mostarda
  function seletor(S, cores, seed) {
    const c = tela(S, S), ctx = c.getContext('2d');
    ctx.setTransform(S, 0, 0, S, 0, 0);
    const pen = pincel(ctx, cores, seed), e = .075, w = .1;
    pen.traco(bez([.03, e, .35, e - .01, .65, e + .01, .97, e], 16), w, { a: .06, b: .12, base: .55 });
    pen.traco(bez([1 - e, .03, 1 - e + .01, .35, 1 - e - .01, .65, 1 - e, .97], 16), w, { a: .06, b: .12, base: .55 });
    pen.traco(bez([.97, 1 - e, .65, 1 - e - .01, .35, 1 - e + .01, .03, 1 - e], 16), w, { a: .06, b: .12, base: .55 });
    pen.traco(bez([e, .97, e - .01, .65, e + .01, .35, e, .03], 16), w, { a: .06, b: .12, base: .55 });
    const p = new Path2D(); p.rect(.17, .17, .66, .66);
    ctx.lineWidth = .03; ctx.strokeStyle = css(cores.mostarda, .95); ctx.stroke(p);
    grao(c, .15, seed + 1);
    return c;
  }

  function url(c, tipo) { return 'url("' + c.toDataURL(tipo || 'image/png', .92) + '")'; }

  /* T: lado do azulejo em px de CSS; dpr: densidade da tela; cores: {vidrado, rejunte, sombra, cobalto,
     cobaltoFundo, mostarda, risco, brilho, almofada, craquele} em [r,g,b] (brilho/almofada/craquele em alfa) */
  function kit(T, dpr, cores) {
    const S = T * dpr, t0 = (root.performance ? performance.now() : 0);
    const out = {};
    out.azulejos = url(azulejos(S, cores, dpr, 11), 'image/jpeg');
    out.friso = url(desenha(S, 1, 1, motivoFriso, cores, 'pinta', 21));
    out.frisoRisco = url(desenha(S, 1, 1, motivoFriso, cores, 'risco', 21, dpr));
    out.canto = url(desenha(S, 1, 1, motivoCanto, cores, 'pinta', 31));
    out.florao = url(desenha(S, 1.4, .6, motivoFlorao, cores, 'pinta', 41));
    out.voluta = url(desenha(S, .5, .9, motivoVoluta, cores, 'pinta', 51));
    const mo = moldura(S, cores, 61); out.moldura = url(mo.canvas); out.molduraCorte = mo.corte;
    const ca = cartela(S, cores, 71); out.cartela = url(ca.canvas); out.cartelaCorte = ca.corte;
    const pl = placa(S, cores, 81, 'pinta'); out.placa = url(pl.canvas); out.placaCorte = pl.corte;
    out.placaEscura = url(placa(S, cores, 81, 'escura').canvas);
    out.placaClara = url(placa(S, cores, 91, 'clara').canvas);
    out.seletor = url(seletor(S, cores, 97));
    out.ms = root.performance ? Math.round(performance.now() - t0) : 0;
    return out;
  }

  root.Pincel = { kit, azulejos, desenha, moldura, cartela, placa, seletor, motivoFriso, motivoCanto, motivoFlorao, motivoVoluta, rng };
})(typeof self !== 'undefined' ? self : globalThis);
