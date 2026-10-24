// O Bit: o guia de todo vídeo do STRATA, o logo e o quadro final (padrão permanente, ver video/README.md).
// Tudo é função do tempo da narração `a` (segundos), sem estado escondido, como o resto do padrão.
//
//   <canvas id="guia" width="1080" height="1920"></canvas>   (por cima do #vp, por baixo do #leg)
//   <script src="../../padrao/bit.js"></script>
//   const guia = Bit.cria($('guia'), { chaves: BK, gestos: GS, boca: BOCA });
//   guia.desenha(a);            // a cada quadro; desenha(a, separa) abre os estratos do corpo (0–1)
//   Bit.logo(ctx, tamanho, progresso)       // o logo de cinco estratos
//   const fim = Bit.final(face.p);          // o quadro final: logo + STRATA + slogan
//   fim.anima(a, t0)                        // entra em t0; devolve 0–1 para apagar o resto da cena
//   BK.push(...Bit.despedida(t0).chaves); GS.push(...Bit.despedida(t0).gestos)  // o Bit desce e acena
//
// chaves: [chegada, x, y, escala, duração do voo]; a primeira é a posição de partida.
// gestos: [início, fim, 'aponta', alvo x, alvo y] ou [início, fim, 'acena'].
// boca: envelope da narração, 30 valores por segundo entre 0 e 1 (monta_tempos.py gera boca.js).
window.Bit = (() => {
  const W = 1080, H = 1920;
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const P = (t, a, d) => clamp((t - a) / d);
  const eo = t => 1 - Math.pow(1 - t, 3);
  const eio = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
  // cores fixas do corpo, de cima para baixo: fecho, níveis 3, 2, 1 e o Q4_K_M na base
  const CORPO = ['#2EE6A6', '#22C6F0', '#3B9BFF', '#5B6CFF', '#F5B83D'];
  const LARG = [96, 128, 140, 128, 104];

  function logo(c, s, prog) {
    const cols = ['#2EE6A6', '#22C6F0', '#3B9BFF', '#5B6CFF', '#F5B83D'], ws = [.34, .48, .64, .82, 1];
    c.clearRect(0, 0, s, s);
    for (let i = 0; i < 5; i++) {
      const k = eo(P(prog, i * .12, .4)), w = s * ws[i] * k, h = s * .13;
      c.fillStyle = cols[i]; c.beginPath(); c.roundRect((s - w) / 2, s * .08 + i * h * 1.32, w, h, h / 2); c.fill();
    }
  }

  function final(parent) {
    const el = document.createElement('div');
    el.className = 'abs';
    el.innerHTML = '<canvas width="260" height="260"></canvas><div style="font:800 116px \'Inter Display\';letter-spacing:6px;margin-top:6px">STRATA</div><div style="font:500 36px Inter;color:#9AA0B4">um arquivo, várias precisões</div>';
    el.style.cssText = 'position:absolute;left:0;top:610px;width:1080px;text-align:center';
    parent.appendChild(el);
    const lc = el.querySelector('canvas').getContext('2d');
    return {
      el,
      anima(a, t0) {
        const lk = eio(P(a, t0, 1.0));
        el.style.opacity = lk;
        el.style.transform = `translateZ(${(1 - lk) * -900}px) rotateY(${(1 - lk) * 120 + 5 * Math.sin(a * .8)}deg)`;
        logo(lc, 260, P(a, t0 + .2, 1.4));
        return lk;
      },
    };
  }

  // o Bit desce para baixo do logo e acena (t0 = início da frase de despedida)
  const despedida = t0 => ({ chaves: [[t0 + .1, 540, 1400, 1.05, 1.3]], gestos: [[t0 + .8, t0 + 3.8, 'acena']] });

  function cria(canvas, { chaves, gestos = [], boca = [] }) {
    const c = canvas.getContext('2d');
    function pos(a) {
      let j = 1;
      while (j < chaves.length && a >= chaves[j][0] - chaves[j][4]) j++;
      const cur = chaves[j - 1], prv = chaves[Math.max(0, j - 2)];
      const p = clamp((a - (cur[0] - cur[4])) / cur[4]);
      if (p >= 1 || j === 1) return { x: cur[1], y: cur[2], s: cur[3], voo: 0 };
      const e = eio(p), d = Math.hypot(cur[1] - prv[1], cur[2] - prv[2]);
      return { x: lerp(prv[1], cur[1], e), y: lerp(prv[2], cur[2], e) - Math.sin(Math.PI * p) * Math.min(140, d * .3), s: lerp(prv[3], cur[3], e), voo: Math.sin(Math.PI * p) };
    }
    function gesto(a) {
      let g = null, k = 0;
      for (const x of gestos) { const kk = Math.min(P(a, x[0], .3), 1 - P(a, x[1] - .3, .3)); if (kk > k) { k = kk; g = x; } }
      return { g, k: eio(clamp(k)) };
    }
    function piscar(a) {
      // piscadas determinísticas a cada ~3–4,5 s
      const n = Math.floor(a / 3.7), base = n * 3.7 + 1.3 * Math.abs(Math.sin(n * 12.9898)), d = a - base;
      return d > 0 && d < .16 ? Math.sin(Math.PI * d / .16) : 0;
    }
    function fala(a) {
      const f = Math.round(a * 30);
      if (f < 0 || f >= boca.length) return 0;
      return Math.max(boca[f], .75 * (boca[f - 1] || 0), .5 * (boca[f - 2] || 0));
    }
    const rr = (x, y, w, h, r) => { c.beginPath(); c.roundRect(x, y, w, h, r); };

    function desenha(a, separa = 0) {
      c.clearRect(0, 0, W, H);
      if (a < -0.4) return;
      const p = pos(a), pa = pos(a - .03), vx = (p.x - pa.x) / .03, vy = (p.y - pa.y) / .03;
      const { g, k } = gesto(a);
      const bob = Math.sin(a * 2.3) * 7 * (1 - p.voo), s = p.s;
      const x = p.x, y = p.y + bob * s;
      const tilt = clamp(vx / 2600, -.35, .35) + .04 * Math.sin(a * 1.1);
      const stz = clamp(Math.abs(vy) / 4000, 0, .14) - clamp(-vy / 6000, 0, .05);
      // sombra no "chão" logo abaixo
      c.save(); c.globalAlpha = .38 * (1 - .5 * p.voo);
      const sg = c.createRadialGradient(x, y + 128 * s, 0, x, y + 128 * s, 80 * s);
      sg.addColorStop(0, 'rgba(0,0,0,.9)'); sg.addColorStop(1, 'rgba(0,0,0,0)');
      c.fillStyle = sg; c.beginPath(); c.ellipse(x, y + 128 * s, 80 * s * (1 - .3 * p.voo), 18 * s, 0, 0, 7); c.fill(); c.restore();
      // alvo do gesto (anel que pulsa) e feixe pontilhado
      let mx = 0, my = 0, alvo = null;
      if (g && g[2] === 'aponta' && k > 0) {
        alvo = [g[3], g[4]];
        const dx = alvo[0] - x, dy = alvo[1] - y, dd = Math.hypot(dx, dy) || 1;
        mx = dx / dd; my = dy / dd;
        c.save(); c.globalAlpha = .55 * k; c.strokeStyle = '#2EE6A6'; c.lineWidth = 3; c.setLineDash([4, 10]);
        c.lineDashOffset = -a * 40;
        c.beginPath(); c.moveTo(x + mx * 112 * s, y + my * 112 * s); c.lineTo(alvo[0] - mx * 26, alvo[1] - my * 26); c.stroke();
        c.setLineDash([]);
        const pr = (a * 1.6) % 1;
        c.globalAlpha = k * (1 - pr) * .8; c.lineWidth = 3;
        c.beginPath(); c.arc(alvo[0], alvo[1], 14 + 26 * pr, 0, 7); c.stroke();
        c.globalAlpha = k; c.fillStyle = '#2EE6A6'; c.beginPath(); c.arc(alvo[0], alvo[1], 7, 0, 7); c.fill();
        c.restore();
      }
      // olhar: para o alvo, para onde voa, ou para quem assiste
      let lx = 0, ly = 0;
      if (alvo) { lx = mx * 6 * k; ly = my * 5 * k; } else if (p.voo > .05) { lx = clamp(vx / 600, -1, 1) * 6; ly = clamp(vy / 600, -1, 1) * 4; }
      c.save(); c.translate(x, y); c.rotate(tilt); c.scale(s * (1 - stz * .6), s * (1 + stz));
      // antena
      c.strokeStyle = '#22C6F0'; c.lineWidth = 5; c.lineCap = 'round';
      c.beginPath(); c.moveTo(0, -78); c.quadraticCurveTo(6 * Math.sin(a * 3), -96, 2 * Math.sin(a * 3), -108); c.stroke();
      const pul = .6 + .4 * Math.sin(a * 5);
      c.save(); c.shadowColor = '#2EE6A6'; c.shadowBlur = 24 * pul; c.fillStyle = '#2EE6A6';
      c.beginPath(); c.arc(2 * Math.sin(a * 3), -114, 9, 0, 7); c.fill(); c.restore();
      // estratos do corpo
      for (let i = 0; i < 5; i++) {
        const hy = -75 + i * 31 + (i - 2) * separa * 22, w = LARG[i] + separa * 10;
        const gr = c.createLinearGradient(0, hy, 0, hy + 27);
        gr.addColorStop(0, CORPO[i]); gr.addColorStop(1, CORPO[i] + 'B0');
        c.save(); c.shadowColor = CORPO[i]; c.shadowBlur = 18 + 20 * separa; c.fillStyle = gr;
        rr(-w / 2 + separa * 8 * Math.sin(a * 2 + i), hy, w, 27, 13.5); c.fill(); c.restore();
        c.fillStyle = 'rgba(255,255,255,.28)'; rr(-w / 2 + 12, hy + 4, w - 24, 5, 2.5); c.fill();
      }
      // visor com olhos e boca
      c.save(); c.globalAlpha = 1 - separa * .85;
      c.fillStyle = 'rgba(8,11,20,.94)'; rr(-56, -40, 112, 68, 22); c.fill();
      c.strokeStyle = 'rgba(255,255,255,.22)'; c.lineWidth = 2; c.stroke();
      const eh = Math.max(2.5, 13 * (1 - piscar(a)));
      c.shadowColor = '#7FF3FF'; c.shadowBlur = 14; c.fillStyle = '#E9FDFF';
      for (const ex of [-22, 22]) { c.beginPath(); c.ellipse(ex + lx, -16 + ly, 8.5, eh, 0, 0, 7); c.fill(); }
      const m = fala(a);
      c.fillStyle = '#2EE6A6'; c.shadowColor = '#2EE6A6';
      const mh = 3 + 12 * m, mw = 20 + 6 * m;
      rr(-mw / 2 + lx * .5, 8 - mh / 2 + 4, mw, mh, Math.min(mh, mw) / 2); c.fill();
      c.restore();
      // mãos
      const mao = (hx, hy) => {
        const gr = c.createRadialGradient(hx - 4, hy - 4, 2, hx, hy, 15);
        gr.addColorStop(0, '#BFE9FF'); gr.addColorStop(1, '#3B9BFF');
        c.save(); c.shadowColor = '#22C6F0'; c.shadowBlur = 12; c.fillStyle = gr; c.beginPath(); c.arc(hx, hy, 14, 0, 7); c.fill(); c.restore();
      };
      let L = [-94, 16 + 5 * Math.sin(a * 2.3 + 1)], R = [94, 16 + 5 * Math.sin(a * 2.3 + 2)];
      if (g && g[2] === 'acena' && k > 0) {
        const ang = -1.1 + .45 * Math.sin(a * 9);
        R = [lerp(R[0], 70 + 40 * Math.cos(ang), k), lerp(R[1], -40 + 40 * Math.sin(ang), k)];
      } else if (alvo) {
        // desfaz a rotação do corpo para apontar no mundo
        const ca = Math.cos(-tilt), sa = Math.sin(-tilt), ux = mx * ca - my * sa, uy = mx * sa + my * ca;
        const tgt = [ux * 118, uy * 118];
        if (ux < 0) L = [lerp(L[0], tgt[0], k), lerp(L[1], tgt[1], k)]; else R = [lerp(R[0], tgt[0], k), lerp(R[1], tgt[1], k)];
      }
      mao(...L); mao(...R);
      c.restore();
    }
    return { desenha, pos };
  }

  return { cria, logo, final, despedida, CORPO, LANE: 965 };
})();
