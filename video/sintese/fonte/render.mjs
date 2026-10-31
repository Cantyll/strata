// Playwright: o pacote instalado no projeto ou o do contêiner do Claude Code.
const { chromium } = await import('playwright').catch(() => import('/opt/node-tools/node_modules/playwright/index.mjs'));
import { spawn } from 'child_process';
import { resolve } from 'path';
// PAGINA=anim.html node render.mjs stills 1,20,40   |   PAGINA=anim.html node render.mjs video <quadro0> <quadro1> saida.mp4
const [,, mode, a0, a1, out] = process.argv;
const FPS = 30;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
page.on('pageerror', e => { console.error('PAGEERR', e.message); process.exit(1); });
await page.goto('file://' + resolve(process.env.PAGINA || 'anim.html'));
await page.waitForFunction(() => window.ready === true);
if (mode === 'stills') {
  for (const t of a0.split(',').map(Number)) {
    await page.evaluate(t => draw(t), t);
    await page.screenshot({ path: `still_${t.toFixed(1)}.png` });
  }
} else {
  const f0 = +a0, f1 = +a1;
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'mjpeg', '-r', String(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', String(FPS), out], { stdio: ['pipe', 'inherit', 'inherit'] });
  for (let f = f0; f < f1; f++) {
    await page.evaluate(t => draw(t), f / FPS);
    const buf = await page.screenshot({ type: 'jpeg', quality: 93 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
}
await browser.close();
