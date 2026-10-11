// Teste do pipeline completo do protótipo (o mesmo que roda no Web Worker da página), no Node 18+.
// Gera o modelo sintético (semente 7), converte para .strata, volta para .safetensors e confere:
// SHA-256 idêntico, tamanho, erro caindo a cada estrato e o filho guardado como diferença do pai.
// Uso: node tests/test-lab.mjs
import fs from 'node:fs';

for (const f of ['strata-core.js', 'strata-lab.js']) {
  new Function(fs.readFileSync(new URL('../src/' + f, import.meta.url), 'utf8'))();
}
const Lab = globalThis.StrataLab;
const pct = (x, d = 1) => (100 * x).toFixed(d) + '%';
let falhas = 0;
const confere = (ok, oQue) => { console.log((ok ? 'ok    ' : 'FALHOU') + '  ' + oQue); if (!ok) falhas++; };

await Lab.prepare({ seed: 7, params: 1000000 });
const r = await Lab.convert(() => {});

console.log(`modelo sintético BF16: ${r.nParams.toLocaleString('pt-BR')} pesos · safetensors ${r.fileBytes} bytes · .strata ${r.strataBytes} bytes (${pct(r.strataBytes / r.fileBytes)})`);
console.log('SHA-256 original:', r.shaOrig);
console.log('SHA-256 de volta:', r.shaBack);
console.log('\nestrato | bytes   | bits/peso acumulado | prefixo do arquivo | erro relativo');
let acc = 0;
r.byStratum.forEach((b, d) => {
  acc += b;
  console.log(`E${d}      | ${String(b).padStart(7)} | ${(8 * acc / r.nFloat).toFixed(2).padStart(19)} | ${pct(r.prefix[d] / r.strataBytes).padStart(18)} | ${d === r.M ? 'zero (exato)' : pct(r.metrics.depth[d], 2)}`);
});
console.log(`referências: INT4 por bloco de 32 = ${pct(r.metrics.int4, 2)} · INT8 por bloco de 32 = ${pct(r.metrics.int8, 2)}`);
console.log(`entropia do expoente: ${r.expEntropy.toFixed(3)} bits`);
console.log(`genealogia: filho sozinho ${r.lineage.standalone} bytes · como diferença do pai ${r.lineage.delta} bytes (${pct(r.lineage.delta / r.lineage.standalone)})\n`);

confere(r.identical && r.shaOrig === r.shaBack, 'volta bit a bit (SHA-256 idêntico)');
confere(r.strataBytes / r.fileBytes < 0.72, '.strata menor que 72% do safetensors');
confere(r.metrics.depth.every((e, d) => d === 0 || e < r.metrics.depth[d - 1]), 'erro cai a cada estrato lido');
confere(r.metrics.depth[1] < r.metrics.int4, 'E8M1 erra menos que o INT4 por bloco');
confere(r.lineage.delta / r.lineage.standalone < 0.3, 'filho como diferença ocupa menos de 30%');
confere(r.prefix.every((p, d) => d === 0 || p > r.prefix[d - 1]), 'cada estrato estende o prefixo do arquivo');

if (falhas) { console.error(`\n${falhas} verificação(ões) falharam`); process.exit(1); }
console.log('\ntudo certo');
