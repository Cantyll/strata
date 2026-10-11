"""Monta dist/strata-lab.html, o protótipo em um único arquivo.

Injeta os três scripts de src/ (codec, pipeline e pincel) nos marcadores <!--CORE-->,
<!--LAB--> e <!--PINCEL--> da página src/strata-lab.src.html.
Uso: python3 build.py
"""
import pathlib

raiz = pathlib.Path(__file__).resolve().parent
src = raiz / 'src'
pagina = (src / 'strata-lab.src.html').read_text(encoding='utf-8')

partes = [
    ('CORE', 'strata-core.js', 'strata-core'),
    ('LAB', 'strata-lab.js', 'strata-lab'),
    ('PINCEL', 'strata-pincel.js', 'strata-pincel'),
]

saida = pagina
for marcador, arquivo, ident in partes:
    js = (src / arquivo).read_text(encoding='utf-8')
    # um "</script" dentro do código fecharia a tag antes da hora
    assert '</script' not in js, arquivo
    tag = '<!--' + marcador + '-->'
    assert saida.count(tag) == 1, tag
    saida = saida.replace(tag, '<script id="' + ident + '">\n' + js + '</script>')

destino = raiz / 'dist' / 'strata-lab.html'
destino.parent.mkdir(exist_ok=True)
destino.write_text(saida, encoding='utf-8')
print(destino.relative_to(raiz), len(saida.encode('utf-8')), 'bytes')
