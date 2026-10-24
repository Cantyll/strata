"""Servidor HTTP de uma pasta com pedidos de faixa (Range) e um teto de banda para todas as conexões
juntas, para simular um download do Hub numa rede de N Mbit/s.

    python3 tests/dist/servidor_faixas.py <pasta> --porta 8765 --mbit 300 [--rtt-ms 20]
"""
import argparse, os, re, threading, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("pasta")
ap.add_argument("--porta", type=int, default=8765)
ap.add_argument("--mbit", type=float, default=0, help="teto de banda somado (0 = sem teto)")
ap.add_argument("--rtt-ms", type=float, default=0, help="atraso antes de cada resposta")
a = ap.parse_args()
PASTA = os.path.abspath(a.pasta)


class Balde:
    """Fichas de bytes por segundo, divididas entre todas as conexões."""
    def __init__(self, bps):
        self.bps, self.t, self.l = bps, time.monotonic(), threading.Lock()

    def tira(self, n):
        if not self.bps:
            return
        with self.l:
            self.t = max(self.t, time.monotonic()) + n / self.bps
            espera = self.t - time.monotonic()
        if espera > 0:
            time.sleep(espera)


balde = Balde(a.mbit * 1e6 / 8)


class H(SimpleHTTPRequestHandler):
    def __init__(self, *x, **k):
        super().__init__(*x, directory=PASTA, **k)

    def log_message(self, *x):
        pass

    def do_GET(self):
        p = self.translate_path(self.path)
        if not os.path.isfile(p):
            return self.send_error(404)
        tam = os.path.getsize(p)
        ini, fim, cod = 0, tam - 1, 200
        m = re.match(r"bytes=(\d+)-(\d*)$", self.headers.get("Range", ""))
        if m:
            ini, fim, cod = int(m.group(1)), min(int(m.group(2) or tam - 1), tam - 1), 206
            if ini > fim:
                return self.send_error(416)
        if a.rtt_ms:
            time.sleep(a.rtt_ms / 1000)
        self.send_response(cod)
        self.send_header("Content-Length", str(fim - ini + 1))
        self.send_header("Accept-Ranges", "bytes")
        if cod == 206:
            self.send_header("Content-Range", f"bytes {ini}-{fim}/{tam}")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(ini)
            falta = fim - ini + 1
            while falta:
                b = f.read(min(falta, 256 << 10))
                balde.tira(len(b))
                self.wfile.write(b)
                falta -= len(b)


ThreadingHTTPServer.daemon_threads = True
ThreadingHTTPServer(("127.0.0.1", a.porta), H).serve_forever()
