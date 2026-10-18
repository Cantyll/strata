"""Confere o binding contra os vetores públicos (tests/vetores) e contra o próprio safetensors.

    cd leitor-py && maturin develop --release && python -m pytest tests -q
"""
import hashlib
import json
import pathlib

import pytest

import strata_ref

VET = pathlib.Path(__file__).resolve().parents[2] / "tests" / "vetores"
MAN = json.loads((VET / "vetores.json").read_text())["vetores"]
OK = [v for v in MAN if v["espera"] == "ok"]
RUIM = [v for v in MAN if v["espera"] == "recusa"]
sha = lambda b: hashlib.sha256(b).hexdigest()


@pytest.mark.parametrize("v", OK, ids=lambda v: v["arquivo"])
def test_volta_e_prefixos(v):
    dados = (VET / v["arquivo"]).read_bytes()
    assert sha(strata_ref.para_safetensors(VET / v["arquivo"])) == v["sha256_safetensors"]
    assert sha(strata_ref.para_safetensors(dados)) == v["sha256_safetensors"]  # também de bytes
    if "id" in v:
        f = strata_ref.abre(dados, exigir_integridade=True, id_esperado=v["id"])
        assert f.id == v["id"] and f.integridade
        with pytest.raises(strata_ref.StrataError):
            strata_ref.abre(dados, id_esperado="0" * 64)
    so_cabecalho = strata_ref.abre_cabecalho(VET / v["arquivo"])
    assert so_cabecalho.tamanho_lido == 12 + int.from_bytes(dados[8:12], "little")
    assert strata_ref.abre_cabecalho(dados).id == strata_ref.abre(dados).id
    with pytest.raises(strata_ref.StrataError):
        so_cabecalho.tensor(so_cabecalho.nomes()[0])
    for p in v.get("prefixos", []):
        d, n = p["profundidade"], p["bytes"]
        assert strata_ref.abre(dados).bytes_ate(d) == so_cabecalho.bytes_ate(d) == n
        f = strata_ref.abre(dados[:n])
        ts = f.todos(d)
        assert sha(b"".join(ts[k].tobytes() for k in f.nomes())) == p["sha256_tensores"]
        # tensor a tensor dá o mesmo que todos de uma vez
        assert all(f.tensor(k, d).tobytes() == ts[k].tobytes() for k in f.nomes())
        with pytest.raises(strata_ref.StrataError):
            strata_ref.abre(dados[: n - 1]).todos(d)


@pytest.mark.parametrize("v", RUIM, ids=lambda v: v["arquivo"])
def test_recusa(v):
    with pytest.raises(strata_ref.StrataError):
        strata_ref.para_safetensors(VET / v["arquivo"])


def test_exigir_integridade():
    with pytest.raises(strata_ref.StrataError):
        strata_ref.abre(VET / "ok-sem-hashes.strata", exigir_integridade=True)
    strata_ref.abre(VET / "ok-sem-hashes.strata")


@pytest.mark.parametrize("v", OK, ids=lambda v: v["arquivo"])
@pytest.mark.parametrize("framework", ["numpy", "pt"])
def test_igual_ao_safetensors(v, framework, tmp_path):
    """load_file e safe_open dão exatamente os tensores que o safetensors lê do original."""
    if framework == "pt":
        pytest.importorskip("torch")
        from safetensors.torch import load_file as st_load
        igual = lambda a, b: a.dtype == b.dtype and a.shape == b.shape and a.view(-1).view(__import__("torch").uint8).equal(b.view(-1).view(__import__("torch").uint8))
    else:
        pytest.importorskip("ml_dtypes")
        from safetensors.numpy import load_file as st_load
        igual = lambda a, b: a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    pytest.importorskip("safetensors")
    st = tmp_path / "x.safetensors"
    st.write_bytes(strata_ref.para_safetensors(VET / v["arquivo"]))
    try:
        ref = st_load(str(st))
    except Exception as e:  # o safetensors.numpy não lê BF16
        pytest.skip(f"safetensors não lê este arquivo em {framework}: {e}")
    nosso = strata_ref.load_file(VET / v["arquivo"], framework=framework)
    assert nosso.keys() == ref.keys()
    assert all(igual(nosso[k], ref[k]) for k in ref)
    with strata_ref.safe_open(VET / v["arquivo"], framework=framework) as f:
        assert set(f.keys()) == set(ref)
        assert all(igual(f.get_tensor(k), ref[k]) for k in ref)


def test_sem_copia():
    """O array do numpy aponta para a memória do Bloco, e o Bloco vive enquanto o array viver."""
    np = pytest.importorskip("numpy")
    f = strata_ref.abre(VET / "ok-tipos-mistos.strata")
    b = f.tensor(f.nomes()[0])
    a = np.frombuffer(b, dtype=np.uint8)
    assert a.base is not None and len(a) == len(b)
    del b, f
    assert a.sum() >= 0  # ainda legível depois de soltar o Bloco


@pytest.mark.parametrize("v", OK, ids=lambda v: v["arquivo"])
def test_threads_nao_mudam_nada(v):
    um, quatro = (strata_ref.abre(VET / v["arquivo"], threads=t) for t in (1, 4))
    for d in (0, 1, None):
        a, b = um.todos(d), quatro.todos(d)
        assert all(a[k].tobytes() == b[k].tobytes() == quatro.tensor(k, d).tobytes() for k in um.nomes())


@pytest.mark.parametrize("v", OK, ids=lambda v: v["arquivo"])
@pytest.mark.parametrize("pedaco", [1, 7, 100, 1 << 20])
def test_fluxo_igual_a_ler_tudo(v, pedaco):
    dados = (VET / v["arquivo"]).read_bytes()
    pedacos = lambda: (dados[i : i + pedaco] for i in range(0, len(dados), pedaco))
    tudo = strata_ref.load(dados, framework="bytes")
    em_fluxo = strata_ref.load_stream(pedacos(), framework="bytes")
    assert {k: b.tobytes() for k, b in em_fluxo.items()} == {k: b.tobytes() for k, b in tudo.items()}
    for p in v.get("prefixos", []):
        d = p["profundidade"]
        ts = strata_ref.load_stream(pedacos(), framework="bytes", prof=d)
        assert sha(b"".join(ts[k].tobytes() for k in strata_ref.abre(dados).nomes())) == p["sha256_tensores"]


@pytest.mark.parametrize("v", RUIM, ids=lambda v: v["arquivo"])
def test_fluxo_recusa(v):
    dados = (VET / v["arquivo"]).read_bytes()
    try:
        strata_ref.load(dados, framework="bytes")
    except strata_ref.StrataError:
        with pytest.raises(strata_ref.StrataError):
            strata_ref.load_stream((dados[i : i + 13] for i in range(0, len(dados), 13)), framework="bytes")
    else:  # aceito pela leitura normal (ex.: só a volta ao .safetensors confere o SHA do original)
        strata_ref.load_stream([dados], framework="bytes")


def test_fluxo_dados_depois_do_fim_e_erro_de_download():
    dados = (VET / "ok-exemplo.strata").read_bytes()
    with pytest.raises(strata_ref.StrataError):
        strata_ref.load_stream([dados, b"x"], framework="bytes")

    def quebra():
        yield dados[:100]
        raise ConnectionError("caiu")

    with pytest.raises(ConnectionError):
        strata_ref.load_stream(quebra(), framework="bytes")
    with pytest.raises(strata_ref.StrataError):  # download incompleto
        strata_ref.load_stream([dados[:-1]], framework="bytes")


def test_load_url(tmp_path):
    """Por HTTP de verdade, num servidor local; com prof, o download para no estrato pedido."""
    import functools, http.server, threading
    v = next(v for v in OK if v.get("prefixos"))
    (tmp_path / "m.strata").write_bytes((VET / v["arquivo"]).read_bytes())
    class Calado(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Calado, directory=tmp_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/m.strata"
        tudo = strata_ref.load_url(url, framework="bytes")
        assert {k: b.tobytes() for k, b in tudo.items()} == {k: b.tobytes() for k, b in strata_ref.load(VET / v["arquivo"], framework="bytes").items()}
        p = v["prefixos"][0]
        ts = strata_ref.load_url(url, framework="bytes", prof=p["profundidade"], pedaco=3)
        assert sha(b"".join(ts[k].tobytes() for k in strata_ref.abre(VET / v["arquivo"]).nomes())) == p["sha256_tensores"]
    finally:
        srv.shutdown()
        srv.server_close()
