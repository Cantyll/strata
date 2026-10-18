"""Modo residual (strata/0.3, A.14) pelo binding, contra tests/vetores/residual.

    cd leitor-py && maturin develop --release && python -m pytest tests -q
"""
import hashlib
import json
import pathlib

import pytest

import strata_ref

VET = pathlib.Path(__file__).resolve().parents[2] / "tests" / "vetores" / "residual"
MAN = json.loads((VET / "vetores.json").read_text())["vetores"]
OK = [v for v in MAN if v["espera"] == "ok"]
RUIM = [v for v in MAN if v["espera"] == "recusa"]
sha = lambda b: hashlib.sha256(b).hexdigest()
Q_PROJ = "model.layers.0.self_attn.q_proj.weight"


@pytest.mark.parametrize("v", OK, ids=lambda v: v["arquivo"])
def test_volta_gguf_e_prefixos(v):
    dados = (VET / v["arquivo"]).read_bytes()
    assert sha(strata_ref.para_safetensors(dados)) == v["sha256_safetensors"]
    f = strata_ref.abre(dados, exigir_integridade=True, id_esperado=v["id"])
    assert f.niveis == v["niveis"]
    assert f.extras == ["gguf:cabecalho", "gguf:rope_freqs.weight"]
    p0 = f.bytes_ate(0)
    assert sha(strata_ref.para_gguf(dados[:p0])) == v["sha256_gguf"] == sha((VET / "modelo.gguf").read_bytes())
    with pytest.raises(strata_ref.StrataError):
        strata_ref.para_gguf(dados[: p0 - 1])
    for p in v["prefixos"]:
        d, n = p["profundidade"], p["bytes"]
        g = strata_ref.abre(dados[:n])
        ts = g.todos(d)
        assert sha(b"".join(ts[k].tobytes() for k in g.nomes())) == p["sha256_tensores"]
        # em fluxo, nos mesmos bytes e sem os extras
        pedacos = (dados[i : i + 7] for i in range(0, n, 7))
        fl = strata_ref.load_stream(pedacos, framework="bytes", prof=d)
        assert set(fl) == set(g.nomes()) - set(g.extras)
        assert all(fl[k].tobytes() == ts[k].tobytes() for k in fl)


@pytest.mark.parametrize("framework", ["numpy", "pt"])
def test_igual_ao_safetensors(framework):
    """load_file e safe_open devolvem só os tensores do safetensors original, iguais a ele."""
    pytest.importorskip("safetensors")
    if framework == "pt":
        torch = pytest.importorskip("torch")
        from safetensors.torch import load_file as st_load
        igual = lambda a, b: a.dtype == b.dtype and a.shape == b.shape and a.view(-1).view(torch.uint8).equal(b.view(-1).view(torch.uint8))
    else:
        pytest.importorskip("ml_dtypes")
        from safetensors.numpy import load_file as st_load
        igual = lambda a, b: a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    try:
        ref = st_load(str(VET / "modelo.safetensors"))
    except Exception as e:  # o safetensors.numpy não lê BF16
        pytest.skip(f"safetensors não lê o modelo em {framework}: {e}")
    nosso = strata_ref.load_file(VET / "ok-residual.strata", framework=framework)
    assert nosso.keys() == ref.keys()
    assert all(igual(nosso[k], ref[k]) for k in ref)
    with strata_ref.safe_open(VET / "ok-residual.strata", framework=framework) as f:
        assert set(f.keys()) == set(ref)
        assert all(igual(f.get_tensor(k), ref[k]) for k in ref)


def test_blocos_q4kr():
    dados = (VET / "ok-residual.strata").read_bytes()
    f = strata_ref.abre(dados)
    q = f.blocos_q4kr(Q_PROJ, 0).tobytes()
    assert len(q) % 144 == 0
    b2 = strata_ref.abre(dados[: f.bytes_ate(2)]).blocos_q4kr(Q_PROJ, 2).tobytes()
    assert len(b2) == len(q) // 144 * 208
    assert all(b2[s * 208 : s * 208 + 144] == q[s * 144 : (s + 1) * 144] for s in range(len(q) // 144))
    for nome, k in [(Q_PROJ, 3), ("model.embed_tokens.weight", 1), ("model.norm.weight", 0)]:
        with pytest.raises(strata_ref.StrataError):
            f.blocos_q4kr(nome, k)
    with pytest.raises(KeyError):
        f.blocos_q4kr("nao.existe", 0)


@pytest.mark.parametrize("v", RUIM, ids=lambda v: v["arquivo"])
def test_recusa(v):
    dados = (VET / v["arquivo"]).read_bytes()
    with pytest.raises(strata_ref.StrataError):
        strata_ref.load(dados, framework="bytes")
    with pytest.raises(strata_ref.StrataError):
        strata_ref.load_stream((dados[i : i + 13] for i in range(0, len(dados), 13)), framework="bytes")


def test_gguf_com_sha_errado():
    dados = (VET / "ruim-gguf-sha.strata").read_bytes()
    strata_ref.para_safetensors(dados)
    with pytest.raises(strata_ref.StrataError):
        strata_ref.para_gguf(dados)
