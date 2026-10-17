"""Leitor de referência do formato .strata, em Rust, para Python.

Toda a validação (docs/02, A.12) e a conferência de hashes (A.13) acontecem no Rust; este módulo
só converte os bytes de cada tensor para numpy ou torch, sem copiar.

    import strata_ref
    pesos = strata_ref.load_file("modelo.strata", framework="torch")       # como safetensors.torch.load_file
    with strata_ref.safe_open("modelo.strata", framework="numpy") as f:     # como safetensors.safe_open
        w = f.get_tensor("model.norm.weight")

`prof` escolhe quantos planos de mantissa ler (None = todos, que dá os pesos originais bit a bit;
0 = só sinal e expoente). Para ler numa profundidade sem baixar o resto:

    cab = strata_ref.abre_cabecalho(primeiros_bytes)   # 12 + HL bytes; HL = uint32 nos bytes 8..12
    n = cab.bytes_ate(3)                                # quanto baixar (ex.: HTTP Range: bytes=0-{n-1})
    pesos = strata_ref.load(prefixo_de_n_bytes, framework="pt", prof=3)
"""
from ._nativo import Arquivo, Bloco, StrataError, abre, abre_cabecalho

__all__ = ["abre", "abre_cabecalho", "load_file", "load", "safe_open", "para_safetensors", "Arquivo", "Bloco", "StrataError"]

_NUMPY = {
    "F64": "float64", "F32": "float32", "F16": "float16",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U64": "uint64", "U32": "uint32", "U16": "uint16", "U8": "uint8", "BOOL": "bool",
}
# o numpy não tem esses tipos; o pacote ml_dtypes (do JAX) tem
_ML_DTYPES = {"BF16": "bfloat16", "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2"}
_TORCH = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
    "I64": "int64", "I32": "int32", "I16": "int16", "I8": "int8",
    "U64": "uint64", "U32": "uint32", "U16": "uint16", "U8": "uint8", "BOOL": "bool",
}


def _converte(bloco, dtype, shape, framework):
    if framework in (None, "bytes"):
        return bloco
    if framework in ("np", "numpy"):
        import numpy as np
        if dtype in _NUMPY:
            dt = np.dtype(_NUMPY[dtype])
        elif dtype in _ML_DTYPES:
            try:
                import ml_dtypes
            except ImportError:
                raise TypeError(f"o numpy não tem {dtype}; instale ml_dtypes ou use framework='pt'") from None
            dt = np.dtype(getattr(ml_dtypes, _ML_DTYPES[dtype]))
        else:
            raise TypeError(f"dtype sem equivalente no numpy: {dtype}")
        return np.frombuffer(bloco, dtype=dt).reshape(shape)
    if framework in ("pt", "torch"):
        import torch
        if dtype not in _TORCH or not hasattr(torch, _TORCH[dtype]):
            raise TypeError(f"dtype sem equivalente nesta versão do torch: {dtype}")
        dt = getattr(torch, _TORCH[dtype])
        if len(bloco) == 0:
            return torch.empty(shape, dtype=dt)
        return torch.frombuffer(bloco, dtype=dt).reshape(shape)
    raise ValueError(f"framework desconhecido: {framework!r} (use 'numpy', 'pt' ou 'bytes')")


def load(fonte, framework="numpy", prof=None, **opcoes):
    """Todos os tensores (decodificados em paralelo) como dict nome -> array. `fonte`: caminho ou bytes."""
    f = abre(fonte, **opcoes)
    blocos = f.todos(prof)
    return {n: _converte(b, *f.info(n), framework) for n, b in blocos.items()}


def load_file(caminho, framework="numpy", prof=None, **opcoes):
    """Como safetensors.<framework>.load_file, para um .strata."""
    return load(caminho, framework, prof, **opcoes)


def para_safetensors(fonte, **opcoes):
    """Os bytes do .safetensors original, conferidos contra o SHA-256 guardado no arquivo."""
    return abre(fonte, **opcoes).para_safetensors().tobytes()


class safe_open:
    """Como safetensors.safe_open: valida o cabeçalho ao abrir e decodifica cada tensor quando pedido."""

    def __init__(self, caminho, framework="numpy", device="cpu", prof=None, **opcoes):
        if device != "cpu":
            raise ValueError("por enquanto só device='cpu'; mova depois com .to(device)")
        self._f = abre(caminho, **opcoes)
        self._framework, self._prof = framework, prof

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def keys(self):
        return self._f.nomes()

    def get_tensor(self, nome, prof=None):
        p = self._prof if prof is None else prof
        return _converte(self._f.tensor(nome, p), *self._f.info(nome), self._framework)

    @property
    def arquivo(self):
        """O Arquivo nativo (id, integridade, bytes_ate, fonte_sha256...)."""
        return self._f
