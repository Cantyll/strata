//! Binding Python do leitor de referência (strata-ref). Toda a validação e a decodificação são as
//! do crate em Rust; aqui só se abre o arquivo, solta o GIL enquanto decodifica e entrega cada
//! tensor como um `Bloco`: um buffer que o numpy e o torch leem sem copiar.
use pyo3::create_exception;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::os::raw::{c_int, c_void};
use std::path::PathBuf;
use strata_ref as sr;

create_exception!(_nativo, StrataError, PyValueError, "Arquivo .strata recusado pelo leitor (regras A.12 e A.13).");

fn erro(e: sr::Erro) -> PyErr {
    StrataError::new_err(e.0)
}

/// Bytes de um tensor, com dono em Rust. Expõe o protocolo de buffer, então
/// `numpy.frombuffer(b)` e `torch.frombuffer(b)` usam a mesma memória, sem cópia.
#[pyclass(module = "strata_ref._nativo")]
struct Bloco {
    v: Vec<u8>,
}

#[pymethods]
impl Bloco {
    fn __len__(&self) -> usize {
        self.v.len()
    }
    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.v)
    }
    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view nulo"));
        }
        // gravável, como um bytearray: o Bloco é o único dono do Vec e não tem nenhum método
        // que o realoque, então o ponteiro vale enquanto o objeto viver (o view guarda uma
        // referência a ele)
        let (ptr, len) = {
            let b = slf.borrow();
            (b.v.as_ptr() as *mut c_void, b.v.len() as ffi::Py_ssize_t)
        };
        if ffi::PyBuffer_FillInfo(view, slf.as_ptr(), ptr, len, 0, flags) != 0 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }
    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

/// Um .strata aberto e com o cabeçalho validado. O corpo só é lido quando se pede um tensor.
#[pyclass(module = "strata_ref._nativo")]
struct Arquivo {
    dados: Vec<u8>,
    a: sr::Arquivo,
    op: sr::Opcoes,
}

fn opcoes(exigir_integridade: bool, id_esperado: Option<String>, limite_saida: Option<u64>, threads: usize) -> sr::Opcoes {
    sr::Opcoes { exigir_integridade, id_esperado, limite_saida, threads }
}
fn prof(p: Option<u32>) -> u32 {
    p.unwrap_or(u32::MAX)
}

#[pymethods]
impl Arquivo {
    /// Nomes dos tensores, na ordem do cabeçalho.
    fn nomes(&self) -> Vec<String> {
        self.a.tensores.iter().map(|t| t.name.clone()).collect()
    }
    /// (dtype, shape) de um tensor.
    fn info(&self, nome: &str) -> PyResult<(String, Vec<u64>)> {
        let t = self.indice(nome).map(|i| &self.a.tensores[i])?;
        Ok((t.dtype.clone(), t.shape.clone()))
    }
    /// Um tensor com `prof` planos de mantissa (None = todos, que devolve os bytes originais).
    #[pyo3(signature = (nome, prof=None))]
    fn tensor(&self, py: Python<'_>, nome: &str, prof: Option<u32>) -> PyResult<Bloco> {
        let i = self.indice(nome)?;
        let p = self::prof(prof);
        let v = py.detach(|| sr::decode_tensor(&self.dados, &self.a, i, p, self.op.threads)).map_err(erro)?;
        Ok(Bloco { v })
    }
    /// Todos os tensores, decodificados em paralelo: dict nome -> Bloco.
    #[pyo3(signature = (prof=None))]
    fn todos<'py>(&self, py: Python<'py>, prof: Option<u32>) -> PyResult<Bound<'py, PyDict>> {
        let p = self::prof(prof);
        let (_, ts) = py.detach(|| sr::decode(&self.dados, p, &self.op)).map_err(erro)?;
        let d = PyDict::new(py);
        for (t, v) in self.a.tensores.iter().zip(ts) {
            d.set_item(&t.name, Bloco { v })?;
        }
        Ok(d)
    }
    /// O .safetensors original, byte a byte (confere o SHA-256 dele quando o arquivo o guarda).
    fn para_safetensors<'py>(&self, py: Python<'py>) -> PyResult<Bloco> {
        let v = py.detach(|| sr::to_safetensors(&self.dados, &self.op)).map_err(erro)?;
        Ok(Bloco { v })
    }
    /// Bytes do início do arquivo que bastam para ler tudo com `prof` planos de mantissa.
    #[pyo3(signature = (prof=None))]
    fn bytes_ate(&self, prof: Option<u32>) -> u64 {
        self.a.bytes_ate(self::prof(prof))
    }
    /// 0.3 (modo residual): o GGUF de origem byte a byte, remontado do estrato 0 (basta abrir o
    /// prefixo `bytes_ate(0)`); confere o SHA-256 dele quando o arquivo o guarda.
    fn para_gguf<'py>(&self, py: Python<'py>) -> PyResult<Bloco> {
        let v = py.detach(|| sr::para_gguf(&self.dados, &self.op)).map_err(erro)?;
        Ok(Bloco { v })
    }
    /// 0.3: blocos Q4_K_R<k> de um tensor residual Q4_K (B.8): por super-bloco de 256 pesos, os
    /// 144 bytes do Q4_K e os k planos de 32 bytes, na ordem do GGUF. Basta o prefixo `bytes_ate(k)`.
    fn blocos_q4kr<'py>(&self, py: Python<'py>, nome: &str, k: u32) -> PyResult<Bloco> {
        let i = self.indice(nome)?;
        let v = py.detach(|| sr::blocos_q4kr(&self.dados, &self.a, i, k)).map_err(erro)?;
        Ok(Bloco { v })
    }
    /// 0.3: níveis de refinamento (estratos 1..K) entre o GGUF do estrato 0 e o fecho exato.
    #[getter]
    fn niveis(&self) -> u32 {
        self.a.niveis
    }
    /// 0.3: tensores extras (pedaços do GGUF que o safetensors não tem); `load` não os devolve.
    #[getter]
    fn extras(&self) -> Vec<String> {
        self.a.tensores.iter().filter(|t| t.extra).map(|t| t.name.clone()).collect()
    }
    /// Id do arquivo: SHA-256 de MAGIC + HL + cabeçalho (A.13).
    #[getter]
    fn id(&self) -> PyResult<String> {
        sr::cabecalho_id(&self.dados).map_err(erro)
    }
    /// True quando o arquivo traz os hashes de A.13 (e eles são conferidos na leitura).
    #[getter]
    fn integridade(&self) -> bool {
        self.a.integridade
    }
    /// Maior número de planos de mantissa entre os tensores.
    #[getter]
    fn planos(&self) -> u32 {
        self.a.m
    }
    /// SHA-256 do .safetensors original, quando o arquivo o guarda (= oid LFS no Hugging Face).
    #[getter]
    fn fonte_sha256(&self) -> Option<String> {
        self.a.fonte_sha256.clone()
    }
    /// Tamanho, em bytes, do arquivo inteiro segundo o cabeçalho.
    #[getter]
    fn tamanho_total(&self) -> u64 {
        self.a.fim
    }
    /// Bytes que de fato foram abertos (menor que tamanho_total quando é um prefixo).
    #[getter]
    fn tamanho_lido(&self) -> usize {
        self.dados.len()
    }
    fn __len__(&self) -> usize {
        self.a.tensores.len()
    }
}

impl Arquivo {
    fn indice(&self, nome: &str) -> PyResult<usize> {
        self.a.tensores.iter().position(|t| t.name == nome).ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(nome.to_string()))
    }
}

/// Leitura em fluxo: cada stream é conferido e descomprimido assim que termina de chegar.
/// `Fluxo(primeiros_bytes, prof)` (pelo menos o cabeçalho), `alimenta(pedaço)` na ordem do
/// arquivo, `termina()` entrega os tensores. Depois de um erro o fluxo não aceita mais nada.
#[pyclass(module = "strata_ref._nativo")]
struct Fluxo {
    f: Option<sr::Fluxo>,
    total: u64,
}

impl Fluxo {
    fn vivo(&self) -> PyResult<&sr::Fluxo> {
        self.f.as_ref().ok_or_else(|| StrataError::new_err("fluxo encerrado (terminado ou recusado)"))
    }
}

#[pymethods]
impl Fluxo {
    #[new]
    #[pyo3(signature = (inicio, prof=None, exigir_integridade=false, id_esperado=None, limite_saida=None, threads=0))]
    fn novo(py: Python<'_>, inicio: &[u8], prof: Option<u32>, exigir_integridade: bool, id_esperado: Option<String>, limite_saida: Option<u64>, threads: usize) -> PyResult<Self> {
        let op = opcoes(exigir_integridade, id_esperado, limite_saida, threads);
        let p = self::prof(prof);
        let f = py.detach(|| sr::Fluxo::novo(inicio, p, &op)).map_err(erro)?;
        let total = f.arquivo().fim;
        Ok(Fluxo { f: Some(f), total })
    }
    /// Os próximos bytes do arquivo. Decodifica (sem o GIL) os streams que ficaram completos.
    fn alimenta(&mut self, py: Python<'_>, dados: &[u8]) -> PyResult<()> {
        let mut f = self.f.take().ok_or_else(|| StrataError::new_err("fluxo encerrado (terminado ou recusado)"))?;
        let r = py.detach(|| f.alimenta(dados).map(|_| f));
        self.f = Some(r.map_err(erro)?);
        Ok(())
    }
    /// Os tensores com os planos pedidos ao criar o fluxo: dict nome -> Bloco. Encerra o fluxo.
    fn termina<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let f = self.f.take().ok_or_else(|| StrataError::new_err("fluxo encerrado (terminado ou recusado)"))?;
        let (a, ts) = py.detach(|| f.termina()).map_err(erro)?;
        let d = PyDict::new(py);
        for (t, v) in a.tensores.iter().zip(ts) {
            d.set_item(&t.name, Bloco { v })?;
        }
        Ok(d)
    }
    /// Bytes recebidos até agora.
    #[getter]
    fn recebidos(&self) -> PyResult<u64> {
        Ok(self.vivo()?.recebidos())
    }
    /// Tamanho do arquivo inteiro segundo o cabeçalho.
    #[getter]
    fn tamanho_total(&self) -> u64 {
        self.total
    }
    /// Bytes do início do arquivo que bastam para `termina(prof)`.
    #[pyo3(signature = (prof=None))]
    fn bytes_ate(&self, prof: Option<u32>) -> PyResult<u64> {
        Ok(self.vivo()?.arquivo().bytes_ate(self::prof(prof)))
    }
    fn nomes(&self) -> PyResult<Vec<String>> {
        Ok(self.vivo()?.arquivo().tensores.iter().map(|t| t.name.clone()).collect())
    }
    /// 0.3: tensores extras (pedaços do GGUF que o safetensors não tem).
    fn extras(&self) -> PyResult<Vec<String>> {
        Ok(self.vivo()?.arquivo().tensores.iter().filter(|t| t.extra).map(|t| t.name.clone()).collect())
    }
    /// (dtype, shape) de cada tensor, na ordem do cabeçalho.
    fn infos(&self) -> PyResult<Vec<(String, String, Vec<u64>)>> {
        Ok(self.vivo()?.arquivo().tensores.iter().map(|t| (t.name.clone(), t.dtype.clone(), t.shape.clone())).collect())
    }
}

/// Origem do arquivo: um caminho (str ou PathLike) ou os próprios bytes.
#[derive(FromPyObject)]
enum Origem {
    Bytes(Vec<u8>),
    Caminho(PathBuf),
}

/// Abre e valida um .strata (só o cabeçalho; o corpo é conferido quando cada tensor é lido).
/// `fonte` pode ser só um prefixo do arquivo, com os estratos que se quer ler.
#[pyfunction]
#[pyo3(signature = (fonte, exigir_integridade=false, id_esperado=None, limite_saida=None, threads=0))]
fn abre(py: Python<'_>, fonte: Origem, exigir_integridade: bool, id_esperado: Option<String>, limite_saida: Option<u64>, threads: usize) -> PyResult<Arquivo> {
    let op = opcoes(exigir_integridade, id_esperado, limite_saida, threads);
    py.detach(|| {
        let dados = match fonte {
            Origem::Bytes(b) => b,
            Origem::Caminho(p) => std::fs::read(&p).map_err(|e| sr::Erro(format!("{}: {e}", p.display())))?,
        };
        let a = sr::abre(&dados, &op)?;
        Ok(Arquivo { dados, a, op })
    })
    .map_err(erro)
}

/// Só o cabeçalho (MAGIC + HL + JSON), para planejar um download parcial: `bytes_ate(prof)` diz
/// quantos bytes do início do arquivo baixar. De um caminho, lê só os 12 + HL primeiros bytes.
/// Não tem corpo, então pedir um tensor dele dá erro; por isso dispensa o limite de saída.
#[pyfunction]
#[pyo3(signature = (fonte, id_esperado=None))]
fn abre_cabecalho(py: Python<'_>, fonte: Origem, id_esperado: Option<String>) -> PyResult<Arquivo> {
    use std::io::Read;
    let op = opcoes(false, id_esperado, Some(u64::MAX), 1);
    py.detach(|| {
        let le = |p: &PathBuf| -> std::io::Result<Vec<u8>> {
            let mut f = std::fs::File::open(p)?;
            let mut d = vec![0u8; 12];
            f.read_exact(&mut d)?;
            let hl = u32::from_le_bytes(d[8..12].try_into().unwrap()) as u64;
            f.take(hl.min(sr::LIMITE_CABECALHO)).read_to_end(&mut d)?;
            Ok(d)
        };
        let mut dados = match fonte {
            Origem::Bytes(b) => b,
            Origem::Caminho(p) => le(&p).map_err(|e| sr::Erro(format!("{}: {e}", p.display())))?,
        };
        if dados.len() >= 12 {
            let hl = u32::from_le_bytes(dados[8..12].try_into().unwrap()) as usize;
            dados.truncate(12usize.saturating_add(hl));
        }
        let a = sr::abre(&dados, &op)?;
        Ok(Arquivo { dados, a, op })
    })
    .map_err(erro)
}

#[pymodule]
fn _nativo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(abre, m)?)?;
    m.add_function(wrap_pyfunction!(abre_cabecalho, m)?)?;
    m.add_class::<Arquivo>()?;
    m.add_class::<Bloco>()?;
    m.add_class::<Fluxo>()?;
    m.add("StrataError", m.py().get_type::<StrataError>())?;
    Ok(())
}
