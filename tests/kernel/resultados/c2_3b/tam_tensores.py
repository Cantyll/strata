"""Bytes de todos os tensores de um GGUF (sem o cabeçalho e o vocabulário): a "RAM" que a régua compara, como no 1B."""
import sys

import gguf

r = gguf.GGUFReader(sys.argv[1])
print(sum(int(t.n_bytes) for t in r.tensors))
