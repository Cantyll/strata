import sys, glob, os
sys.path.insert(0, '/home/user/work/kt/ik/gguf-py')
from gguf import GGUFReader
import numpy as np
for f in sys.argv[1:]:
    r = GGUFReader(f)
    db=dn=tb=tn=0; types={}
    for t in r.tensors:
        n=int(np.prod(t.shape)); b=int(t.n_bytes)
        tb+=b; tn+=n
        if t.name.startswith('blk.') and len(t.shape)>1 and n>=1<<16:
            db+=b; dn+=n; types[t.tensor_type.name]=types.get(t.tensor_type.name,0)+n
    print(f"{os.path.basename(f)}\tdecoder_bpw={8*db/dn:.4f}\ttotal_bpw={8*tb/tn:.4f}\tsize={os.path.getsize(f)}\t"+",".join(f"{k}:{v/dn:.2f}" for k,v in sorted(types.items())))
