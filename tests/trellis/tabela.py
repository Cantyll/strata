import re, subprocess, sys, math, os
W='/home/user/work/kt'
types=sys.argv[1:]
rows=[]
for t in types:
    kl=open(f'{W}/logs/kl_{t}.log').read()
    m=re.search(r'Mean +KLD:\s+([\d.]+)\s+±\s+([\d.]+)',kl); k,ke=float(m[1]),float(m[2])
    p=re.search(r'Mean PPL\(Q\)\s+:\s+([\d.]+)',kl); ppl=float(p[1])
    s=re.search(r'Same top p:\s+([\d.]+)',kl); top=float(s[1]) if s else float('nan')
    b=open(f'{W}/logs/bench_{t}.log').read()
    pp=re.search(r'pp512 \|\s+([\d.]+) ± ([\d.]+)',b); tg=re.search(r'tg128 \|\s+([\d.]+) ± ([\d.]+)',b)
    out=subprocess.run([sys.executable,'/home/user/work/ktscripts/bpw.py',f'{W}/q/{t}.gguf'],capture_output=True,text=True).stdout
    d=float(re.search(r'decoder_bpw=([\d.]+)',out)[1]); tot=float(re.search(r'total_bpw=([\d.]+)',out)[1])
    rows.append((t,d,tot,k,ke,ppl,top,float(pp[1]),float(pp[2]),float(tg[1]),float(tg[2])))
rows.sort(key=lambda r:r[1])
print('| tipo | b/peso decoder | b/peso arquivo | KL ×1000 | PPL | mesmo top-1 % | pp512 tok/s | tg128 tok/s |')
print('|---|---:|---:|---:|---:|---:|---:|---:|')
for r in rows:
    print(f'| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {1000*r[3]:.2f} ± {1000*r[4]:.2f} | {r[5]:.3f} | {r[6]:.2f} | {r[7]:.0f} ± {r[8]:.0f} | {r[9]:.2f} ± {r[10]:.2f} |')
import json; json.dump(rows,open(f'{W}/tabela.json','w'))
