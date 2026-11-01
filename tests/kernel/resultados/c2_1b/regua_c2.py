import re, json, math, os
G='/home/user/work/g2'; C='/home/user/work/c2l'
MiB = {'Q4_K_M': 762.81, 'Q5_K_S': 843.75, 'Q5_K_M': 861.81, 'Q6_K': 967.00, 'Q8_0': 1252.41}
NCAM = 973078528; Q4F = os.path.getsize(f'{G}/Q4_K_M.gguf')
def kl(f):
    t=open(f).read(); m=re.search(r'Mean +KLD: +([0-9.]+) +± +([0-9.]+)', t); return (float(m.group(1)), float(m.group(2)))
def frente(t):
    pts = sorted((MiB[q], kl(f'{G}/{t}_F16_{q}.txt')[0]) for q in MiB)
    return [p for p in pts if not any(o[0] <= p[0] and o[1] < p[1] for o in pts if o != p)]
def tam(k, t):
    pts = frente(t); lk = math.log(k)
    for (s1, k1), (s2, k2) in zip(pts, pts[1:]):
        if k >= k2 or (s2, k2) == pts[-1]: break
    a = (lk - math.log(k1)) / (math.log(k2) - math.log(k1)); return s1 + a * (s2 - s1)
bits = lambda mib: mib * 8 * 2**20 / NCAM
RAM = {0: MiB['Q4_K_M'], 1: MiB['Q4_K_M'] + NCAM/8/2**20, 2: MiB['Q4_K_M'] + 2*NCAM/8/2**20}
niv = {}
for t in ('wiki','c4'):
    niv[t] = {0: kl(f'{G}/{t}_F16_Q4_K_M.txt')[0], 1: kl(f'{G}/{t}_P,DL_gptq_a0.1_h500_N1.txt')[0], 2: kl(f'{G}/{t}_P,DL_gptq_a0.1_h500_N2.txt')[0]}
def interp(r, t):  # níveis inteiros, interpolação do log da KL em RAM
    k = 0 if r < RAM[1] else 1
    a = (r - RAM[k]) / (RAM[k+1] - RAM[k]); return math.exp(math.log(niv[t][k]) + a*(math.log(niv[t][k+1]) - math.log(niv[t][k])))
for t in ('wiki','c4'):
    print(t, 'níveis', {k: round(1000*v,2) for k,v in niv[t].items()}, 'bits a mais', {k: round(bits(RAM[k]-tam(v,t)),3) for k,v in niv[t].items()})
out = {}
for m in (850, 900, 950):
    for a in ('dkl','mse'):
        p = json.load(open(f'{C}/p_{a}_{m}.json')); r = MiB['Q4_K_M'] + (p['gib']*2**30 - Q4F)/2**20
        row = [f'{m} {a}: {r:6.1f} MiB']
        for t in ('wiki','c4'):
            v, e = kl(f'{C}/{t}_{a}_{m}.txt'); li = interp(r, t)
            row.append(f'{t} {1000*v:6.2f}±{1000*e:.2f} ({bits(r-tam(v,t)):+.3f} b; níveis inteiros {1000*li:.2f}, {bits(r-tam(li,t)):+.3f} b)')
            out[f'{t}_{a}_{m}'] = dict(mib=r, kl=v, err=e, bits=bits(r-tam(v,t)), kl_inteiros=li, bits_inteiros=bits(r-tam(li,t)))
        print('  '.join(row))
json.dump(out, open(f'{C}/regua.json','w'), indent=1)
