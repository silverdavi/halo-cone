"""Independent standard-library implementation of the five-term obstruction.

Imports neither the original calculation nor NumPy/SciPy. Inputs are decimal
strings; integration is analytic on each linear flux-table segment. Helm
factors use power-series trigonometry. High precision is NOT interval arithmetic.
"""
from decimal import Decimal as D, localcontext
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ATOMIC = ('123.9058920','125.9042983','127.9035310','128.9047808611',
          '129.903509349','130.90508406','131.9041550856','133.90539466','135.907214484')
ABUNDANCE = ('.00095','.00089','.0191','.264','.0407','.212','.269','.104','.0886')
ISOTOPES = (124,126,128,129,130,131,132,134,136)

def atan(x):
    total = term = x
    n = 0
    while True:
        n += 1
        term *= -x*x
        new = total + term/D(2*n+1)
        if new == total: return total
        total = new

def sincos(x):
    sine = st = x
    cosine = ct = D(1)
    n = 0
    while True:
        n += 1
        st *= -x*x/D((2*n)*(2*n+1))
        ct *= -x*x/D((2*n-1)*(2*n))
        sn, cn = sine+st, cosine+ct
        if sn == sine and cn == cosine: return sine, cosine
        sine, cosine = sn, cn

def cubert(x):
    value = D(5)
    # Fixed iteration avoids a one-ulp cycle in rounded Newton iteration.
    for _ in range(30): value = (2*value+x/(value*value))/3
    return value

def evaluate(precision):
    with localcontext() as ctx:
        ctx.prec = precision
        pi = 16*atan(D(1)/5)-4*atan(D(1)/239)
        atomic = list(map(D, ATOMIC))
        mass = [x*D('.93149410242')-54*D('.00051099895') for x in atomic]
        raw_f = list(map(D, ABUNDANCE))
        f = [x/sum(raw_f) for x in raw_f]
        mean_mass = sum(x*y for x,y in zip(f,mass))
        xi = [x*y/mean_mass for x,y in zip(f,mass)]
        nt = D('1e6')*D('6.02214076e23')/sum(x*y for x,y in zip(f,atomic))
        constant = nt*D('5.25e6')*D('31557600')*D('1.1663787e-5')**2/(4*pi)*D('.3893793721e-27')*D('1e-6')
        qw = [D(a-54)-(1-4*D('.23857'))*54 for a in ISOTOPES]
        radius = [((D('1.23')*cubert(D(a))-D('.60'))**2+7*pi*pi*D('.52')**2/3-5*D('.9')**2).sqrt() for a in ISOTOPES]
        rows = [(D(r[0]),D(r[1])) for r in csv.reader((HERE/'b8_spectrum.csv').open())]
        norm = sum((u1-u0)*(p0+p1)/2 for (u0,p0),(u1,p1) in zip(rows,rows[1:]))
        segments = [(u0,u1,p0/norm,p1/norm) for (u0,p0),(u1,p1) in zip(rows,rows[1:]) if p0 or p1]

        def psi(q):
            total = D(0)
            for u0,u1,p0,p1 in segments:
                if u1 <= q: continue
                left = max(q,u0)
                slope = (p1-p0)/(u1-u0)
                intercept = p0-slope*u0
                # Direct definite integral of (slope*u+intercept)*(1-q^2/u^2).
                total += slope*((u1*u1-left*left)/2-q*q*(u1/left).ln())
                total += intercept*((u1-left)+q*q*(1/u1-1/left))
            return total

        def helm(E,i):
            momentum = (2*mass[i]*E*D('1e-6')).sqrt()/D('.1973269804')
            z = momentum*radius[i]
            sine,cosine = sincos(z)
            sphere = 3*(sine-z*cosine)/(z*z*z)
            return sphere*sphere*(-(momentum*D('.9'))**2).exp()

        def coeff(E,i): return xi[i]*ISOTOPES[i]**2*helm(E,i)

        def spectrum(E):
            return constant*sum(f[i]*mass[i]*qw[i]**2*helm(E,i)*psi((mass[i]*E/2).sqrt()) for i in range(9))

        def T(E): return spectrum(E)/coeff(E,0)
        def ratio(E,i): return coeff(E,i)/coeff(E,0)

        endpoint = 2*D('.01636')**2/mass[0]*D('1e6')
        E = D('.98')**2*endpoint
        results = {}
        for mstring in ('6.65','7','7.80'):
            m = D(mstring)
            mu = [m*M/(m+M) for M in mass]
            alpha = [(M/mass[0]).sqrt()*mu[0]/mi for M,mi in zip(mass,mu)]
            L = -D('.98').ln()
            margins = [L-alpha[3].ln(),3*alpha[1].ln()-L,
                       alpha[4].ln()-L,alpha[1].ln()+alpha[2].ln()-L]
            assert min(margins)>0
            terms = [T(E),-ratio(E,1)*T(alpha[1]**2*E),
                     ratio(E,1)*ratio(alpha[1]**2*E,1)*T(alpha[1]**4*E),
                     -ratio(E,2)*T(alpha[2]**2*E),-ratio(E,3)*T(alpha[3]**2*E)]
            results[mstring] = {'terms':terms,'sum':sum(terms),'ordering_margins':margins}
        uniform = results['6.65']['sum']-results['6.65']['terms'][2]+results['7.80']['terms'][2]
        return {'precision':precision,'flux_sha256':hashlib.sha256((HERE/'b8_spectrum.csv').read_bytes()).hexdigest(),
                'masses_GeV':mass,'flux_integral':norm,'E0_keV':E,'constant':constant,
                'evaluations':results,'uniform_upper_bound':uniform}

if __name__ == '__main__':
    outputs = [evaluate(p) for p in (60,100)]
    for result in outputs:
        print('DECIMAL PRECISION',result['precision'])
        print('E0_keV',result['E0_keV'])
        for mass,values in result['evaluations'].items():
            print('m =',mass)
            for i,t in enumerate(values['terms'],1): print(f'  term {i}: {t:.24E}')
            print(f"  SUM: {values['sum']:.24E}")
        print(f"UNIFORM UPPER BOUND: {result['uniform_upper_bound']:.24E}")
    with localcontext() as ctx:
        ctx.prec=110
        for m in ('6.65','7','7.80'):
            for key in ('sum',):
                error=abs(outputs[0]['evaluations'][m][key]-outputs[1]['evaluations'][m][key])
                assert error < D('1e-60'),error
    (HERE/'results.json').write_text(json.dumps(outputs,default=str,indent=2)+'\n')
