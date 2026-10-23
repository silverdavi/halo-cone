"""Rigorous certificate for the supplied piecewise-linear 8B table.

Python standard library only. All input and arithmetic are exact rationals.
Logarithms are enclosed by a convergent rational series with a proved remainder.
Interval endpoints are rounded OUTWARD to a fixed rational lattice after every
operation. No floating-point or Decimal transcendental evaluation is used.

Run: python certify.py [b8_spectrum.csv]
"""
from fractions import Fraction as F
from functools import lru_cache
from bisect import bisect_right
from pathlib import Path
import csv
import hashlib
import json
import sys

SCALE = 10**45
LOG_ERROR = F(1,10**55)
MASS_MEV = F(122300)
DELTA = F('0.1')

def floor(x): return x.numerator//x.denominator
def ceil(x): return -floor(-x)

class I:
    """Closed interval with exact, outward-rounded rational endpoints."""
    def __init__(self,lo=0,hi=None):
        lo=F(lo);hi=lo if hi is None else F(hi)
        assert lo<=hi
        self.lo=F(floor(lo*SCALE),SCALE)
        self.hi=F(ceil(hi*SCALE),SCALE)
    @staticmethod
    def cast(x): return x if isinstance(x,I) else I(x)
    def __add__(self,x):
        x=I.cast(x);return I(self.lo+x.lo,self.hi+x.hi)
    __radd__=__add__
    def __neg__(self):return I(-self.hi,-self.lo)
    def __sub__(self,x):return self+-I.cast(x)
    def __rsub__(self,x):return I.cast(x)+-self
    def __mul__(self,x):
        x=I.cast(x);v=[self.lo*x.lo,self.lo*x.hi,self.hi*x.lo,self.hi*x.hi]
        return I(min(v),max(v))
    __rmul__=__mul__
    def __truediv__(self,x):
        x=I.cast(x);assert x.lo*x.hi>0
        return self*I(1/x.hi,1/x.lo)

@lru_cache(None)
def log_i(r):
    r=F(r);assert r>0
    if r<1:return -log_i(1/r)
    if r==1:return I(0)
    # Range reduction is exact. The direct series then has 0 <= z <= 1/3.
    if r>2:
        k=0
        while r>2:r/=2;k+=1
        return k*log_i(F(2))+log_i(r)
    z=(r-1)/(r+1);z2=z*z;power=z;total=F(0);n=0
    while True:
        total+=2*power/(2*n+1)
        n+=1;power*=z2
        remainder=2*power/((2*n+1)*(1-z2))
        if remainder<=LOG_ERROR:return I(total,total+remainder)

def decimals(x,places=12,upper=False):
    scale=10**places
    n=ceil(x*scale) if upper else floor(x*scale)
    sign='-' if n<0 else '';n=abs(n)
    return f'{sign}{n//scale}.{n%scale:0{places}d}'

def enclosure(x,places=16):
    return [decimals(x.lo,places),decimals(x.hi,places,True)]

def root_bracket(fun,left,right,increasing=False):
    assert (fun(left).hi<0 and fun(right).lo>0) if increasing else (fun(left).lo>0 and fun(right).hi<0)
    while right-left>F(1,10**14):
        middle=(left+right)/2;v=fun(middle)
        assert v.lo>0 or v.hi<0, 'Increase SCALE/LOG precision to decide this sign'
        if (v.hi<0) == increasing:left=middle
        else:right=middle
    # Return a visibly decimal, outward enclosure; recheck its endpoint signs.
    left=F(floor(left*10**12),10**12)
    right=F(ceil(right*10**12),10**12)
    assert (fun(left).hi<0 and fun(right).lo>0) if increasing else (fun(left).lo>0 and fun(right).hi<0)
    return left,right

class Spectrum:
    def __init__(self,rows,column,literal_cutoff=False):
        x=[r[0] for r in rows];y=[r[column] for r in rows]
        assert all(x[i]<x[i+1] for i in range(len(x)-1))
        assert x[0]==0 and y[0]==0 and all(v>=0 for v in y)
        last=max(i for i,v in enumerate(y) if v>0)
        assert last+1<len(x) and y[last+1]==0
        self.last_positive=x[last]
        end=last if literal_cutoff else last+1
        self.x=x[:end+1];raw=y[:end+1];self.U=self.x[-1]
        self.norm=sum((b-a)*(u+v)/2 for a,b,u,v in zip(self.x,self.x[1:],raw,raw[1:]))
        self.y=[v/self.norm for v in raw]
        self.s=[(v-u)/(b-a) for a,b,u,v in zip(self.x,self.x[1:],self.y,self.y[1:])]
        self.c=[v-s*a for v,s,a in zip(self.y,self.s,self.x)]
        self.mode=self.y.index(max(self.y))
        assert all(s>0 for s in self.s[:self.mode])
        assert all(s<=0 for s in self.s[self.mode:])
        assert all(v>0 for v in self.y[1:-1])
        self.tail2=[I(0) for _ in self.x];self.tail1=[I(0) for _ in self.x]
        for i in range(len(self.s)-1,0,-1):
            a,b=self.x[i:i+2];s,c=self.s[i],self.c[i];L=log_i(b/a)
            self.tail2[i]=self.tail2[i+1]+s*L+c*(1/a-1/b)
            self.tail1[i]=self.tail1[i+1]+s*(b-a)+c*L
    def hp(self,q,full=False,segment=None):
        q=F(q);assert 0<q<=self.U
        i=min(bisect_right(self.x,q)-1,len(self.s)-1) if segment is None else segment
        a,b=self.x[i:i+2];assert a<=q<=b
        s,c=self.s[i],self.c[i];phi=s*q+c;L=log_i(b/q)
        J2=self.tail2[i+1]+s*L+c*(1/q-1/b)
        hp=J2-phi/q
        if full:
            J1=self.tail1[i+1]+s*(b-q)+c*L
            hp+=(2*J1-3*phi-q*s)/MASS_MEV
        return hp
    def full_derivative_numerator(self,q,i):
        # h_full'' = -[s*(M+6q)+2c]/(M*q).
        return self.s[i]*(MASS_MEV+6*q)+2*self.c[i]
    def extrema_points(self,a,b,i,full):
        points=[a,b]
        if full and self.s[i]:
            critical=-(MASS_MEV*self.s[i]+2*self.c[i])/(6*self.s[i])
            if a<critical<b:points.append(critical)
        return points
    def minus_hp_bound(self,left,right,full=False):
        bounds=[]
        for i in range(len(self.s)):
            a=max(left,self.x[i]);b=min(right,self.x[i+1])
            if a>=b:continue
            for q in self.extrema_points(a,b,i,full):
                bounds.append(-self.hp(q,full,i).hi)
        assert bounds and min(bounds)>0
        return min(bounds)

def certify(rows,column,literal=False):
    sp=Spectrum(rows,column,literal)
    lo,hi=root_bracket(sp.hp,F(3),F('3.4'))
    assert hi<sp.x[sp.mode]
    # Strict increasing phi before its mode + nonincreasing after proves uniqueness.
    leading_lower=sp.minus_hp_bound(lo+DELTA,sp.U-DELTA)
    assert leading_lower>F('6e-9')
    energy=I(2*lo*lo*1000/MASS_MEV,2*hi*hi*1000/MASS_MEV)
    out={'column':column+1,'endpoint_convention':'last-positive cutoff' if literal else 'continuous interpolant',
         'raw_normalization':str(sp.norm),'last_positive_node_MeV':str(sp.last_positive),
         'U_MeV':str(sp.U),'mode_MeV':str(sp.x[sp.mode]),
         'q_star_MeV':[decimals(lo),decimals(hi,upper=True)],
         'hprime_at_bracket_endpoints':[enclosure(sp.hp(lo),30),enclosure(sp.hp(hi),30)],
         'E_star_keV':enclosure(energy,12),
         'delta_MeV':'0.1','minus_hprime_uniform_certified_lower':decimals(leading_lower,18),
         'hprime_3_18':enclosure(sp.hp(F('3.18')),20)}
    fun=lambda q:sp.hp(q,True)
    fl,fr=root_bracket(fun,F(3),F('3.4'))
    root_segment=bisect_right(sp.x,fl)-1
    assert root_segment==bisect_right(sp.x,fr)-1
    assert sp.s[root_segment]>0
    # Exactly one primary zero. With continuous vanishing endpoint there is
    # exactly one further zero on the final segment; with the literal cutoff,
    # certify every remaining segment instead.
    for i in range(len(sp.s)-(0 if literal else 1)):
        a,b=sp.x[i:i+2]
        if i==0:
            assert sp.s[i]>0 and sp.c[i]==0 and sp.hp(b,True,i).lo>0
        elif i<root_segment:
            assert min(sp.hp(q,True,i).lo for q in sp.extrema_points(a,b,i,True))>0
        elif i==root_segment:
            assert sp.hp(a,True,i).lo>0 and sp.hp(b,True,i).hi<0
            assert min(sp.full_derivative_numerator(q,i) for q in [a,b])>0
        else:
            assert max(sp.hp(q,True,i).hi for q in sp.extrema_points(a,b,i,True))<0
    if not literal:
        last=len(sp.s)-1;a,b=sp.x[-2:]
        assert max(sp.full_derivative_numerator(q,last) for q in [a,b])<0
        el,er=root_bracket(fun,a,b,increasing=True)
    shift=I(fl-hi,fr-lo)
    assert shift.lo>F('-0.000046') and shift.hi<0
    full_lower=sp.minus_hp_bound(lo+DELTA,sp.U-DELTA,True)
    assert full_lower>F('5e-9')
    out.update({'q_full_primary_MeV':[decimals(fl),decimals(fr,upper=True)],
                'full_curvature_zero_count':1 if literal else 2,
                'full_primary_shift_MeV':enclosure(shift,12),
                'minus_h_full_prime_uniform_lower':decimals(full_lower,18),
                'psi_full_second_derivative_U_minus':enclosure(-2*sp.hp(sp.U,True),20)})
    if not literal:
        out.update({'q_full_endpoint_zero_MeV':[decimals(el),decimals(er,upper=True)],
                    'endpoint_phi_linear_coefficient':str(-sp.s[-1])})
    return out

def main():
    path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name('b8_spectrum.csv')
    raw=path.read_bytes()
    rows=[[F(x) for x in row] for row in csv.reader(raw.decode().splitlines()) if row]
    assert len(rows)==829 and all(len(r)==4 for r in rows)
    digest=hashlib.sha256(raw).hexdigest()
    assert digest=='b07b3ba052c08cf675c8fcc5c5f674e0946390b721a0ea4da719077bdf5bc892'
    results=[certify(rows,c,literal) for literal in (False,True) for c in (1,2,3)]
    report={'status':'ALL RATIONAL INTERVAL ASSERTIONS PASSED','input_sha256':digest,
            'arithmetic':'fractions.Fraction; outward rational intervals',
            'interval_lattice_denominator':str(SCALE),'per_log_series_remainder_max':str(LOG_ERROR),
            'mass_MeV':str(MASS_MEV),'results':results}
    text=json.dumps(report,indent=2)
    Path(__file__).with_name('certificate.json').write_text(text+'\n')
    print(text)

if __name__=='__main__':main()
