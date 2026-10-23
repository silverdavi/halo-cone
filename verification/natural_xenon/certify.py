"""Arb interval certificate for specified natural-Xe exclusions.
Run python certify.py. Requires python-flint (tested 0.9.0).
All physical decimal inputs are exact decimals enclosed by Arb.
No numerical quadrature, floating-point sign decisions, or mass-grid
interpolation are used. Whole mass cells are evaluated as balls.
"""
from flint import arb as B, ctx
from pathlib import Path
from fractions import Fraction
import csv, hashlib, json
ctx.prec=192
HERE=Path(__file__).resolve().parent
A=[124,126,128,129,130,131,132,134,136]
ATOMIC=['123.9058920','125.9042983','127.9035310','128.9047808611',
 '129.903509349','130.90508406','131.9041550856','133.90539466','135.907214484']
RAW_F=['.00095','.00089','.0191','.264','.0407','.212','.269','.104','.0886']
M=[B(x)*B('.93149410242')-54*B('.00051099895') for x in ATOMIC]
f=[B(x)/sum(map(B,RAW_F)) for x in RAW_F]
Q=[B(a-54)-(1-4*B('.23857'))*54 for a in A]
C=B('299792.458'); EARTH=B(232); S=B(776); U=B('16.36')
HBARC=B('.1973269804')
R=[((B('1.23')*B(a).root(3)-B('.60'))**2+
    7*B.pi()**2*B('.52')**2/3-5*B('.9')**2).sqrt() for a in A]
raw=(HERE/'b8_spectrum.csv').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='b07b3ba052c08cf675c8fcc5c5f674e0946390b721a0ea4da719077bdf5bc892'
rows=[[B(v) for v in row] for row in csv.reader(raw.decode().splitlines())]
X=[r[0] for r in rows]; Y=[r[1] for r in rows]
norm=sum((X[i+1]-X[i])*(Y[i+1]+Y[i])/2 for i in range(len(X)-1))
Y=[y/norm for y in Y]
sl=[(Y[i+1]-Y[i])/(X[i+1]-X[i]) for i in range(len(X)-1)]
ic=[Y[i]-sl[i]*X[i] for i in range(len(sl))]
J0=[B(0) for _ in X]; J2=[B(0) for _ in X]
for i in range(len(sl)-1,0,-1):
 a,b=X[i:i+2]
 J0[i]=J0[i+1]+(b-a)*(Y[i]+Y[i+1])/2
 J2[i]=J2[i+1]+sl[i]*(b/a).log()+ic[i]*(1/a-1/b)

def hull(a,b):
 lo=min(a.lower(),b.lower());hi=max(a.upper(),b.upper())
 return B((lo+hi)/2,(hi-lo)/2)

def point_values(q):
 if q>=U:return B(0),B(0),B(0)
 # Float is used only to propose a segment; rigorous comparisons verify it.
 j=int(float(q)*50)
 j=max(1,min(j,len(sl)-1))
 while q<X[j]:j-=1
 while q>X[j+1]:j+=1
 if not (q>=X[j] and q<=X[j+1]):
  # A point ball straddling a table node: enclose both polynomial pieces.
  choices=[k for k in (j-1,j,j+1) if 1<=k<len(sl) and
           not(q.upper()<X[k].lower() or q.lower()>X[k+1].upper())]
 else: choices=[j]
 values=[]
 for k in choices:
  b=X[k+1];s,c=sl[k],ic[k]
  t2=J2[k+1]+s*(b/q).log()+c*(1/q-1/b)
  t0=J0[k+1]+s*(b*b-q*q)/2+c*(b-q)
  values.append((t0-q*q*t2,-2*q*t2,2*(s*q+c)/q-2*t2))
 out=values[0]
 for v in values[1:]:out=tuple(hull(x,y) for x,y in zip(out,v))
 return out

def psi(q):
 # Psi is nonincreasing for every nonnegative input spectrum.
 return hull(point_values(q.upper())[0],point_values(q.lower())[0])

def helm(E,i):
 p=(2*M[i]*E*B('1e-6')).sqrt()/HBARC;z=p*R[i]
 sphere=3*(z.sin()-z*z.cos())/(z**3)
 return sphere**2*(-(p*B('.9'))**2).exp()

def coeff(E,i):return f[i]*M[i]*A[i]**2*helm(E,i)
def nu(E):return sum(f[i]*M[i]*Q[i]**2*helm(E,i)*psi((M[i]*E/2).sqrt()) for i in range(9))
def T(E):return nu(E)/coeff(E,0)
def ratio(E,i):return coeff(E,i)/coeff(E,0)

END=2*U**2/M[0];E0=B('.98')**2*END
def alphas2(m):
 # v_i/v_0 squared; avoids unnecessary square roots and mu dependency.
 return [M[0]/mi*((m+mi)/(m+M[0]))**2 for mi in M]

def five(m):
 z=alphas2(m);Bcut=1/B('.98')**2
 assert all(z[i+1]>z[i] for i in range(8))
 assert z[3]<Bcut and z[1]**2<Bcut
 assert z[4]>Bcut and z[1]**3>Bcut and z[1]*z[2]>Bcut
 return [T(E0),-ratio(E0,1)*T(z[1]*E0),
 ratio(E0,1)*ratio(z[1]*E0,1)*T(z[1]**2*E0),
 -ratio(E0,2)*T(z[2]*E0),-ratio(E0,3)*T(z[3]*E0)]

def lp_lpp(x,i):
 d=(2*M[i]*B('1e-6')).sqrt()/HBARC;k=d*R[i];z=k*x
 den=z.sin()-z*z.cos();u=z*z.sin()/den
 lp=2*k*(u-3/z)-2*(d*B('.9'))**2*x
 lpp=2*k*k*((z.sin()+z*z.cos())/den-u*u+3/(z*z))-2*(d*B('.9'))**2
 return lp,lpp

def curvature(E=B('.51'),p=B('.146'),r=B('.092')):
 x=E.sqrt()
 nus=[B(0),B(0),B(0)]
 bmin=None;cmax=None;tmin=None
 for i in range(9):
  lp,lpp=lp_lpp(x,i)
  bb=2*lp+p;cc=lpp+lp*lp+p*lp+r
  assert bb>0 and cc<0
  bmin=bb.lower() if bmin is None else min(bmin,bb.lower())
  cmax=cc.upper() if cmax is None else max(cmax,cc.upper())
  kap=(M[i]/2).sqrt();q=kap*x
  ps,ps1,ps2=point_values(q)
  component=kap*kap*ps2+bb*kap*ps1+cc*ps
  assert component>0
  tmin=component.lower() if tmin is None else min(tmin,component.lower())
  w=f[i]*M[i]*Q[i]**2*helm(E,i)
  nus[0]+=w*ps
  nus[1]+=w*(lp*ps+kap*ps1)
  nus[2]+=w*((lpp+lp*lp)*ps+2*lp*kap*ps1+kap*kap*ps2)
 L=nus[2]+p*nus[1]+r*nus[0]
 assert L>0
 return {'p':str(p),'r':str(r),'min_2lp_plus_p':str(bmin),'max_c':str(cmax),
         'min_target_component_after_dividing_weight':str(tmin),
         'nu_nux_nuxx':list(map(str,nus)),'Lnu':str(L)}

DUAL={0:'-0.0000001',86:'-2.5091846054218256e-08',89:'4.232602196532087e-07',
 92:'-3.12282552577074e-06',93:'-5.49029821603542e-05',94:'-0.0029039370021078214',
 97:'0.04322536788188634',98:'-0.49420890705022336',99:'0.45960331182823005'}

def positive_part(z):
 if z<0:return B(0)
 if z>0:return z
 return hull(B(0),z.upper())

def maxball(a,b):
 if a>b:return a
 if b>a:return b
 return hull(max(a.lower(),b.lower()),max(a.upper(),b.upper()))

def dual_certificate(m=B('6.0233'),dual=DUAL):
 energies=[B('.51')+B('2.99')*j/99 for j in dual]
 ds=list(map(B,dual.values()));tar=[nu(e) for e in energies]
 weights=[[coeff(e,i) for i in range(9)] for e in energies]
 speeds=[[C*(M[i]*e*B('1e-6')/2).sqrt()*(1/m+1/M[i]) for i in range(9)] for e in energies]
 # Below 400 km/s every positively weighted observation is identically zero.
 assert all(w>632 for row,d in zip(speeds,ds) if d>0 for w in row)
 # Breakpoints of shell responses; at fixed observations each response vector
 # is affine in 1/u between these breakpoints, so endpoint signs suffice.
 points=[B(400),B(544)]
 for row in speeds:
  for w in row:
   for u in [abs(w-EARTH),w+EARTH]:
    if u.upper()>400 and u.lower()<544:
     points.append(hull(max(B(400),u.lower()),min(B(544),u.upper())))
 worst=None
 for u in points:
  val=B(0)
  for d,wts,row,t in zip(ds,weights,speeds,tar):
   rate=sum(wt*positive_part(u+EARTH-maxball(w,abs(u-EARTH)))/(2*u*EARTH)
            for wt,w in zip(wts,row))
   val+=d*rate/t
  assert val<0, (m,u,val)
  worst=val.upper() if worst is None else max(worst,val.upper())
 assert sum(ds)>B('.00565')
 return {'mass_GeV':str(m),'indices_and_coefficients':dual,'energy_rule_keV':'.51+2.99*j/99',
         'target_functional':str(sum(ds)),'max_shell_functional_above_400':str(worst),
         'breakpoints_checked':len(points)}

def mass_for_velocity(E,i,v):
 mu=C*(M[i]*E*B('1e-6')/2).sqrt()/v
 return M[i]*mu/(M[i]-mu)

def main():
 result={'precision_bits':ctx.prec,'input_sha256':hashlib.sha256(raw).hexdigest(),
         'masses_GeV':list(map(str,M)),'raw_spectrum_integral':str(norm),
         'endpoint_keV':str(END),'E0_keV':str(E0),'terms_m7':list(map(str,five(B(7))))}
 result['sum_m7']=str(sum(five(B(7))))
 # Exact rational partition, covering [6.65,7.80] without gaps.
 max_upper=None;n=0
 stack=[(Fraction('6.65'),Fraction('7.80'))]
 while stack:
  a,b=stack.pop();mid=(a+b)/2;rad=(b-a)/2
  m=B(str(mid),str(rad))
  try:
   val=sum(five(m))
   good=val<0
  except AssertionError:good=False
  if not good:
   assert b-a>Fraction(1,10**7),'Subdivision failed'
   stack.extend([(a,mid),(mid,b)]);continue
  max_upper=val.upper() if max_upper is None else max(max_upper,val.upper());n+=1
 assert max_upper<B('-2.15e-10')
 result['mass_partition_cells']=n;result['uniform_upper_bound']=str(max_upper)
 result['curvature']=curvature()
 result['curvature_neighborhood']=curvature(B('.5100000005','.0000000005'))
 result['B2_counterexample']=dual_certificate()
 # A separate, coarse three-point certificate excludes a whole low-mass
 # interval. Together with curvature at .30 keV it proves emptiness for
 # every threshold <= .30 keV (at cutoff 3.5 keV).
 lowdual={0:'-0.0000001',93:'-0.18393443965754055',99:'0.8160655603424595'}
 stack=[(Fraction('5.764'),Fraction('5.893'))];nlow=0
 while stack:
  a,b=stack.pop();mid=(a+b)/2;rad=(b-a)/2
  try:
   dual_certificate(B(str(mid),str(rad)),lowdual);good=True
  except AssertionError:good=False
  if good:nlow+=1
  else:
   assert b-a>Fraction(1,10**8),'Low-mass dual subdivision failed'
   stack.extend([(a,mid),(mid,b)])
 result['low_mass_uniform_dual']={'interval_GeV':['5.764','5.893'],'cells':nlow,'coefficients':lowdual}
 result['curvature_at_030']=curvature(B('.3000000005','.0000000005'),B('.113'),B('.089'))
 result['upper_mass_at_030']=str(mass_for_velocity(B('.3'),8,EARTH))
 result['strict_low_mass']=str(mass_for_velocity(END,0,S))
 result['upper_mass_each']=list(map(str,[mass_for_velocity(B('.51'),i,EARTH) for i in range(9)]))
 result['cutoff_support_mass_each']=list(map(str,[mass_for_velocity(B('3.5'),i,S) for i in range(9)]))
 assert mass_for_velocity(END,0,S)>B('6.65')
 assert all(mass_for_velocity(B('.51'),i,EARTH)<B('7.80') for i in range(9))
 assert all(mass_for_velocity(B('3.5'),i,S)<B('6.0233') for i in range(9))
 assert mass_for_velocity(B('3.5'),0,S)>B('5.764')
 assert all(mass_for_velocity(B('.3'),i,EARTH)<B('5.893') for i in range(9))
 result['status']='ALL INTERVAL ASSERTIONS PASSED'
 print(json.dumps(result,indent=2));(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')

def _radii_ls(s, a_surf='.52', c_lead='1.23', c_shift='.60'):
 # Lewin & Smith eq. (4.11): R_1^2 = c^2 + (7/3) pi^2 a^2 - 5 s^2,
 # c = c_lead * A^{1/3} - c_shift. Nominal certificate values are the defaults.
 sB,aB=B(s),B(a_surf)
 return [((B(c_lead)*B(aa).root(3)-B(c_shift))**2
          +7*B.pi()**2*aB**2/3-5*sB**2).sqrt() for aa in A]

def _radii_rms(s, rms_lead):
 # r_rms^2 = (3/5) R_1^2 + 3 s^2, solved for the Helm diffraction radius.
 sB=B(s)
 out=[]
 for aa in A:
  rms=B(rms_lead)*B(aa).root(3)
  out.append((5*(rms**2-3*sB**2)/3).sqrt())
 return out

def _helm_w(E,i,radii,s):
 q=(2*M[i]*E*B('1e-6')).sqrt()/HBARC;z=q*radii[i]
 sphere=3*(z.sin()-z*z.cos())/(z**3)
 return sphere**2*(-(q*B(s))**2).exp()

def _helm_d(x,i,radii,s):
 d=(2*M[i]*B('1e-6')).sqrt()/HBARC;k=d*radii[i];z=k*x
 den=z.sin()-z*z.cos();u=z*z.sin()/den;sB=B(s)
 lp=2*k*(u-3/z)-2*(d*sB)**2*x
 lpp=2*k*k*((z.sin()+z*z.cos())/den-u*u+3/(z*z))-2*(d*sB)**2
 return lp,lpp

def _msq_from(radii,s):
 sB=B(s)
 return [(3*radii[i]**2)/5+3*sB**2 for i in range(9)]

def _lowq_w(E,i,msq):
 q2=2*M[i]*E*B('1e-6')/HBARC**2
 return 1-q2*msq[i]/3

def _lowq_d(x,i,msq):
 d=(2*M[i]*B('1e-6')).sqrt()/HBARC
 beta=msq[i]*d**2/3
 H=1-beta*x*x;Hp=-2*beta*x
 return Hp/H,(-2*beta)/H-Hp*Hp/(H*H)

def _curvature_of(E,p,r,deriv,weight):
 # Same tests as curvature(), for a caller-supplied form factor.
 # Comparisons are rigorous ball comparisons: True only if the whole ball lies on that side of zero.
 x=E.sqrt();nus=[B(0),B(0),B(0)]
 bmin=cmax=tmin=None
 flags={'ineq_2lp_plus_p':True,'ineq_c':True,'each_component':True}
 for i in range(9):
  lp,lpp=deriv(x,i)
  bb=2*lp+p;cc=lpp+lp*lp+p*lp+r
  if not bb>0:flags['ineq_2lp_plus_p']=False
  if not cc<0:flags['ineq_c']=False
  bmin=bb.lower() if bmin is None else min(bmin,bb.lower())
  cmax=cc.upper() if cmax is None else max(cmax,cc.upper())
  kap=(M[i]/2).sqrt();q=kap*x
  ps,ps1,ps2=point_values(q)
  component=kap*kap*ps2+bb*kap*ps1+cc*ps
  if not component>0:flags['each_component']=False
  tmin=component.lower() if tmin is None else min(tmin,component.lower())
  w=f[i]*M[i]*Q[i]**2*weight(E,i)
  nus[0]+=w*ps
  nus[1]+=w*(lp*ps+kap*ps1)
  nus[2]+=w*((lpp+lp*lp)*ps+2*lp*kap*ps1+kap*kap*ps2)
 L=nus[2]+p*nus[1]+r*nus[0]
 flags['Lnu']=bool(L>0)
 flags['passed']=all(flags.values())
 return {'p':str(p),'r':str(r),
         'min_2lp_plus_p':str(bmin),'max_c':str(cmax),
         'min_target_component_after_dividing_weight':str(tmin),
         'Lnu':str(L),'flags':flags}

def _agree(a,b):
 d=a-b
 return bool(d.contains(0) and d.rad()<B('1e-18'))

def _max_upper(balls):
 # A point ball at or above every input. Comparisons are ball comparisons.
 m=balls[0].upper()
 for b in balls[1:]:
  u=b.upper()
  if u>m: m=u
  elif not u<=m: m=hull(m,u).upper()
 return m

def _dec_above(x,places):
 # Exact decimal string strictly above the ball x. Float only proposes the start.
 scale=10**places
 n=int(float(x.upper())*scale)-1
 for _ in range(100000):
  n+=1
  s=f'{n/scale:.{places}f}'
  if B(s)>x: return s
 raise AssertionError('no decimal above')

def _dec_below(x,places):
 scale=10**places
 n=int(float(x.lower())*scale)+1
 for _ in range(100000):
  n-=1
  s=f'{n/scale:.{places}f}'
  if B(s)<x: return s
 raise AssertionError('no decimal below')

def rechoose(E,deriv,weight,places=8):
 # p = max_i(-2 l_i') + shift + delta, r = -max_i(l''+l'^2+p l') - delta,
 # then the certified curvature test. Smallest shift, then smallest delta, that passes.
 # Returns the passing point, or None with the corner evaluation if none does.
 x=E.sqrt()
 lps=[]; lpps=[]
 for i in range(9):
  lp,lpp=deriv(x,i)
  lps.append(lp); lpps.append(lpp)
 p_need=_max_upper([-2*lp for lp in lps])
 shifts=['0','1e-4','5e-4','1e-3','2e-3','5e-3','1e-2','2e-2','5e-2']
 deltas=['1e-6','1e-5','1e-4','1e-3']
 corner=None
 for sh in shifts:
  for de in deltas:
   p_s=_dec_above(p_need+B(sh)+B(de),places)
   p=B(p_s)
   gs=[lpps[i]+lps[i]**2+p*lps[i] for i in range(9)]
   r_s=_dec_below(-_max_upper(gs)-B(de),places)
   row=_curvature_of(E,B(p_s),B(r_s),deriv,weight)
   rec={'shift':sh,'delta':de,'p':p_s,'r':r_s,
        'min_2lp_plus_p':row['min_2lp_plus_p'],'max_c':row['max_c'],
        'Lnu':row['Lnu'],'flags':row['flags']}
   if corner is None: corner=rec
   if row['flags']['passed']:
    rec['feasible']=True
    rec['search']='smallest shift then smallest delta from p=max(-2 l_i\')+delta'
    return rec
 corner['feasible']=False
 corner['search']='no (shift, delta) in the scanned grid passed the ball test'
 return corner

def form_factor_variations():
 # Does not touch results.json. Default `python certify.py` is main() alone.
 # Nominal Helm path below is checked against helm() and curvature(), which the default run uses.
 radii_nom=_radii_ls('.9')
 for i in range(9):
  assert _agree(radii_nom[i],R[i])
  assert _agree(_helm_w(B('.51'),i,radii_nom,'.9'),helm(B('.51'),i))
  x=B('.51').sqrt()
  for u,v in zip(_helm_d(x,i,radii_nom,'.9'),lp_lpp(x,i)):
   assert _agree(u,v)
 nom_loop=_curvature_of(B('.51'),B('.146'),B('.092'),lp_lpp,helm)
 nom_ref=curvature()
 for k in ('min_2lp_plus_p','max_c','min_target_component_after_dividing_weight','Lnu'):
  assert nom_loop[k]==nom_ref[k],(k,nom_loop[k],nom_ref[k])
 msq=_msq_from(R,'.9')
 # Derivative check for the low-q expansion: analytic slope versus a centered difference.
 x=B('.51').sqrt();h=B('1e-6');i=4
 def lnH(xx):
  return _lowq_w(xx*xx,i,msq).log()
 fd=(lnH(x+h)-lnH(x-h))/(2*h)
 lp,_lpp=_lowq_d(x,i,msq)
 assert _agree(lp,fd) or (lp-fd).rad()<B('1e-8') and abs((lp-fd).mid())<B('1e-8')
 cases=[]
 def add(name,why,params,deriv,weight):
  row={'name':name,'why':why,'parameters':params}
  row['at_0.51']=_curvature_of(B('.51'),B('.146'),B('.092'),deriv,weight)
  row['neighborhood_0.51']=_curvature_of(B('.5100000005','.0000000005'),B('.146'),B('.092'),deriv,weight)
  row['at_0.30']=_curvature_of(B('.3000000005','.0000000005'),B('.113'),B('.089'),deriv,weight)
  cases.append(row)
  print(name,'0.51',row['at_0.51']['flags'],'0.30',row['at_0.30']['flags'])
 s1=_radii_ls('1')
 add('helm_skin_1.0',
     'Skin thickness s=1 fm, the value Lewin & Smith (1996) attribute to Engel and compare with their s=0.9 fm. '
     'R_1 is recomputed from their eq. (4.11) at a=0.52 fm and c=1.23*A^(1/3)-0.60 fm, so the rms radius fixed by c and a is unchanged. '
     'Shift is +0.1 fm in s.',
     {'s_fm':'1','a_fm':'.52','c':'1.23*A^(1/3)-0.60'},
     lambda x,i:_helm_d(x,i,s1,'1'),lambda E,i:_helm_w(E,i,s1,'1'))
 s08=_radii_ls('.8')
 add('helm_skin_0.8',
     'Same Lewin-Smith radius formula with s=0.8 fm: the 0.1 fm skin step in the other direction from the certificate value s=0.9 fm. '
     'Not a second value quoted by Lewin & Smith.',
     {'s_fm':'.8','a_fm':'.52','c':'1.23*A^(1/3)-0.60'},
     lambda x,i:_helm_d(x,i,s08,'.8'),lambda E,i:_helm_w(E,i,s08,'.8'))
 eR=_radii_rms('1','0.93')
 add('engel',
     'Helm model Lewin & Smith plot against theirs (Figs. 6 and 7): Engel fit r_rms=0.93*A^(1/3) fm, s=1 fm. '
     'R_1 from r_rms^2=(3/5)*R_1^2+3*s^2. Shifts both skin and radius.',
     {'s_fm':'1','r_rms':'0.93*A^(1/3)'},
     lambda x,i:_helm_d(x,i,eR,'1'),lambda E,i:_helm_w(E,i,eR,'1'))
 eRad=_radii_rms('.9','0.93')
 add('engel_radius_s0.9',
     'Engel radius r_rms=0.93*A^(1/3) fm at the certificate skin s=0.9 fm, to separate the radius shift from the skin shift. Hybrid, not a published model.',
     {'s_fm':'.9','r_rms':'0.93*A^(1/3)'},
     lambda x,i:_helm_d(x,i,eRad,'.9'),lambda E,i:_helm_w(E,i,eRad,'.9'))
 add('lowq_same_msq',
     'F^2=1-q^2*<r^2>/3 with <r^2>=(3/5)*R_1^2+3*s^2 from the certificate Helm parameters (s=0.9 fm, a=0.52 fm, c=1.23*A^(1/3)-0.60 fm). '
     'This is the first two terms of that Helm form factor. Appropriate at the curvature energies, where q is a fraction of an inverse fermi.',
     {'expansion':'1-q^2*<r^2>/3','<r^2>':'(3/5)*R_1^2+3*s^2','s_fm':'.9'},
     lambda x,i:_lowq_d(x,i,msq),lambda E,i:_lowq_w(E,i,msq))
 # Re-chosen (p, r). Fixed-point rows above are unchanged.
 # Neighborhood balls are the same ones curvature() uses, so a pass covers the point and a 1e-9 keV interval.
 re_specs={
  'engel':(lambda x,i:_helm_d(x,i,eR,'1'), lambda E,i:_helm_w(E,i,eR,'1'), True),
  'lowq_same_msq':(lambda x,i:_lowq_d(x,i,msq), lambda E,i:_lowq_w(E,i,msq), False),
 }
 re_E={'0.51':(B('.51'), B('.5100000005','.0000000005')),
       '0.30':(B('.30'), B('.3000000005','.0000000005'))}
 by={c['name']:c for c in cases}
 for name,(deriv,weight,both) in re_specs.items():
  energies=('0.51','0.30') if both else ('0.51',)
  chosen={}
  for tag in energies:
   point,neigh=re_E[tag]
   # Choose (p, r) on the neighborhood enclosure, then confirm the point.
   rec=rechoose(neigh,deriv,weight)
   rec['at_point']=_curvature_of(point,B(rec['p']),B(rec['r']),deriv,weight)
   chosen[tag]=rec
   print('rechoose',name,tag,'feasible',rec['feasible'],'p',rec['p'],'r',rec['r'],
         'neigh',rec['flags']['passed'],'point',rec['at_point']['flags']['passed'])
  by[name]['rechosen']=chosen
 out={'nominal_crosscheck':'generalized Helm at s=0.9 reproduces helm(), lp_lpp() and curvature()','cases':cases}
 text=json.dumps(out,indent=2)+'\n'
 (HERE/'form_factor_variations.json').write_text(text)
 print('wrote',HERE/'form_factor_variations.json')

if __name__=='__main__':
 import sys
 if sys.argv[1:]==['--form-factors']:form_factor_variations()
 else:main()
