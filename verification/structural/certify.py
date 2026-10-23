"""Arb certificate: structural robustness, skin bounds, and exact exclusions.
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
 assert sum(ds)>0
 return {'mass_GeV':str(m),'indices_and_coefficients':dual,'energy_rule_keV':'.51+2.99*j/99',
         'target_functional':str(sum(ds)),'max_shell_functional_above_400':str(worst),
         'breakpoints_checked':len(points)}

def mass_for_velocity(E,i,v):
 mu=C*(M[i]*E*B('1e-6')/2).sqrt()/v
 return M[i]*mu/(M[i]-mu)

def baseline_certificates():
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
 return result

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
 out={'nominal_crosscheck':'generalized Helm at s=0.9 reproduces helm(), lp_lpp() and curvature()','cases':cases}
 text=json.dumps(out,indent=2)+'\n'
 (HERE/'form_factor_variations.json').write_text(text)
 print('wrote',HERE/'form_factor_variations.json')

# ---------------------------------------------------------------------------
# Structural supplement. All output bounds below are ball or rational tests.
# The inherited fixed-coefficient tests are rerun, never trusted as JSON input.

def maxb(xs):
 xs=list(xs)
 return hull(max(x.lower() for x in xs),max(x.upper() for x in xs))

def minb(xs):
 xs=list(xs)
 return hull(min(x.lower() for x in xs),min(x.upper() for x in xs))

def abmax(x):return max(abs(x.lower()),abs(x.upper()))

def spectrum_box(q):
 """Valid for q>q_*: Psi and Psi' monotone; Psi'' peaks at mode 6.44."""
 assert q>B('3.181534858301')
 left=point_values(q.lower());right=point_values(q.upper())
 vals=[left,right]
 if q.contains(B('6.44')):vals.append(point_values(B('6.44')))
 return tuple(hull(min(v[j].lower() for v in vals),max(v[j].upper() for v in vals)) for j in range(3))

def model_functions(name):
 if name=='lowq':
  msq=_msq_from(R,'.9')
  return (lambda x,i:_lowq_d(x,i,msq)),(lambda E,i:_lowq_w(E,i,msq))
 if name=='engel':s='1';radii=_radii_rms(s,'.93')
 else:s=name.split('_')[1];radii=_radii_ls(s)
 return (lambda x,i:_helm_d(x,i,radii,s)),(lambda E,i:_helm_w(E,i,radii,s))

def target_jets(E,deriv,weight):
 x=E.sqrt();n=[B(0),B(0),B(0)];ls=[];lss=[]
 for i in range(9):
  si,ti=deriv(x,i);ls.append(si);lss.append(ti)
  k=(M[i]/2).sqrt();ps,d,dd=spectrum_box(k*x)
  w=f[i]*M[i]*Q[i]**2*weight(E,i)
  n[0]+=w*ps;n[1]+=w*(si*ps+k*d)
  n[2]+=w*((ti+si*si)*ps+2*si*k*d+k*k*dd)
 return n,ls,lss

def robust_model(name,threshold):
 deriv,weight=model_functions(name);E=B(threshold)
 n,ls,lss=target_jets(E,deriv,weight)
 a=-2*minb(ls);b=maxb(t+s*s for s,t in zip(ls,lss))
 dp=maxb(ls)-minb(ls);dd=maxb(lss)-minb(lss)
 assert a>0 and dp>0 and dd>0
 p=a+dp;r=-b+p*(a/2-dp)-dd
 Bnu=n[1]/n[0];Cnu=n[2]/n[0];gap=Cnu+p*Bnu+r
 assert gap>0
 # The exact constants are the formula above at the threshold. Ball-valued
 # p,r enclose those exact constants and are retained throughout the test.
 nb,ss,tt=target_jets(E+hull(B(0),B('1e-10')),deriv,weight)
 first=minb(2*s+p for s in ss)
 second=maxb(t+s*s+p*s+r for s,t in zip(ss,tt))
 L=nb[2]+p*nb[1]+r*nb[0]
 assert first>0 and second<0 and L>0
 # Exact feasibility test reduces to a one-dimensional concave envelope.
 # Here Bnu < min_i s_i, so its maximum is attained at p=a.
 assert Bnu<minb(ls)
 optimal=Cnu+minb(a*(Bnu-s)-t-s*s for s,t in zip(ls,lss))
 assert optimal>0
 return dict(model=name,threshold_keV=threshold,a=str(a),b=str(b),
  delta_prime=str(dp),delta_second=str(dd),p=str(p),r=str(r),
  nu_prime_over_nu=str(Bnu),nu_second_over_nu=str(Cnu),
  sufficient_gap=str(gap),optimal_point_gap=str(optimal),
  interval_E_keV=[threshold,str(E+B('1e-10'))],
  min_first=str(first),max_second=str(second),Lnu=str(L),feasible=True)

def moment_bounds(E,R2max):
 x=E.sqrt();t=2*maxb(M)*B('1e-6')*R2max/(3*HBARC**2)
 floor=1-t*x*x/2;assert floor>0
 beta1=2*t*x/floor;beta2=2*t/floor+2*(t*x/floor)**2
 return beta1,beta2,floor

def radius_certificate(threshold,cap):
 E=B(threshold)+hull(B(0),B('1e-10'));x=E.sqrt()
 b1,b2,fl=moment_bounds(E,B(cap));gaps=[]
 b1=b1.upper();b2=b2.upper() # Uniform constants on the entire interval.
 for mi in M:
  k=(mi/2).sqrt();ps,d,dd=spectrum_box(k*x)
  gaps.append(k*k*dd/ps-5*b1*k*abs(d)/ps-2*b2-8*b1*b1)
 assert minb(gaps)>0
 return dict(threshold_keV=threshold,maximum_mean_square_radius_fm2=cap,
  beta1=str(b1),beta2=str(b2),amplitude_lower_bound=str(fl),
  p=str(3*b1),r=str(-b2-5*b1*b1),
  min_component_L_over_component=str(minb(gaps)))

RP=list(map(B,['4.766','4.774','4.777','4.777','4.781','4.780','4.785','4.789','4.796']))
HMEV=HBARC*1000

def helm_amplitude_jets(q,rms,s=B('.9')):
 """F, dF/dq, d2F/dq2, physical momentum = 2q, q in MeV.
 Stable power series of 3 j1(z)/z; a geometric bound covers all tails.
 """
 import math
 k=2*(5*(rms*rms-3*s*s)/3).sqrt()/HMEV
 z=k*q;assert abmax(z)<2 and q>0
 sphere=[]
 for j in range(3):
  total=B(0)
  for n in range(20):
   if 2*n<j:continue
   coef=B((-1)**n*6*(n+1))/math.factorial(2*n+3)
   mult=math.prod(range(2*n-j+1,2*n+1))
   total+=coef*mult*k**j*z**(2*n-j)
  n=20;coef=B(6*(n+1))/math.factorial(2*n+3)
  mult=math.prod(range(2*n-j+1,2*n+1))
  # For |z|<2,n>=20,j<=2, consecutive absolute terms have ratio <1/4.
  tail=abmax(coef*mult*k**j*z**(2*n-j))*4/3
  sphere.append(total+B(0,tail))
 alpha=2*s*s/HMEV**2;g=(-alpha*q*q).exp()
 g1=-2*alpha*q;g2=4*alpha*alpha*q*q-2*alpha
 return sphere[0]*g,(sphere[1]+g1*sphere[0])*g,(sphere[2]+2*g1*sphere[1]+g2*sphere[0])*g

def rho_jets(q,skin,i):
 proton=helm_amplitude_jets(q,RP[i]);neutron=helm_amplitude_jets(q,RP[i]+skin)
 N=A[i]-54;eps=1-4*B('.23857')
 weak=[(N*neutron[j]-eps*54*proton[j])/Q[i] for j in range(3)]
 mass=[(N*neutron[j]+54*proton[j])/A[i] for j in range(3)]
 assert weak[0]>0 and mass[0]>0
 sw=weak[1]/weak[0];sm=mass[1]/mass[0]
 lp=2*(sw-sm)
 lpp=2*(weak[2]/weak[0]-sw*sw-mass[2]/mass[0]+sm*sm)
 return (weak[0]/mass[0])**2,lp,lpp+lp*lp

def skin_certificate():
 # A q interval slightly larger than [q_*+.1,U-.1]; exact skin range [0,.3].
 lo=Fraction('3.281534858300');hi=Fraction('16.34');step=Fraction('.02')
 skin=hull(B(0),B('.3'));worst=B(0);bounds1=B(0);bounds2=B(0);cells=0;generic=B(0)
 a=lo
 while a<hi:
  b=min(a+step,hi);q=B(str((a+b)/2),str((b-a)/2))
  ps,d,dd=spectrum_box(q);assert dd>0
  box_loss=(2*B('.00485')*abs(d)+B('.00051')*ps)/dd
  assert box_loss<1
  generic=max(generic,box_loss.upper())
  for i in range(9):
   rho,l1,l2=rho_jets(q,skin,i)
   z1=abmax(l1);z2=abmax(l2)
   loss=(2*z1*abs(d)+z2*ps)/dd
   assert loss<1,(a,i,loss)
   worst=max(worst,loss.upper());bounds1=max(bounds1,z1);bounds2=max(bounds2,z2)
  cells+=1;a=b
 # Last segment, including arbitrarily close approach to U, uses an analytic
 # inequality, not a grid sampling or division by a ball containing zero.
 q=hull(B('16.34'),U);tail=B(0)
 for i in range(9):
  rho,l1,l2=rho_jets(q,skin,i)
  bounds1=max(bounds1,abmax(l1));bounds2=max(bounds2,abmax(l2))
  loss=2*abmax(l1)*B('.02')+abmax(l2)*B('.02')**2/2
  assert loss<1;tail=max(tail,loss.upper())
 assert bounds1<B('.00485') and bounds2<B('.00051')
 qp=B('3.281534858300');qu=B('16.26')
 lower=minb([point_values(qp)[2]/2,point_values(qu)[2]/2])
 assert lower>B('6e-9')
 crude=(2*bounds1*abs(point_values(qp)[1])+bounds2*point_values(qp)[0])/(2*lower)
 return dict(model='separate proton/neutron Helm amplitudes; isoscalar DM; weak charges as input',
  skin_interval_fm=['0','.3'],proton_rms_fm=list(map(str,RP)),
  q_interval_MeV=[str(qp),'16.36'],q_cells=cells,
  max_abs_rho_prime_over_rho=str(bounds1),max_abs_rho_second_over_rho=str(bounds2),
  maximum_pointwise_fractional_curvature_loss=str(worst),
  last_segment_fractional_loss_upper=str(tail),
  generic_sufficient_derivative_box={'b1_per_MeV':'.00485','b2_per_MeV2':'.00051',
   'maximum_fractional_loss_before_last_segment':str(generic),
   'last_segment_loss_upper':str(2*B('.00485')*B('.02')+B('.00051')*B('.02')**2/2)},
  hprime_margin_on_requested_compact_interval=str(lower),
  crude_global_margin_loss_ratio=str(crude))

def local_margin_table():
 output=[]
 for threshold in ['.51','.30']:
  qmins=[];qmaxes=[];lower=[];cross=[];details=[]
  for i in range(9):
   q0=(M[i]*B(threshold)/2).sqrt()
   mu40=B(40)*M[i]/(40+M[i]);q40=1000*mu40*EARTH/C
   qend=hull(q0,q0+B('.01'))
   margin=spectrum_box(qend)[2]/2
   assert margin>0
   muU=U*C/(1000*EARTH);mU=M[i]*muU/(M[i]-muU)
   qmins.append(q0);qmaxes.append(q40);lower.append(margin);cross.append(mU)
   details.append(dict(A=A[i],q_threshold_MeV=str(q0),
    minus_hprime_at_threshold=str(point_values(q0)[2]/2),
    mass_where_J_reaches_U_GeV=str(mU),q_at_vE_for_m40_MeV=str(q40)))
  bound=min(v.lower() for v in lower)
  assert bound>B('.01204' if threshold=='.51' else '.00730')
  samples=[]
  for mass in ['7.7','8','10','15','20','25','30','40']:
   margins=[];empty=[]
   for i in range(9):
    q0=qmins[i];mu=B(mass)*M[i]/(B(mass)+M[i]);q1=1000*mu*EARTH/C
    if q1<=q0:empty.append(A[i]);continue
    if q1>=U:margins.append(B(0))
    else:margins.append(minb([point_values(q0)[2]/2,point_values(q1)[2]/2]))
   samples.append(dict(mass_GeV=mass,empty_J_isotopes=empty,
    minimum_over_nonempty_J=str(minb(margins))))
  output.append(dict(threshold_keV=threshold,by_isotope=details,
   q_threshold_range=[str(minb(qmins)),str(maxb(qmins))],
   uniform_local_minus_hprime_lower=str(bound),
   local_q_width_MeV='.01',
   full_J_uniform_margin_on_mass_range_7p7_to_40='0; J includes zero-target region at large masses',
   endpoint_crossing_mass_range_GeV=[str(minb(cross)),str(maxb(cross))],samples=samples))
 return output

def skin_moment_certificates():
 """Only positivity, spherical symmetry, second moments; no Helm shape."""
 output=[];eps=1-4*B('.23857')
 for threshold,cap in [('.51','69'),('.30','44')]:
  worst=B(0);b1max=B(0);b2max=B(0);deltas=[];rabs=[]
  for i in range(9):
   q=(M[i]*B(threshold)/2).sqrt()+hull(B(0),B('.01'))
   N=A[i]-54;rp=RP[i];rn=rp+B('.3')
   rw=(N*rn*rn+eps*54*rp*rp)/Q[i]
   rd=(N*rn*rn+54*rp*rp)/A[i]
   t1=4*rw/(3*HMEV**2);t2=4*rd/(3*HMEV**2)
   flo1=1-t1*q*q/2;flo2=1-t2*q*q/2
   assert flo1>0 and flo2>0
   u1=t1*q/flo1;u2=t2*q/flo2
   b1=2*(u1+u2)
   b2=2*(t1/flo1+u1*u1+t2/flo2+u2*u2)+b1*b1
   ps,d,dd=spectrum_box(q);loss=(2*b1*abs(d)+b2*ps)/dd
   assert loss<1
   worst=max(worst,loss.upper());b1max=max(b1max,b1.upper());b2max=max(b2max,b2.upper())
   assert rw<B(cap) and rd<B(cap)
   rabs.append(rw)
   deltas.append(((Q[i]*B(cap)-eps*54*rp*rp)/N).sqrt()-rp)
  output.append(dict(threshold_keV=threshold,skin_interval_fm=['0','.3'],
   local_q_width_MeV='.01',fractional_curvature_loss_upper=str(worst),
   max_abs_rho_prime_over_rho=str(b1max),max_abs_rho_second_over_rho=str(b2max),
   maximum_absolute_weak_second_moment_fm2=str(maxb(rabs)),
   mixture_upper_mass_bound_survives=True,
   sufficient_skin_cap_for_mixture_moment_operator_fm=str(minb(deltas)),
   caveat='Last cap is for the mixture operator, not for uniform global convexity or a physical uncertainty estimate.'))
 return output

def moving_five_cell(a,b):
 """Select one E0 at the rational cell midpoint, certify all masses in cell."""
 mid=(a+b)/2;m=B(str(mid),str((b-a)/2));zm=alphas2(B(str(mid)))
 zlo=maxb([zm[3],zm[1]**2]);zhi=minb([zm[4],zm[1]**3,zm[1]*zm[2]])
 cut=(zlo+9*zhi)/10;e0=END/cut
 z=alphas2(m)
 assert all(z[i+1]>z[i] for i in range(8))
 assert z[3]<cut and z[1]**2<cut
 assert z[4]>cut and z[1]**3>cut and z[1]*z[2]>cut
 assert e0>B('4.45')
 val=T(e0)-ratio(e0,1)*T(z[1]*e0)+ratio(e0,1)*ratio(z[1]*e0,1)*T(z[1]**2*e0)-ratio(e0,2)*T(z[2]*e0)-ratio(e0,3)*T(z[3]*e0)
 assert val<0
 return val,e0

def full_endpoint_threshold_bracket():
 upper=robust_model('helm_.9','4.45')
 assert mass_for_velocity(END,0,S)>B('6.65')
 assert all(mass_for_velocity(B('4.45'),i,EARTH)<27 for i in range(9))
 stack=[(Fraction('6.65'),Fraction(27))];count=0;worst=None;emin=None;emax=None
 while stack:
  a,b=stack.pop()
  try:val,e0=moving_five_cell(a,b)
  except AssertionError:
   assert b-a>Fraction(1,10**8)
   mid=(a+b)/2;stack.extend([(a,mid),(mid,b)]);continue
  count+=1;worst=val.upper() if worst is None else max(worst,val.upper())
  emin=e0.lower() if emin is None else min(emin,e0.lower())
  emax=e0.upper() if emax is None else max(emax,e0.upper())
 # An exact positive realizing halo at threshold 4.58: only 124Xe is active.
 m=B(7);eth=B('4.58');z=alphas2(m)
 assert all(eth>END/z[i] for i in range(1,9))
 assert all(eth>2*U*U/M[i] for i in range(1,9))
 w1=C*(M[0]*eth*B('1e-6')/2).sqrt()*(1/m+1/M[0])
 w2=C*U*B('.001')*(1/m+1/M[0])
 assert w1>EARTH and w2<S
 # q at threshold is past the mode: strict convexity is analytic there.
 assert (M[0]*eth/2).sqrt()>B('6.44')
 return dict(cutoff_scope='E_cut >= endpoint(124Xe), including 5 keV',
  critical_threshold_bracket_keV=['4.45','4.58'],
  meaning='empty for thresholds <=4.45; explicitly nonempty at 4.58',
  exclusion_mass_cells=count,exclusion_mass_interval_GeV=['6.65','27'],
  uniform_five_term_upper=str(worst),evaluation_energy_range=[str(emin),str(emax)],
  curvature_at_4p45=upper,witness_mass_GeV='7',witness_threshold_keV='4.58',
  witness_w1=str(w1),witness_w2=str(w2),
  largest_heavy_DM_onset_E=str(END/z[1]),
  largest_heavy_neutrino_endpoint_E=str(2*U*U/M[1]),
  single_isotope_critical_thresholds_keV=[str(2*U*U/mi*(EARTH/S)**2) for mi in M])

def low_edge_exclusion():
 # In coordinates t=1/m and u>=400>a, 2au times the functional is piecewise
 # affine in two variables. Its maximum is at a line-arrangement vertex.
 dual=dict(DUAL);dual[0]='-.001'
 tlo=1/B('6.0233');thi=1/B('5.764');ulo=B(400);uhi=B(544)
 terms=[];lines=[]
 for j,d in dual.items():
  e=B('.51')+B('2.99')*j/99;target=nu(e)
  for i in range(9):
   slope=C*(M[i]*e*B('1e-6')/2).sqrt();intercept=slope/M[i]
   if B(d)>0:assert slope*tlo+intercept>632
   terms.append((B(d)*coeff(e,i)/target,slope,intercept))
   for off in [-EARTH,EARTH]:
    line=(slope,intercept+off,j,i)
    if slope*thi+intercept+off<ulo or slope*tlo+intercept+off>uhi:continue
    lines.append(line)
 vertices=[(t,u) for t in [tlo,thi] for u in [ulo,uhi]]
 def add(t,u):
  if t< tlo or t>thi or u<ulo or u>uhi:return
  vertices.append((t,u))
 for s,c,j,i in lines:
  for t in [tlo,thi]:add(t,s*t+c)
  for u in [ulo,uhi]:add((u-c)/s,u)
 for n,(s,c,j,i) in enumerate(lines):
  for ss,cc,jj,ii in lines[:n]:
   if (j,i)==(jj,ii):continue  # Same slope, distinct parallel break lines.
   den=s-ss;assert not den.contains(0)
   t=(cc-c)/den;add(t,s*t+c)
 worst=None
 for t,u in vertices:
  value=B(0)
  for wt,s,c in terms:
   # clamp(z,0,464) = positive_part(z)-positive_part(z-464).
   z=u+EARTH-s*t-c
   value+=wt*(positive_part(z)-positive_part(z-2*EARTH))
  assert value<0,(t,u,value)
  worst=value.upper() if worst is None else max(worst,value.upper())
 assert sum(map(B,dual.values()))>0
 return dict(cutoff_keV='3.5',threshold_keV='.51',
  excluded_mass_interval_GeV=['0','6.0233'],arrangement_vertices=len(vertices),
  uniform_2au_times_dual_upper=str(worst),coefficients=dual,
  lower_edge_if_nonempty='infimum >=6.0233 GeV; this is not a two-sided existence bracket',
  existence='OPEN',exact_membership='OPEN')

def short_interval_member():
 """An exact realizing construction on [3.499,3.5], not a binned fit.
 Eight disjoint speed bands are affine; the ninth is defined by exact
 division of the target residual. Convex bridges connect all bands.
 """
 m=B(7);e0=B('3.499');e1=B('3.5');ec=B('3.4995')
 x0=e0.sqrt();x1=e1.sqrt();xc=ec.sqrt();t=B('.0831753');pivot=6
 lam=[C*(mi*B('1e-6')/2).sqrt()*(1/m+1/mi) for mi in M]
 assert all(lam[i+1]*x0>lam[i]*x1 for i in range(8))
 hs=[(-t*(l-lam[0])*xc).exp() for l in lam]
 amp=nu(ec)/sum(coeff(ec,i)*hs[i] for i in range(9))
 hs=[amp*h for h in hs]
 def branch(x,i):
  if i!=pivot:
   return hs[i]*(1-t*lam[i]*(x-xc)),-t*hs[i]*lam[i],B(0)
  n,ss,tt=target_jets(x*x,lp_lpp,helm);res=list(n)
  for j in range(9):
   if j==pivot:continue
   g=hs[j]*(1-t*lam[j]*(x-xc));gp=-t*hs[j]*lam[j]
   wt=coeff(x*x,j)
   res[0]-=wt*g
   res[1]-=wt*(ss[j]*g+gp)
   res[2]-=wt*((tt[j]+ss[j]**2)*g+2*ss[j]*gp)
  sp=ss[pivot];tp=tt[pivot];wp=coeff(x*x,pivot)
  return res[0]/wp,(res[1]-sp*res[0])/wp,(res[2]-2*sp*res[1]+(sp*sp-tp)*res[0])/wp
 # Split only the varying pivot band to avoid interval dependency inflation.
 gp_lower=None;gpp_lower=None;g_lower=None
 for j in range(100):
  ea=e0+(e1-e0)*j/100;eb=e0+(e1-e0)*(j+1)/100;x=hull(ea,eb).sqrt()
  g,gp,gpp=branch(x,pivot)
  assert g>0 and gp<0 and gpp>0
  g_lower=g.lower() if g_lower is None else min(g_lower,g.lower())
  gp_lower=gp.lower() if gp_lower is None else min(gp_lower,gp.lower())
  gpp_lower=gpp.lower() if gpp_lower is None else min(gpp_lower,gpp.lower())
 joins=[]
 for i in range(8):
  v,dx,_=branch(x1,i);vv,ddx,_=branch(x0,i+1)
  left=dx/lam[i];right=ddx/lam[i+1]
  gap=lam[i+1]*x0-lam[i]*x1;secant=(vv-v)/gap
  assert left<secant and secant<right and right<0
  joins.append(dict(A_left=A[i],A_right=A[i+1],speed_gap=str(gap),
   left_slope_slack=str(secant-left),right_slope_slack=str(right-secant)))
 for i in range(9):
  g,gp,_=branch(x1,i);assert g>0 and gp<0
 g,gp,_=branch(x1,8);tail_end=lam[8]*x1+g/(-gp/lam[8])
 assert lam[0]*x0>EARTH and tail_end<S
 return dict(mass_GeV='7',threshold_keV='3.499',cutoff_keV='3.5',
  exact_decay_parameter_per_kms='.0831753',pivot_A=132,
  construction='eight affine speed bands, exact residual on 132Xe band, convex linear bridges and a linear terminal tail',
  amplitude=str(amp),pivot_value_lower=str(g_lower),pivot_x_curvature_lower=str(gpp_lower),
  terminal_zero_speed_kms=str(tail_end),joins=joins,
  critical_threshold_bracket_for_cut_3p5_keV=['.3128','3.499'],
  sharp_threshold='OPEN; this bracket is intentionally non-sharp')

def table_certificate():
 mode=322;endpoint=818
 assert X[mode].contains(B('6.44')) and X[endpoint].contains(U)
 assert Y[0]==0 and Y[endpoint]==0
 assert all(y>0 for y in Y[1:endpoint])
 assert all(Y[i+1]>Y[i] for i in range(mode))
 # Decimal equalities are represented by overlapping balls; use original
 # rational ordinates for non-strict monotonicity and exact table identities.
 rational=list(csv.reader(raw.decode().splitlines()))
 ry=[Fraction(row[1]) for row in rational]
 assert all(ry[i+1]<=ry[i] for i in range(mode,endpoint))
 assert all(y==0 for y in ry[endpoint:])
 lo=B('3.181534858300');hi=B('3.181534858301')
 assert point_values(lo)[2]<0 and point_values(hi)[2]>0
 return dict(mode_MeV='6.44',endpoint_MeV='16.36',qstar_bracket_MeV=['3.181534858300','3.181534858301'],
  Estar_at_M122p3_keV=str(2*hull(lo,hi)**2/B('122.3')),
  maximum_Estar_Xe_keV=str(2*hi**2/minb(M)))

def run_supplement():
 out={'precision_bits':ctx.prec,'input_sha256':hashlib.sha256(raw).hexdigest(),
      'scope':'See PROOF.txt. All asserted signs use Arb; model assumptions are explicit.'}
 out['claim_status']={
  'proved':['All requested form-factor cases feasible by closed formula',
   'Moment-only upper bounds and skin robustness under explicit density assumptions',
   'No endpoint atom; corrected cutoff atom formula and value-slope condition',
   'All m<=6.0233 excluded at [.51,3.5]',
   'M(.51,5) is empty',
   'Full-endpoint critical threshold bracket [4.45,4.58]',
   'Truncated 3.5 critical threshold bracket [.3128,3.499]'],
  'open':['Nonemptiness and lower edge of M(.51,3.5)',
   'Finite iff for that membership set','Sharp critical thresholds'],
  'numerically_indicated_only':['User-reported binned cone windows; not used in proofs']}
 out['table']=table_certificate()
 out['A_models']=[robust_model(name,e) for e in ['.51','.30']
  for name in ['helm_.8','helm_.9','helm_1','engel','lowq']]
 out['A_moment_only']=[radius_certificate('.51','69'),radius_certificate('.30','44')]
 out['nominal_mean_square_radii_fm2']=list(map(str,_msq_from(R,'.9')))
 out['B_skin']=skin_certificate()
 out['B_J_margins']=local_margin_table()
 out['B_shape_independent_skin']=skin_moment_certificates()
 out['C_full_endpoint_threshold']=full_endpoint_threshold_bracket()
 out['C_truncated_lower_exclusion']=low_edge_exclusion()
 out['C_short_interval_member']=short_interval_member()
 out['C_truncated_threshold_lower_operator']=robust_model('helm_.9','.3128')
 assert mass_for_velocity(B('.3128'),8,EARTH)<B('6.0233')
 out['C_truncated_threshold_lower_upper_mass']=str(mass_for_velocity(B('.3128'),8,EARTH))
 out['baseline']=baseline_certificates()
 out['status']='ALL ASSERTED INTERVAL TESTS PASSED; OPEN QUESTIONS ARE NOT CERTIFIED'
 (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
 print('All asserted interval tests passed; wrote results.json.')

if __name__=='__main__':run_supplement()

