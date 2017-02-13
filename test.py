from _expokit import dsexpv,zgexpv
import numpy as np
from scipy.sparse import random
from scipy.sparse.linalg import onenormest




n=1000
m=100

v = np.random.ranf(n)
v /= np.linalg.norm(v)
A = random(n,n,format="csr")


A = -1j*(A.T + A)/2
anorm = onenormest(A)
wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=np.float64)
iwsp = np.zeros(m+2,dtype=np.int32)


u,tol,iflag = zgexpv(m,1.0,v,0.0,anorm,wsp,iwsp,A.dot,0)

print iflag
print u,tol




