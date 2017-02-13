from _expokit import dgexpv, dsexpv, zgexpv, zhexpv
import numpy as np
from scipy.sparse import random
from scipy.sparse.linalg import onenormest,expm_multiply,aslinearoperator




n=100
m=20
time = 10.0
iflag = np.array([1])
tol = 0.0
v = np.zeros(n,dtype=complex)
v[0]=1
v /= np.linalg.norm(v)
A = random(n,n,format="csr")
A = -1j*((A.T + A)/2)
anorm = onenormest(A)
wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=complex)
iwsp = np.zeros(m+2,dtype=int)

A_op = aslinearoperator(A)

output_vec,tol0,iflag0 = zgexpv(m,time,v,tol,anorm,wsp,iwsp,A_op.matvec,0)

exact_vec = expm_multiply(time*A,v)

print iflag0
print np.linalg.norm(output_vec-exact_vec)
