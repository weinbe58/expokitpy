import _expokit as _ek 
import numpy as np
from scipy.sparse.linalg import aslinearoperator,onenormest


__all__ = ["dsexpv","dgexpv","zhexpv","zgexpv"]

class ExpokitError(Exception):
    pass 


messages = {"maximum number of steps reached without convergence":1,
            "requested tolerance was too high":2}




def dsexpv(v,A,anorm=None,wsp=None,iwsp=None,m=20,t=1.0,tol=0.0,return_anorm=False):
    A = aslinearoperator(A)

    v = v.astype(np.float64,casting="safe",copy=False).reshape((-1,))
    
    n = v.shape[0]

    if A.shape[1] != A.shape[0]:
        raise ValueError("Expecting square LinearOperator.")

    if A.shape[1] != v.shape[0]:
        raise ValueError("Dimension mismatch between LinearOperator and input vector.")

    if anorm is None:
        anorm=onenormest(A)

    if wsp is None:
        wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=np.float64)

    if iwsp is None:
        iwsp = np.zeros(m+2,dtype=np.int32)


    u,tol,iflag = _ek.dsexpv(m,t,v,tol,anorm,wsp,iwsp,A.matvec,0)



    if iflag > 0:
        raise ExpokitError(messages[iflag])
    elif iflag < 0:
        raise ExpokitError("bad input arguments")

    if return_anorm:
        return u, anorm
    else:
        return u

def dgexpv(v,A,anorm=None,wsp=None,iwsp=None,m=20,t=1.0,tol=0.0,return_anorm=False):
    A = aslinearoperator(A)

    v = v.astype(np.float64,casting="safe",copy=False).reshape((-1,))
    
    n = v.shape[0]

    if A.shape[1] != A.shape[0]:
        raise ValueError("Expecting square LinearOperator.")

    if A.shape[1] != v.shape[0]:
        raise ValueError("Dimension mismatch between LinearOperator and input vector.")

    if anorm is None:
        anorm=onenormest(A)

    if wsp is None:
        wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=np.float64)

    if iwsp is None:
        iwsp = np.zeros(m+2,dtype=np.int32)


    u,tol,iflag = _ek.dgexpv(m,t,v,tol,anorm,wsp,iwsp,A.matvec,0)



    if iflag > 0:
        raise ExpokitError(messages[iflag])
    elif iflag < 0:
        raise ExpokitError("bad input arguments")

    if return_anorm:
        return u, anorm
    else:
        return u




def zhexpv(v,A,anorm=None,wsp=None,iwsp=None,m=20,t=1.0,tol=0.0,return_anorm=False):
    A = aslinearoperator(A)

    v = v.astype(np.complex128,casting="safe",copy=False).reshape((-1,))
    
    n = v.shape[0]

    if A.shape[1] != A.shape[0]:
        raise ValueError("Expecting square LinearOperator.")

    if A.shape[1] != v.shape[0]:
        raise ValueError("Dimension mismatch between LinearOperator and input vector.")

    if anorm is None:
        anorm=onenormest(A)

    if wsp is None:
        wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=np.complex128)

    if iwsp is None:
        iwsp = np.zeros(m+2,dtype=np.int32)


    u,tol,iflag = _ek.zhexpv(m,t,v,tol,anorm,wsp,iwsp,A.matvec,0)



    if iflag > 0:
        raise ExpokitError(messages[iflag])
    elif iflag < 0:
        raise ExpokitError("bad input arguments")

    if return_anorm:
        return u, anorm
    else:
        return u

def zgexpv(v,A,anorm=None,wsp=None,iwsp=None,m=20,t=1.0,tol=0.0,return_anorm=False):
    A = aslinearoperator(A)

    v = v.astype(np.complex128,casting="safe",copy=False).reshape((-1,))
    
    n = v.shape[0]

    if A.shape[1] != A.shape[0]:
        raise ValueError("Expecting square LinearOperator.")

    if A.shape[1] != v.shape[0]:
        raise ValueError("Dimension mismatch between LinearOperator and input vector.")

    if anorm is None:
        anorm=onenormest(A)

    if wsp is None:
        wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=np.complex128)

    if iwsp is None:
        iwsp = np.zeros(m+2,dtype=np.int32)


    u,tol,iflag = _ek.zgexpv(m,t,v,tol,anorm,wsp,iwsp,A.matvec,0)



    if iflag > 0:
        raise ExpokitError(messages[iflag])
    elif iflag < 0:
        raise ExpokitError("bad input arguments")

    if return_anorm:
        return u, anorm
    else:
        return u


















