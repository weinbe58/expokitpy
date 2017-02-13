from ._expokit import dgexpv, dsexpv, zgexpv, zhexpv
import numpy as np
from scipy.sparse.linalg import aslinearoperator,onenormest


__all__ = ["py_dsexpv","py_dgexpv","py_zhexpv","py_zgexpv"]

class ExpokitError(Exception):
    pass 


messages = {"maximum number of steps reached without convergence":1,
            "requested tolerance was too high":2}




def py_dsexpv(v,A,anorm=None,wsp=None,iwsp=None,m=20,t=1.0,tol=0.0,return_work=False):
    A = aslinearoperator(A)

    v = v.astype(np.float64,casting="safe",copy=False).ravel()
    
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

    if return_work:
        return dict(anorm=anorm,wsp=wsp,iwsp=iwsp,m=m,return_work=False)

    u,tol0,iflag0 = dsexpv(m,t,v,tol,anorm,wsp,iwsp,A.matvec,0)



    if iflag0 > 0:
        raise ExpokitError(messages[iflag])
    elif iflag0 < 0:
        raise ExpokitError("bad input arguments")


    return u



def py_dgexpv(v,A,anorm=None,wsp=None,iwsp=None,m=20,t=1.0,tol=0.0,return_work=False):
    A = aslinearoperator(A)

    v = v.astype(np.float64,casting="safe",copy=False).ravel()
    
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

    if return_work:
        return dict(anorm=anorm,wsp=wsp,iwsp=iwsp,m=m,return_work=False)

    u,tol0,iflag0 = dgexpv(m,t,v,tol,anorm,wsp,iwsp,A.matvec,0)



    if iflag0 > 0:
        raise ExpokitError(messages[iflag0])
    elif iflag0 < 0:
        raise ExpokitError("bad input arguments")

    return u




def py_zhexpv(v,A,anorm=None,wsp=None,iwsp=None,m=20,t=1.0,tol=0.0,return_work=False):
    A = aslinearoperator(A)

    v = v.astype(np.complex128,casting="safe",copy=False).ravel()
    
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

    if return_work:
        return dict(anorm=anorm,wsp=wsp,iwsp=iwsp,m=m,return_work=False)


    u,tol0,iflag0 = zhexpv(m,t,v,tol,anorm,wsp,iwsp,A.matvec,0)



    if iflag0 > 0:
        raise ExpokitError(messages[iflag0])
    elif iflag0 < 0:
        raise ExpokitError("bad input arguments")


    return u




def py_zgexpv(v,A,anorm=None,wsp=None,iwsp=None,m=20,t=1.0,tol=0.0,return_work=False):
    A = aslinearoperator(A)

    v = v.astype(np.complex128,casting="safe",copy=False).ravel()
    
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

    if return_work:
        return dict(anorm=anorm,wsp=wsp,iwsp=iwsp,m=m,return_work=False)


    u,tol,iflag0 = zgexpv(m,t,v,tol,anorm,wsp,iwsp,A.matvec,0)

    if iflag0 > 0:
        raise ExpokitError(messages[iflag])
    elif iflag0 < 0:
        raise ExpokitError("bad input arguments")


    return u


















