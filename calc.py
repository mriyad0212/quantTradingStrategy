import numpy as np
import pandas as pd
import numpy_ext as npe

def union_idx_op(op,insts,al,win,*chress):
    res=[]
    chidxs=[[] for _ in range(len(chress))]
    for inst in insts:
        for chidx,chres in zip(chidxs,chress):
            chidx.append(chres[inst].dropna().index)
    opdf = op(*[chres.fillna(method="ffill") for chres in chress])
    for inst, *chidx in zip(insts,*chidxs):
        i_res = opdf[inst].loc[sorted(set().union(*chidx))]
        res.append(al.join(i_res))
    return pd.concat(res,axis=1)

def slow_idx_op(op,insts,al,win,*chress):
    res=[]
    for inst in insts:
        chids=[chres[inst].dropna().index for chres in  chress]
        lis=[len(chi) for chi in chids]
        sidx=chids[np.argmin(lis)]
        operands=[
            chres[inst].fillna(method="ffill").loc[sidx] for chres in chress
        ]
        rollop=npe.rolling_apply(op,win,*[operand.values for operand in operands])
        inst_res=pd.Series(data=rollop,index=sidx,name=inst)
        res.append(al.join(inst_res))
    return pd.concat(res,axis=1)

def self_idx_op(op,insts,al,win,*chress):
    res=[]
    for inst in insts:       
        instop=chress[0][inst].dropna().rolling(win).apply(op,raw=True)
        res.append(al.join(instop))
    return pd.concat(res,axis=1)

def self_idx_op2(op,insts,al,win,*chress):
    return op(chress[0])

def all_idx_op(op,insts,al,win,*chress):
    return chress[0].fillna(method="ffill").apply(op,axis=1).apply(pd.Series)