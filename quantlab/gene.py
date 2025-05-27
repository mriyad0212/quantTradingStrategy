import scipy
import bottleneck
import numpy as np
import pandas as pd
from .utils import Alpha

class GeneticAlpha(Alpha):
    
    def __init__(self,insts,dfs,start,end,genome):
        super().__init__(insts,dfs,start,end)
        self.genome = genome

    def pre_compute(self,trade_range):
        alphadf=self.genome.evaluate_node(insts=self.insts,dfs=self.dfs,idx=trade_range)
        alphadf=alphadf.fillna(method="ffill")
        self.alphadf = alphadf
        return 
    
    def post_compute(self,trade_range):
        self.eligblesdf = self.eligiblesdf & (~pd.isna(self.alphadf))
        return 

    def compute_signal_distribution(self, eligibles, date):
        return self.alphadf.loc[date] 
    
class Gene():

    @staticmethod
    def str_to_gene(strgene):
        def index_of(c,s):
            return s.index(c) if c in s else -1

        if index_of("(",strgene) < 0:
            return Gene(
                prim=strgene.split("_")[0],
                space=strgene.split("_")[1] if index_of("_",strgene) >= 0 else None,
                is_terminal=True
            )

        prim_space = strgene[:index_of("(",strgene)]
        child_strs=[]

        count=0
        start = index_of("(",strgene)+1
        for pos in range(len(strgene)):
            ch = strgene[pos]
            count = count + 1 if ch == "(" else count
            if ch == "," and not "(" in strgene[start:pos]:
                child_strs.append(strgene[start:pos])
                start = pos + 1

            if ch == ")":
                count -= 1
                if count == 0:
                    child_strs.append(strgene[start:pos])
                    start = pos + 1
                if count == 1:
                    child_strs.append(strgene[start:pos+1])
                    start = pos + 2
        
        assert count == 0
        child_strs = [child for child in child_strs if child]
        children = [Gene.str_to_gene(child) for child in child_strs]
        gene = Gene(
            prim=prim_space.split("_")[0],
            space=prim_space.split("_")[1] if index_of("_", prim_space) >= 0 else None,
            is_terminal=False,parent=None,children=children
        )
        for child in gene.children:
            child.parent = gene
        return gene

    def __init__(self,prim,space,is_terminal,parent=None,children=[]):
        def _extract_numerics(s):
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except:
                    return s
        self.prim = prim
        self.space = _extract_numerics(space) if space else space
        self.is_terminal = is_terminal
        self.parent = parent
        self.children = children

    def evaluate_terminal(self,insts,dfs,idx):
        aligner = pd.DataFrame(index=idx)
        temp = []
        if self.prim == "const":
            for inst in insts:
                aligner[inst] = self.space
            return aligner
        if self.prim in ["open","high","low","close","volume"]:
            for inst in insts:
                temp.append(aligner.join(dfs[inst][self.prim]))
        else:
            for inst in insts:
                temp.append(aligner.join(dfs[inst + "_" + self.prim]))
        res = pd.concat(temp,axis=1)
        res.columns = insts
        return res
    
    def evaluate_function(self,insts,dfs,idx,chress):
        moment = lambda x,k: np.mean((x-np.mean(x))**k)
        stdmoment = lambda x,k: moment(x,k)/moment(x,2)**(k/2)
        
        from .calc import union_idx_op, self_idx_op, self_idx_op2, all_idx_op, slow_idx_op
        aligner = pd.DataFrame(index=idx)
        space = self.space

        if self.prim == "abs":
            func_type,op = self_idx_op2,np.abs
        if self.prim == "neg":
            func_type,op = self_idx_op2,lambda x:-1*x
        if self.prim == "log":
            func_type,op = self_idx_op2,np.log
        if self.prim == "sign":
            func_type,op = self_idx_op2,np.sign
        if self.prim == "recpcal":
            func_type,op = self_idx_op2,lambda x:1/x
        if self.prim == "pow":
            func_type,op = self_idx_op2,lambda x:np.power(x,space)

        if self.prim == "csrank":
            func_type,op = all_idx_op,lambda x:scipy.stats.rankdata(x,method="average",nan_policy="omit")
        if self.prim == "cszscre":
            func_type,op = all_idx_op,lambda x:(x-np.mean(x))/np.std(x)
        if self.prim == "ls":
            def ls_basket(x):
                is_short = x < np.nanpercentile(x,float(space.split("/")[0]))
                is_long = x > np.nanpercentile(x,float(space.split("/")[1]))
                return (-1*(0+is_short))+(0+is_long)
            func_type,op = all_idx_op,ls_basket

        if self.prim == "max":
            def max_dfs(*dfs):
                res=pd.DataFrame(np.maximum.reduce([df.values for df in dfs]))
                res.index=dfs[0].index
                res.columns=dfs[0].columns
                return res
            func_type,op = union_idx_op,max_dfs
        if self.prim == "plus":
            def plus_dfs(*dfs):
                res=pd.DataFrame(np.add.reduce([df.values for df in dfs]))
                res.index=dfs[0].index
                res.columns=dfs[0].columns
                return res
            func_type,op = union_idx_op,plus_dfs
        if self.prim == "minus":
            func_type,op = union_idx_op,lambda a,b: a-b
        if self.prim == "mult":
            func_type,op = union_idx_op,lambda a,b: a*b
        if self.prim == "div":
            func_type,op = union_idx_op,lambda a,b: a/b
        if self.prim == "and":
            func_type,op = union_idx_op,lambda a,b: np.logical_and(a,b)
        if self.prim == "or":
            func_type,op = union_idx_op,lambda a,b: np.logical_or(a,b)
        if self.prim == "eq":
            func_type,op = union_idx_op,lambda a,b: a.eq(b)
        if self.prim == "gt":
            func_type,op = union_idx_op,lambda a,b: a>b
        if self.prim == "lt":
            func_type,op = union_idx_op,lambda a,b: a<b
        if self.prim == "ite":
            func_type,op = union_idx_op,lambda a,b,c: a.fillna(0).astype(int)*b + (~a.astype(bool)).fillna(0).astype(int)*c
        
        if self.prim == "delta":
            space+=1
            func_type,op = self_idx_op, lambda a: a[-1]-a[0]
        if self.prim == "delay":
            space+=1
            func_type,op = self_idx_op, lambda a: a[0]
        if self.prim == "sum":
            func_type,op = self_idx_op, np.sum
        if self.prim == "prod":
            func_type,op = self_idx_op, np.prod
        if self.prim == "mean":
            func_type,op = self_idx_op, np.mean
        if self.prim == "median":
            func_type,op = self_idx_op, np.median
        if self.prim == "std":
            func_type,op = self_idx_op, np.std
        if self.prim == "var":
            func_type,op = self_idx_op, np.var
        if self.prim == "skew":
            func_type,op = self_idx_op, lambda r: stdmoment(r,3)
        if self.prim == "kurt":
            func_type,op = self_idx_op, lambda r: stdmoment(r,4)-3
        if self.prim == "tsrank":
            func_type,op = self_idx_op, lambda x: bottleneck.rankdata(x)[-1]
        if self.prim == "tsmax":
            func_type,op = self_idx_op, np.max
        if self.prim == "tsargmax":
            func_type,op = self_idx_op, np.argmax
        if self.prim == "tsmin":
            func_type,op = self_idx_op, np.min
        if self.prim == "tsargmin":
            func_type,op = self_idx_op, np.argmin
        if self.prim == "tszscre":
            func_type,op = self_idx_op, lambda x: (x[-1]-np.mean(x))/np.std(x)

        if self.prim == "cor":
            func_type,op = slow_idx_op, lambda a,b:np.corrcoef(a,b)[0][1]
        if self.prim == "kentau":
            func_type,op = slow_idx_op, lambda a,b:scipy.stats.kendalltau(a,b)[0]
        if self.prim == "cov":
            func_type,op = slow_idx_op, lambda a,b:np.cov(a,b)[0][1]

        equiv=None
        if self.prim == "grssret":
            equiv=Gene.str_to_gene(f"div(close,delay_{space}(close))")
        if self.prim == "logret":
            equiv=Gene.str_to_gene(f"log(grssret_{space}())")
        if self.prim == "netret":
            equiv=Gene.str_to_gene(f"minus(grssret_{space}(),const_1)")
        if self.prim == "volatility":
            equiv=Gene.str_to_gene(f"std_{space}(logret_1())")
        if self.prim == "obv":
            equiv=Gene.str_to_gene(f"sum_{space}(mult(volume,sign(netret_1())))")
        if self.prim == "addv":
            equiv=Gene.str_to_gene(f"mean_{space}(mult(volume,close))")
        if self.prim == "mac":
            arg=str(self)[str(self).index("("):][1:-1]
            p1,p2=space.split("/")[0],space.split("/")[1]
            equiv=Gene.str_to_gene(f"ite(gt(mean_{p1}({arg}),mean_{p2}({arg})),const_1,const_0)")
        if equiv:
            res=equiv.evaluate_node(insts,dfs,idx)
            return res
        
        res=func_type(op,insts,aligner,space,*chress)
        res.columns = list(insts)    
        return res
    
    def evaluate_node(self,insts,dfs,idx):
        chress = [child.evaluate_node(insts=insts,dfs=dfs,idx=idx) for child in self.children]
        if self.is_terminal:
            res = self.evaluate_terminal(insts,dfs,idx)
        else:
            res = self.evaluate_function(insts,dfs,idx,chress)
        res = res.replace([-np.inf, np.inf], np.nan)
        return res
    
    def __repr__(self):
        if self.is_terminal:
            res = self.prim
            res += f"_{self.space}" if (self.space or self.space == 0) else ""
            return res
        res = f'{self.prim}'
        res += f'_{self.space}' if (self.space or self.space == 0) else ""
        args = []
        for child in self.children:
            args.append(child.__repr__())
        return res + "(" + ",".join(args) + ")"

    #size, depth, height
    def size(self):
        if not self.children:
            return 1
        return 1 + np.sum([child.size() for child in self.children])

    def depth(self):
        if not self.parent:
            return 0
        return 1 + self.parent.depth()
    
    def height(self):
        if not self.children:
            return 0
        return 1 + np.max([child.height() for child in self.children])
    
    def make_dot(self):
        import graphviz
        gr = graphviz.Digraph()
        def _populate(tr,gr,pridx,rnidx):
            label=tr.prim
            label+=f"_{tr.space}" if (tr.space or tr.space == 0) else ""
            rnidx += 1
            gr.node(str(rnidx),label)
            if not pridx == -1: gr.edge(str(pridx),str(rnidx))
            new_pridx=rnidx
            for child in tr.children:
                gr,rnidx=_populate(tr=child,gr=gr,pridx=new_pridx,rnidx=rnidx)
            return gr,rnidx
        gr,rnidx=_populate(self,gr,-1,-1)
        return gr.source
    


'''
function_10()
terminal
func_10(func_15(),term3,func(term2))
'''