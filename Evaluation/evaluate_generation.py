# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 23:14:40 2023

@author: lyw
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 00:34:22 2023

@author: lyw
"""

import warnings
from multiprocessing import Pool
import numpy as np
from scipy.spatial.distance import cosine as cos_distance
from fcd_torch import FCD as FCDMetric
from scipy.stats import wasserstein_distance
import pandas as pd
from moses.dataset import get_dataset, get_statistics
from moses.utils import mapper
from moses.utils import disable_rdkit_log, enable_rdkit_log
from moses.metrics.utils import compute_fragments, average_agg_tanimoto, \
    compute_scaffolds, fingerprints, \
    get_mol, canonic_smiles, mol_passes_filters, \
    logP, QED, SA, weight




def fraction_passes_filters(gen, n_jobs=1):
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)


def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()


def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(
                "Can't compute unique@{}.".format(k) +
                "gen contains only {} molecules".format(len(gen))
            )
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        raise ValueError("Invalid molecule passed to unique@k")
    return len(canonic) / len(gen)


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return len(gen_smiles_set - train_set) / len(gen_smiles_set)

def get_uniq_novel(gen,train,n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    new_smiles_set = gen_smiles_set-train_set
    return new_smiles_set
    

def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]


class Metric:
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.values():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, moleclues):
        raise NotImplementedError

    def metric(self, pref, pgen):
        raise NotImplementedError


class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs,
                                    fp_type=self.fp_type)}

    def metric(self, pref, pgen):
        return average_agg_tanimoto(pref['fps'], pgen['fps'],
                                    device=self.device)


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between
     dictionaries of form {name: count}. Non-present
     elements are considered zero:

     sim = <r, g> / ||r|| / ||g||
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)


class FragMetric(Metric):
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['frag'], pgen['frag'])


class ScafMetric(Metric):
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['scaf'], pgen['scaf'])


class WassersteinMetric(Metric):
    def __init__(self, func=None, **kwargs):
        self.func = func
        super().__init__(**kwargs)

    def precalc(self, mols):
        if self.func is not None:
            values = mapper(self.n_jobs)(self.func, mols)
        else:
            values = mols
        return {'values': values}

    def metric(self, pref, pgen):
        return wasserstein_distance(
            pref['values'], pgen['values']
        )

def compute_intermediate_statistics(smiles, n_jobs=1, device='cpu',
                                    batch_size=512, pool=None):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.
    """
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    #kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    #statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(smiles)
    statistics['SNN'] = SNNMetric(**kwargs).precalc(mols)
    statistics['Frag'] = FragMetric(**kwargs).precalc(mols)
    statistics['Scaf'] = ScafMetric(**kwargs).precalc(mols)
    for name, func in [('logP', logP), ('SA', SA),
                       ('QED', QED),
                       ('weight', weight)]:
        statistics[name] = WassersteinMetric(func, **kwargs).precalc(mols)
    if close_pool:
        pool.terminate()
    return statistics
 
class SNNMetric(Metric):
    """
    Computes average max similarities of gen SMILES to ref SMILES
    """

    def __init__(self, fp_type='morgan', **kwargs):
        self.fp_type = fp_type
        super().__init__(**kwargs)

    def precalc(self, mols):
        return {'fps': fingerprints(mols, n_jobs=self.n_jobs,
                                    fp_type=self.fp_type)}

    def metric(self, pref, pgen):
        return average_agg_tanimoto(pref['fps'], pgen['fps'],
                                    device=self.device)
def pre_compute_statistics(smiles, n_jobs=1, device='cpu',
                                   ):

     statistics = {}
     mols = get_mol(smiles)
     
     #kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
     #statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(smiles)
     statistics['SNN'] = SNNMetric().precalc(mols)
     statistics['Frag'] = FragMetric().precalc(mols)
     statistics['Scaf'] = ScafMetric().precalc(mols)
   

     return statistics




def calculate_metrics(gen,test,test_scaffolds =None, ptest =None,ptest_scaffolds =None):
    
    validity =fraction_valid(gen)
    gen = remove_invalid(gen, canonize=True)
    canon_smiles = [canonic_smiles(s) for s in gen]
    unique_smiles = list(set(canon_smiles))
    novel_ratio = novelty(unique_smiles, test)   # replace 'source' with 'split' for moses
    mols = get_mol(gen)
    pgen = pre_compute_statistics(gen)

    metrics = {}
    if ptest is None:
        ptest =pre_compute_statistics(test)
    if test_scaffolds is not None and ptest_scaffolds is None:
        ptest_scaffolds = pre_compute_statistics(test_scaffolds)

    if ptest_scaffolds is not None:
        
        
        metrics['SNN/TestSF'] = SNNMetric().metric(ptest_scaffolds['SNN'], pgen['SNN'])
        metrics['Frag/TestSF'] = FragMetric().metric(ptest_scaffolds['Frag'], pgen['Frag'])
        metrics['Scaf/TestSF'] = ScafMetric().metric(ptest_scaffolds['Scaf'], pgen['Scaf'])
   
    metrics['validity'] = validity 
    metrics['uniqueness'] = len(unique_smiles)/len(gen)
    metrics['novelty'] = novel_ratio 
    
    metrics['IntDiv'] = internal_diversity(mols)
    metrics['IntDiv2'] = internal_diversity(mols, p=2)
    metrics['Filters'] = fraction_passes_filters(mols)
    metrics['FCD/Test'] = FCDMetric().metric(gen=gen, pref=ptest['FCD'])
    
    metrics['SNN/Test'] = SNNMetric().metric(ptest['SNN'], pgen['SNN'])
    metrics['Frag/Test'] = FragMetric().metric(ptest['Frag'], pgen['Frag'])
    metrics['Scaf/Test'] = ScafMetric().metric(ptest['Scaf'], pgen['Scaf'])

    return metrics

import os
folder = os.getcwd() + '//***//' #'*'为指定评价结果输出的文件夹
if not os.path.exists(folder):
    os.makedirs(folder)    
path_gen = os.getcwd()+'//***//' #'*'为存放生成SMILES的文件夹
path_test = os.getcwd()+'//***//' #'*'为存放训练集(真实数据)SMILES的文件夹
path_list = os.listdir(path_gen)
data_sum = []
for i in range(0,len(path_list)):
    print(path_gen+path_list[i])
    result = {}
    title = path_list[i] 
    title = title.split("_")
    EP = title[4]
    gen = pd.read_csv(path_gen+path_list[i],header=None,names=['SMILES'])
    gen = gen['SMILES']
    test = pd.read_csv(path_test+'***.csv',header=None,names=['SMILES']) # '*'为训练集文件名
    test = test['SMILES']    
    result =calculate_metrics(gen,test)     
    gen = remove_invalid(gen, canonize=True)
    gen.to_csv(folder+'gen_valid_{}.csv'.format(EP))
    gen = get_uniq_novel(gen,test)
    gen = pd.DataFrame(gen)
    gen.to_csv(folder+'gen_uniq_{}.csv'.format(EP)) #输出每条件下生成结果的处理结果
    data_sum.append(result)
data_sum =pd.DataFrame(data_sum)
data_sum.to_csv(folder+'data_sum.csv') #评价结果汇总

