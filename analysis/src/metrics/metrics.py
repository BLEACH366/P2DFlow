import os
from typing import *

import numpy as np
import torch
from scipy.spatial import distance
from deeptime.decomposition import TICA
from src.common.geo_utils import rmsd, _find_rigid_alignment, squared_deviation
from scipy.linalg import fractional_matrix_power
from sklearn.mixture import GaussianMixture
from Bio.PDB import PDBParser
import freesasa
from Bio.PDB.Polypeptide import PPBuilder
import multiprocessing as mp

EPS = 1e-12
PSEUDO_C = 1e-6


def adjacent_ca_distance(coords):
    """Calculate distance array for a single chain of CA atoms. Only k=1 neighbors.
    Args:
        coords: (..., L, 3)
    return 
        dist: (..., L-1)
    """    
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    dX = coords[..., :-1, :] - coords[..., 1:, :] # (..., L-1, 3)
    dist = np.sqrt(np.sum(dX**2, axis=-1))
    return dist # (..., L-1)


def distance_matrix_ca(coords):
    """Calculate distance matrix for a single chain of CA atoms. W/o exclude neighbors.
    Args:
        coords: (..., L, 3)
    Return:
        dist: (..., L, L)
    """
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    dX = coords[..., None, :, :] - coords[..., None, :] # (..., L, L, 3)
    dist = np.sqrt(np.sum(dX**2, axis=-1))
    return dist # (..., L, L)


def pairwise_distance_ca(coords, k=1):
    """Calculate pairwise distance vector for a single chain of CA atoms. W/o exclude neighbors.
    Args:
        coords: (..., L, 3)
    Return:
        dist: (..., D) (D=L * (L - 1) // 2) when k=1)
    """
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    dist = distance_matrix_ca(coords)
    L = dist.shape[-1]
    row, col = np.triu_indices(L, k=k)
    triu = dist[..., row, col]  # unified (but unclear) order
    return triu # (..., D)


def radius_of_gyration(coords, masses=None):
    """Compute the radius of gyration for every frame.
    
    Args:
        coords: (..., num_atoms, 3)
        masses: (num_atoms,)
        
    Returns:
        Rg: (..., )
        
    If masses are none, assumes equal masses.
    """
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    
    if masses is None:
        masses = np.ones(coords.shape[-2])
    else:
        assert len(masses.shape) == 1, f"masses should be 1D, got {masses.shape}"
        assert masses.shape[0] == coords.shape[-2], f"masses {masses.shape} != number of particles {coords.shape[-2]}"

    weights = masses / masses.sum()
    centered = coords - coords.mean(-2, keepdims=True) 
    squared_dists = (centered ** 2).sum(-1)
    Rg = (squared_dists * weights).sum(-1) ** 0.5
    return Rg


def _steric_clash(coords, ca_vdw_radius=1.7, allowable_overlap=0.4, k_exclusion=0):
    """ https://www.schrodinger.com/sites/default/files/s3/public/python_api/2022-3/_modules/schrodinger/structutils/interactions/steric_clash.html#clash_iterator
    Calculate the number of clashes in a single chain of CA atoms.
    
    Usage: 
        n_clash = calc_clash(coords)
    
    Args:
        coords: (n_atoms, 3), CA coordinates, coords should from one protein chain.
        ca_vdw_radius: float, default 1.7.
        allowable_overlap: float, default 0.4.
        k_exclusion: int, default 0. Exclude neighbors within [i-k-1, i+k+1].
        
    """
    assert np.isnan(coords).sum() == 0, "coords should not contain nan"
    assert len(coords.shape) in (2, 3), f"CA coords should be 2D or 3D, got {coords.shape}" # (B, L, 3)
    assert k_exclusion >= 0, "k_exclusion should be non-negative"
    bar = 2 * ca_vdw_radius - allowable_overlap
    # L = len(coords)
    # dist = np.sqrt(np.sum((coords[:L-k_exclusion, None, :] - coords[None, k_exclusion:, :])**2, axis=-1))   
    pwd = pairwise_distance_ca(coords, k=k_exclusion+1) # by default, only excluding self (k=1)

    # print('val_clash')
    # print(pwd.shape)
    # print(pwd.max(),pwd.min())
    # idx_min=-1
    # smin=10
    # for idx,pwd_single in enumerate(pwd):
    #     if pwd_single.min()<smin:
    #         smin=pwd_single.min()
    #         idx_min=idx+1
    # print('smin=',smin)
    # print('idx_min=',idx_min)
    # if pwd.shape[0]==250:
    #     print(np.min(pwd, axis=-1))


    assert len(pwd.shape) == 2, f"pwd should be 2D, got {pwd.shape}"
    n_clash = np.sum(pwd < bar, axis=-1)
    return n_clash.astype(int) #(..., )  #np.prod(dist.shape)


def validity(ca_coords_dict, **clash_kwargs):
    """Calculate clash validity of ensembles. 
    Args:
        ca_coords_dict: {k: (B, L, 3)}
    Return:
        valid: {k: validity in [0,1]}
    """
    num_residue = float(ca_coords_dict['target'].shape[1])
    n_clash = {
        k: _steric_clash(v, **clash_kwargs)
            for k, v in ca_coords_dict.items()
    }
    # results = {
    #     k: 1.0 - (v>0).mean() for k, v in n_clash.items()
    # }
    results = {
        k: 1.0 - (v/num_residue).mean() for k, v in n_clash.items()
    }

    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results


def bonding_validity(ca_coords_dict, ref_key='target', eps=1e-6):
    """Calculate bonding dissociation validity of ensembles."""
    adj_dist = {k: adjacent_ca_distance(v)
            for k, v in ca_coords_dict.items()
    }
    thres = adj_dist[ref_key].max()+ 1e-6

    # print('val_bond')
    # print('target')
    # print(adj_dist['target'].shape)
    # print(adj_dist['target'].max(),adj_dist['target'].min())
    # print('pred')
    # print(adj_dist['pred'].shape)
    # print(adj_dist['pred'].max(),adj_dist['pred'].min())

    # idx_max=-1
    # smax=0
    # for idx,adj in enumerate(adj_dist['pred']):
    #     if adj.max()>smax:
    #         smax=adj.max()
    #         idx_max=idx+1
    # print('smax=',smax)
    # print('idx_max=',idx_max)
    # max_index2 = np.argmax(adj_dist['pred'], axis=-1)
    # print('res_max_index_all',max_index2+1)
    # print('res_max=',max_index2[idx_max-1]+1)

    # results = {
    #     k: (v < thres).all(-1).sum().item() / len(v) 
    #         for k, v in adj_dist.items()
    # }
    results = {
        k: (v < thres).mean()
            for k, v in adj_dist.items()
    }

    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results


def js_pwd(ca_coords_dict, ref_key='target', n_bins=50, pwd_offset=3, weights=None):
    # n_bins = 50 follows idpGAN
    # k=3 follows 2for1
    
    ca_pwd = {
        k: pairwise_distance_ca(v, k=pwd_offset) for k, v in ca_coords_dict.items()
    }   # (B, D)
    
    if weights is None:
        weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
        
    d_min = ca_pwd[ref_key].min(axis=0) # (D, )
    d_max = ca_pwd[ref_key].max(axis=0)
    ca_pwd_binned = {
        k: np.apply_along_axis(lambda a: np.histogram(a[:-2], bins=n_bins, weights=weights[k], range=(a[-2], a[-1]))[0]+PSEUDO_C, 0, 
                            np.concatenate([v, d_min[None], d_max[None]], axis=0))
        for k, v in ca_pwd.items()
    }   # (N_bins, D)-> (N_bins * D, )
    # js divergence per channel and average
    results = {k: distance.jensenshannon(v, ca_pwd_binned[ref_key], axis=0).mean() 
                    for k, v in ca_pwd_binned.items() if k != ref_key}
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results

def js_tica(ca_coords_dict, ref_key='target', n_bins=50, lagtime=20, return_tic=True, weights=None):
    # n_bins = 50 follows idpGAN
    ca_pwd = {
        k: pairwise_distance_ca(v) for k, v in ca_coords_dict.items()
    }   # (B, D)
    
    print('tica1', ca_pwd[ref_key].shape)
    estimator = TICA(dim=2, lagtime=lagtime).fit(ca_pwd[ref_key])
    print('tica2')
    tica = estimator.fetch_model()
    # dimension reduction into 2D
    ca_dr2d = {  
        k: tica.transform(v) for k, v in ca_pwd.items()
    }
    if weights is None: weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
    
    d_min = ca_dr2d[ref_key].min(axis=0) # (D, )
    d_max = ca_dr2d[ref_key].max(axis=0)
    ca_dr2d_binned = {
        k: np.apply_along_axis(lambda a: np.histogram(a[:-2], bins=n_bins, weights=weights[k], range=(a[-2], a[-1]))[0]+PSEUDO_C, 0, 
                            np.concatenate([v, d_min[None], d_max[None]], axis=0))
                for k, v in ca_dr2d.items()
    }   # (N_bins, 2) 
    results = {k: distance.jensenshannon(v, ca_dr2d_binned[ref_key], axis=0).mean() 
                for k, v in ca_dr2d_binned.items() if k != ref_key}
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    if return_tic:
        return results, ca_dr2d
    return results

def js_tica_pos(ca_coords_dict, ref_key='target', n_bins=50, lagtime=20, return_tic=True, weights=None):
    # n_bins = 50 follows idpGAN
    v_ref  = torch.as_tensor(ca_coords_dict['target'][0])
    for k, v in ca_coords_dict.items():
        v = torch.as_tensor(v)
        for idx in range(v.shape[0]):
            R, t = _find_rigid_alignment(v[idx], v_ref)
            v[idx] = (torch.matmul(R, v[idx].transpose(-2, -1))).transpose(-2, -1) + t.unsqueeze(0)
        ca_coords_dict[k] = v.numpy()

    ca_pos = { k: v.reshape(v.shape[0],-1) for k, v in ca_coords_dict.items()}   # (B, 3*N)
    
    estimator = TICA(dim=2, lagtime=lagtime).fit(ca_pos[ref_key])
    tica = estimator.fetch_model()
    # dimension reduction into 2D
    ca_dr2d = {  
        k: tica.transform(v) for k, v in ca_pos.items()
    }
    if weights is None: weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
    
    d_min = ca_dr2d[ref_key].min(axis=0) # (D, )
    d_max = ca_dr2d[ref_key].max(axis=0)
    ca_dr2d_binned = {
        k: np.apply_along_axis(lambda a: np.histogram(a[:-2], bins=n_bins, weights=weights[k], range=(a[-2], a[-1]))[0]+PSEUDO_C, 0, 
                            np.concatenate([v, d_min[None], d_max[None]], axis=0))
                for k, v in ca_dr2d.items()
    }   # (N_bins, 2) 
    results = {k: distance.jensenshannon(v, ca_dr2d_binned[ref_key], axis=0).mean() 
                for k, v in ca_dr2d_binned.items() if k != ref_key}
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    if return_tic:
        return results, ca_dr2d
    return results

def js_rg(ca_coords_dict, ref_key='target', n_bins=50, weights=None):
    ca_rg = {
        k: radius_of_gyration(v) for k, v in ca_coords_dict.items()
    }   # (B, )
    if weights is None:
        weights = {}
    weights.update({k: np.ones(len(v)) for k,v in ca_coords_dict.items() if k not in weights})
        
    d_min = ca_rg[ref_key].min() # (1, )
    d_max = ca_rg[ref_key].max()
    ca_rg_binned = {
        k: np.histogram(v, bins=n_bins, weights=weights[k], range=(d_min, d_max))[0]+PSEUDO_C 
            for k, v in ca_rg.items()
    }   # (N_bins, )
    # print("ca_rg_binned shape", {k: v.shape for k, v in ca_rg_binned.items()})
    results = {k: distance.jensenshannon(v, ca_rg_binned[ref_key], axis=0).mean() 
                for k, v in ca_rg_binned.items() if k != ref_key}
    
    results[ref_key] = 0.0
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results

def div_rmsd(ca_coords_dict):
    results = {}
    for k, v in ca_coords_dict.items():

        # print(k)  # [target, pred]
        # print(v.shape)  # (25,356,3)
        # only calculate Ca

        v = torch.as_tensor(v)
        # for idx in range(v.shape[0]):
        #     R, t = _find_rigid_alignment(v[idx], v[0])
        #     v[idx] = (torch.matmul(R, v[idx].transpose(-2, -1))).transpose(-2, -1) + t.unsqueeze(0)

        # v1 = v.numpy()
        # count = v1.shape[0]
        # rmsd_sum = np.sum(np.sqrt(np.sum((v1[:, None, :] - v1[None, :, :])**2, axis=-1)))

        count = 0
        rmsd_2_sum = 0
        for coord1 in v:
            for coord2 in v:
                count += 1
                rmsd_2_sum += squared_deviation(coord1,coord2,reduction='none')  # (356,)

        # with mp.Pool() as pool:
        #     res= pool.starmap(squared_deviation,[(coord1, coord2, 'none') for coord1 in v for coord2 in v])
        # pool.close()
        # pool.join()
        # count = len(res)-v.shape[0]
        # rmsd_2_sum = sum(res)

        results[k]=torch.sqrt(rmsd_2_sum/count)
        results[k]=np.around(float(torch.mean(results[k])), decimals=4)
    results['pred'] = (results['pred']-results['target'])/results['target']
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    # print(result)
    return results
        
def div_rmsf(ca_coords_dict):
    '''
        1D and 0D data
    '''
    results = {}
    for k, v in ca_coords_dict.items():

        v = torch.as_tensor(v)  # (250,356,3)
        # for idx in range(v.shape[0]):
        #     R, t = _find_rigid_alignment(v[idx], v[0])
        #     v[idx] = (torch.matmul(R, v[idx].transpose(-2, -1))).transpose(-2, -1) + t.unsqueeze(0)

        count = 0
        rmsd_2_sum = 0
        mean_str = torch.mean(v,dim = 0)  # (356,3)
        for coord1 in v:
            count += 1
            rmsd_2_sum += squared_deviation(coord1,mean_str,reduction='none')  # (356,)

        # count = v.shape[0]
        # rmsd_2_sum = torch.sum(torch.norm(v - mean_str[None,...], dim=-1) ** 2)

        # mean_str = torch.mean(v,dim = 0)  # (356,3)
        # with mp.Pool() as pool:
        #     res= pool.starmap(squared_deviation,[[(coord1, mean_str, 'none') for coord1 in v]])
        # pool.close()
        # pool.join()
        # count = len(res)
        # rmsd_2_sum = sum(res)

        results[k]=torch.sqrt(rmsd_2_sum/count)
        results[k]=np.around(float(torch.mean(results[k])), decimals=4)
        # print(result[k])
    results['pred'] = (results['pred']-results['target'])/results['target']
    results = {k: np.around(v, decimals=4) for k, v in results.items()}
    return results

def w2_rmwd(ca_coords_dict):
    result = {}
    means_total = {}
    covariances_total = {}
    count = 0
    v_ref  = torch.as_tensor(ca_coords_dict['target'][0])
    for k, v in ca_coords_dict.items():

        v = torch.as_tensor(v)
        # for idx in range(v.shape[0]):
        #     R, t = _find_rigid_alignment(v[idx], v_ref)
        #     v[idx] = (torch.matmul(R, v[idx].transpose(-2, -1))).transpose(-2, -1) + t.unsqueeze(0)

        means_total[k] = []
        covariances_total[k] = []

        for idx_residue in range(v.shape[1]):
            gmm = GaussianMixture(n_components=1)
            gmm.fit(v[:, idx_residue, :])
            means = torch.as_tensor(gmm.means_[0])  # 形状为 (3,)
            covariances = torch.as_tensor(gmm.covariances_[0])  # 形状为 (3, 3)

            means_total[k].append(means)
            covariances_total[k].append(covariances)
        means_total[k] = torch.stack(means_total[k], dim=0)  # (356, 3)
        covariances_total[k] = torch.stack(covariances_total[k], dim=0)  # (356, 3, 3)
        # print(means_total[k].shape, covariances_total[k].shape)
        # print(means_total[k][0], covariances_total[k][0])

    sigma_1_2_sqrt = [torch.as_tensor(fractional_matrix_power(i, 0.5)) for i in torch.matmul(covariances_total['target'], covariances_total['pred'])]
    sigma_1_2_sqrt = torch.stack(sigma_1_2_sqrt, dim=0)
    sigma_trace = covariances_total['target'] + covariances_total['pred'] - 2 * sigma_1_2_sqrt
    sigma_trace = [torch.trace(i) for i in sigma_trace]
    sigma_trace = torch.stack(sigma_trace, dim=0)

    result_1D = torch.sum((means_total['target'] - means_total['pred'])**2, dim=-1) + sigma_trace
    result['pred'] = np.around(float(torch.mean(result_1D)), decimals=4)
    # print(result['pred'])

    return result

def pro_w_contacts(ca_coords_dict, cry_ca_coords, dist_threshold = 8.0, percent_threshold = 0.1):
    result = {}
    w_contacts_total = {}

    dist = distance_matrix_ca(cry_ca_coords)
    L = dist.shape[-1]
    row, col = np.triu_indices(L, k=1)
    triu = dist[..., row, col]  # (n*(n-1)/2)
    w_contacts_crystall =  (triu < dist_threshold)

    for k, v in ca_coords_dict.items():

        dist = distance_matrix_ca(v)

        L = dist.shape[-1]
        row, col = np.triu_indices(L, k=1)
        triu = dist[..., row, col]  # (b, n*(n-1)/2)

        w_contacts =  (torch.tensor(triu) > dist_threshold).type(torch.float32)
        w_contacts = torch.mean(w_contacts, dim=0)  # (n*(n-1)/2,)
        w_contacts = w_contacts > percent_threshold

        w_contacts_total[k] = w_contacts & w_contacts_crystall
    
    jac_w_contacts = torch.sum(w_contacts_total['target'] & w_contacts_total['pred'])/torch.sum(w_contacts_total['target'] | w_contacts_total['pred'])
    result['pred'] = np.around(float(jac_w_contacts), decimals=4)
    # print(result['pred'])

    return result

def pro_t_contacts(ca_coords_dict, cry_ca_coords, dist_threshold = 8.0, percent_threshold = 0.1):
    result = {}
    w_contacts_total = {}

    dist = distance_matrix_ca(cry_ca_coords)
    L = dist.shape[-1]
    row, col = np.triu_indices(L, k=1)
    triu = dist[..., row, col]  # (n*(n-1)/2)
    w_contacts_crystall =  (triu >= dist_threshold)

    for k, v in ca_coords_dict.items():

        dist = distance_matrix_ca(v)

        L = dist.shape[-1]
        row, col = np.triu_indices(L, k=1)
        triu = dist[..., row, col]  # (b, n*(n-1)/2)

        w_contacts =  (torch.tensor(triu) <= dist_threshold).type(torch.float32)
        w_contacts = torch.mean(w_contacts, dim=0)  # (n*(n-1)/2,)
        w_contacts = w_contacts > percent_threshold

        w_contacts_total[k] = w_contacts & w_contacts_crystall

    jac_w_contacts = torch.sum(w_contacts_total['target'] & w_contacts_total['pred'])/torch.sum(w_contacts_total['target'] | w_contacts_total['pred'])
    result['pred'] = np.around(float(jac_w_contacts), decimals=4)
    # print(result['pred'])

    return result

def pro_c_contacts(target_file, pred_file, cry_target_file, area_threshold = 2.0, percent_threshold = 0.1):
    result = {}
    c_contacts_total = {}

    parser = PDBParser()
    params = freesasa.Parameters({'algorithm': 'ShrakeRupley', 'probe-radius': 2.8})
    
    structure_cry_target = parser.get_structure('cry_target', cry_target_file)
    str_params = {'separate-chains': False, 'separate-models': True}
    structure_target = freesasa.structureArray(target_file,str_params)
    structure_pred = freesasa.structureArray(pred_file,str_params)


    structure_cry_target = freesasa.structureFromBioPDB(structure_cry_target)
    sasa = freesasa.calc(structure_cry_target,params)
    residue_sasa = sasa.residueAreas()

    c_contacts_crystall = []
    # 打印每个残基的 SASA
    for chain_id in residue_sasa:
        for residue_id in residue_sasa[chain_id]:
            # print(f"Chain {chain_id}, Residue {residue_id}: {residue_sasa[chain_id][residue_id].residueType}, area: {residue_sasa[chain_id][residue_id].total}")
            c_contacts_crystall.append(residue_sasa[chain_id][residue_id].total < area_threshold)
    c_contacts_crystall = torch.tensor(c_contacts_crystall)

    c_contacts_target = 0
    count = 0
    for structure_temp in structure_target:
        count += 1
        sasa = freesasa.calc(structure_temp,params)
        residue_sasa = sasa.residueAreas()

        c_contacts_temp = []
        # 打印每个残基的 SASA
        for chain_id in residue_sasa:
            for residue_id in residue_sasa[chain_id]:
                # print(f"Chain {chain_id}, Residue {residue_id}: {residue_sasa[chain_id][residue_id].residueType}, area: {residue_sasa[chain_id][residue_id].total}")
                c_contacts_temp.append(residue_sasa[chain_id][residue_id].total > area_threshold)
        c_contacts_temp = torch.tensor(c_contacts_temp).type(torch.float32)
        c_contacts_target += c_contacts_temp
    c_contacts_target = c_contacts_target / count
    c_contacts_total['target'] = (c_contacts_target > percent_threshold) & c_contacts_crystall


    c_contacts_pred = 0
    count = 0
    for structure_temp in structure_pred:
        count += 1
        sasa = freesasa.calc(structure_temp,params)
        residue_sasa = sasa.residueAreas()

        c_contacts_temp = []
        # 打印每个残基的 SASA
        for chain_id in residue_sasa:
            for residue_id in residue_sasa[chain_id]:
                # print(f"Chain {chain_id}, Residue {residue_id}: {residue_sasa[chain_id][residue_id].residueType}, area: {residue_sasa[chain_id][residue_id].total}")
                c_contacts_temp.append(residue_sasa[chain_id][residue_id].total > area_threshold)
        c_contacts_temp = torch.tensor(c_contacts_temp).type(torch.float32)
        c_contacts_pred += c_contacts_temp
    c_contacts_pred = c_contacts_pred / count
    c_contacts_total['pred'] = (c_contacts_pred > percent_threshold) & c_contacts_crystall

    jac_w_contacts = torch.sum(c_contacts_total['target'] & c_contacts_total['pred'])/torch.sum(c_contacts_total['target'] | c_contacts_total['pred'])
    result['pred'] = np.around(float(jac_w_contacts), decimals=4)
    # print(jac_w_contacts)
    return result