import argparse
import dataclasses
import functools as fn
import pandas as pd
import os
import tree
import torch
import multiprocessing as mp
import time
import esm
from Bio import PDB
import numpy as np
from data import utils as du
from data import parsers
from data import errors
from data.repr import get_pre_repr
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from data.cal_trans_rotmats import cal_trans_rotmats
from data.ESMfold_pred import ESMFold_Pred


def process_file(file_path: str, write_dir: str):
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    pdb_name = os.path.basename(file_path).replace('.pdb', '')
    metadata['pdb_name'] = pdb_name

    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    metadata['processed_path'] = os.path.abspath(processed_path)
    metadata['raw_path'] = file_path
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata['quaternary_category'] = 'homomer'
    else:
        metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_feats['modeled_idx'] = modeled_idx
    
    # try:
    #     # MDtraj
    #     traj = md.load(file_path)
    #     # SS calculation
    #     pdb_ss = md.compute_dssp(traj, simplified=True)
    #     # DG calculation
    #     pdb_dg = md.compute_rg(traj)
    #     # os.remove(file_path)
    # except Exception as e:
    #     # os.remove(file_path)
    #     raise errors.DataError(f'Mdtraj failed with error {e}')

    # chain_dict['ss'] = pdb_ss[0]
    # metadata['coil_percent'] = np.sum(pdb_ss == 'C') / metadata['modeled_seq_len']
    # metadata['helix_percent'] = np.sum(pdb_ss == 'H') / metadata['modeled_seq_len']
    # metadata['strand_percent'] = np.sum(pdb_ss == 'E') / metadata['modeled_seq_len']

    # # Radius of gyration
    # metadata['radius_gyration'] = pdb_dg[0]
    
    # Write features to pickles.
    du.write_pkl(processed_path, complex_feats)

    return metadata


def process_serially(all_paths, write_dir):
    all_metadata = []
    for i, file_path in enumerate(all_paths):
        try:
            start_time = time.time()
            metadata = process_file(
                file_path,
                write_dir)
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {file_path}: {e}')
    return all_metadata


def process_fn(
        file_path,
        verbose=None,
        write_dir=None):
    try:
        start_time = time.time()
        metadata = process_file(
            file_path,
            write_dir)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')


def main(args):
    pdb_dir = args.pdb_dir
    all_file_paths = [
        os.path.join(pdb_dir, x)
        for x in os.listdir(args.pdb_dir) if '.pdb' in x]
    total_num_paths = len(all_file_paths)
    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_file_paths,
            write_dir)
    else:
        _process_fn = fn.partial(  #  fix some args of fn
            process_fn,
            verbose=args.verbose,
            write_dir=write_dir)
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(_process_fn, all_file_paths)  # all jobs to be done are saved in a iterable object (such as list)
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')


def cal_repr(processed_file_path, model_esm2, alphabet, batch_converter, esm_device):
    print(f'cal_repr for {processed_file_path}')
    processed_feats_org = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats_org)

    # Only take modeled residues.
    modeled_idx = processed_feats['modeled_idx']
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    # del processed_feats['modeled_idx']
    processed_feats = tree.map_structure(
        lambda x: x[min_idx:(max_idx+1)], processed_feats)

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    # res_idx = processed_feats['residue_index']

    node_repr_pre, pair_repr_pre = get_pre_repr(chain_feats['aatype'], model_esm2, alphabet, batch_converter, device = esm_device)  # (B,L,d_node_pre=1280), (B,L,L,d_edge_pre=20)
    node_repr_pre = node_repr_pre[0].cpu()
    pair_repr_pre = pair_repr_pre[0].cpu()

    out = {
            'aatype': chain_feats['aatype'],
            'rotmats_1': rotmats_1,
            'trans_1': trans_1,  # (L,3)
            'res_mask': torch.tensor(processed_feats['bb_mask']).int(),
            'bb_positions': processed_feats['bb_positions'],
            'all_atom_positions':chain_feats['all_atom_positions'],
            'node_repr_pre':node_repr_pre,
            'pair_repr_pre':pair_repr_pre,
        }

    du.write_pkl(processed_file_path, out)

def cal_static_structure(processed_file_path, raw_pdb_file, ESMFold):
    output_total = du.read_pkl(processed_file_path)

    save_dir = os.path.join(os.path.dirname(raw_pdb_file), 'ESMFold_Pred_results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(processed_file_path)[:6]+'_esmfold.pdb')
    if not os.path.exists(save_path):
        print(f'cal_static_structure for {processed_file_path}')
        ESMFold.predict_str(raw_pdb_file, save_path)
    trans, rotmats = cal_trans_rotmats(save_path)
    output_total['trans_esmfold'] = trans
    output_total['rotmats_esmfold'] = rotmats

    du.write_pkl(processed_file_path, output_total)


def merge_pdb(metadata_path, traj_info_file, valid_seq_file, merged_output_file):
    df1 = pd.read_csv(metadata_path)
    df2 = pd.read_csv(traj_info_file)
    df3 = pd.read_csv(valid_seq_file)

    # 获取文件名
    df1['traj_filename'] = [os.path.basename(i) for i in df1['raw_path']]

    # 合并数据
    merged = df1.merge(df2[['traj_filename', 'energy']], on='traj_filename', how='left')
    merged['is_trainset'] = ~merged['traj_filename'].str[:6].isin(df3['file'])

    # 保存结果
    merged.to_csv(merged_output_file, index=False)
    print('merge complete!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdb_dir", type=str, default="./dataset/ATLAS/select")
    parser.add_argument("--write_dir", type=str, default="./dataset/ATLAS/select/pkl")
    parser.add_argument("--csv_name", type=str, default="metadata.csv")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--num_processes", type=int, default=48)
    parser.add_argument('--verbose', help='Whether to log everything.',action='store_true')

    parser.add_argument("--esm_device", type=str, default='cuda')

    parser.add_argument("--traj_info_file", type=str, default='./dataset/ATLAS/select/traj_info_select.csv')
    parser.add_argument("--valid_seq_file", type=str, default='./inference/valid_seq.csv')
    parser.add_argument("--merged_output_file", type=str, default='./dataset/ATLAS/select/pkl/metadata_merged.csv')

    args = parser.parse_args()

    # process .pdb to .pkl
    main(args)

    # cal_repr
    csv_path = os.path.join(args.write_dir, args.csv_name)
    pdb_csv = pd.read_csv(csv_path)
    pdb_csv = pdb_csv.sort_values('modeled_seq_len', ascending=False)
    model_esm2, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model_esm2.eval()
    model_esm2.requires_grad_(False)
    model_esm2.to(args.esm_device)
    for idx in range(len(pdb_csv)):
        cal_repr(pdb_csv.iloc[idx]['processed_path'], model_esm2, alphabet, batch_converter, args.esm_device)

    # cal_static_structure
    csv_path = os.path.join(args.write_dir, args.csv_name)
    pdb_csv = pd.read_csv(csv_path)
    ESMFold = ESMFold_Pred(device = args.esm_device)
    for idx in range(len(pdb_csv)):
        cal_static_structure(pdb_csv.iloc[idx]['processed_path'], pdb_csv.iloc[idx]['raw_path'], ESMFold)

    # merge csv
    csv_path = os.path.join(args.write_dir, args.csv_name)
    merge_pdb(csv_path, args.traj_info_file, args.valid_seq_file, args.merged_output_file)


    
    

