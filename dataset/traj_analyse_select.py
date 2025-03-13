import os
import argparse
import shutil

import pandas as pd
import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

from scipy.stats import gaussian_kde
from MDAnalysis.analysis import align



def cal_energy(para1):
    file_md, dirpath = para1
    mdpath = os.path.join(dirpath, file_md)
    filename = file_md

    k=2.32*1e-4  # unit(eV/K)
    T=298.15  # unit(K)

    pdb_filepath = os.path.join(mdpath, filename+".pdb")
    topology_filepath = os.path.join(mdpath, filename+".pdb")

    u_ref = mda.Universe(pdb_filepath)
    protein_ref = u_ref.select_atoms('protein')
    bb_atom_ref = protein_ref.select_atoms('name CA or name C or name N')

    info = {
            'rad_gyr': [],
            'rmsd_ref':[],
            'traj_filename':[],
            'energy':[],
        }

    for xtc_idx in range(1,4):
        trajectory_filepath = os.path.join(mdpath,filename+"_R"+str(xtc_idx)+".xtc") 

        u = mda.Universe(topology_filepath, trajectory_filepath)
        
        protein = u.select_atoms('protein')
        bb_atom = protein.select_atoms('name CA or name C or name N')

        # CA_atoms = u.select_atoms('name CA')
        # bb_atoms = u.select_atoms('backbone')

        count = 0
        # for ts in u.trajectory:
        for _ in u.trajectory:
            count += 1

            rad_gyr = bb_atom.radius_of_gyration()
            rmsd_ref = align.alignto(bb_atom, bb_atom_ref, select='all', match_atoms=False)[-1]
            info['rad_gyr'].append(rad_gyr)
            info['rmsd_ref'].append(rmsd_ref)

            traj_filename = filename + '_R' + str(xtc_idx) + '_'+str(count)+".pdb"
            info['traj_filename'].append(traj_filename)
            print(traj_filename)
            protein.write(os.path.join(mdpath, traj_filename))

        
    info_array = np.stack([info['rad_gyr'],info['rmsd_ref']],axis=0)  # (2,2500)
    kde = gaussian_kde(info_array)
    density = kde(info_array)  # (2500,)
    G = k*T*np.log(np.max(density)/density)  # (2500,)
    G = (G-np.min(G))/(np.max(G)-np.min(G))
    
    info['energy'] += G.tolist()

    out_total = pd.DataFrame(info)
    x, y = np.meshgrid(np.linspace(min(out_total['rad_gyr'])-0.25, max(out_total['rad_gyr'])+0.25, 200),
                       np.linspace(min(out_total['rmsd_ref'])-0.25, max(out_total['rmsd_ref'])+0.25, 200))
    grid_coordinates = np.vstack([x.ravel(), y.ravel()])
    density_values = kde(grid_coordinates)
    # 将密度值变形为与网格坐标相同的形状
    density_map = density_values.reshape(x.shape)
    # 绘制高斯核密度估计图
    plt.contourf(x, y, density_map, levels= np.arange(np.max(density_map)/20, np.max(density_map)*1.1, np.max(density_map)/10))
    plt.colorbar()

    plt.savefig(os.path.join(mdpath,"md.png"))
    plt.close()

    out_total.to_csv(os.path.join(mdpath,"traj_info.csv"),index=False)


def select_str(file, data_dir, output_dir, select_num=100):
    info_total = {
        'rad_gyr': [],
        'rmsd_ref': [],
        'traj_filename': [],
        'energy': [],
    }

    print(f"Processing {file}")
    md_dir = os.path.join(data_dir, file)
    md_csv = pd.read_csv(os.path.join(md_dir, 'traj_info.csv'))
    md_csv = md_csv.sort_values('energy', ascending=True)

    idx_total = np.linspace(0, len(md_csv) - 1, select_num)
    idx_total = (idx_total / idx_total[-1]) ** (1 / 3) * (len(md_csv) - 1)  # mapping energy with f(x)=x**(1/3) to get more relatively high-energy structure
    idx_total = np.unique(np.round(idx_total).astype(int))

    for idx in idx_total:
        info = md_csv.iloc[idx]
        traj_filename = info['traj_filename']
        shutil.copy(os.path.join(md_dir, traj_filename), output_dir)

        info_total['traj_filename'].append(traj_filename)
        info_total['energy'].append(info['energy'])
        info_total['rad_gyr'].append(info['rad_gyr'])
        info_total['rmsd_ref'].append(info['rmsd_ref'])

    return info_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dir_path", type=str, default="./dataset/ATLAS")
    parser.add_argument("--filename", type=str, default="ATLAS_filename.txt")

    parser.add_argument("--select_num", type=int, default=100)
    parser.add_argument("--select_dir", type=str, default="./dataset/ATLAS/select")

    args = parser.parse_args()

    num_processes = 48

    file_txt = os.path.join(args.dir_path, args.filename)
    os.makedirs(args.select_dir, exist_ok=True)

    with open(file_txt,'r+') as f:
        file_cont = f.read()
        file_list = file_cont.split("\n")

    para1_list = [(file, args.dir_path) for file in file_list]
    para2_list = [(file, args.dir_path, args.select_dir, args.select_num) for file in file_list]

    info_total_all = {
        'rad_gyr': [],
        'rmsd_ref': [],
        'traj_filename': [],
        'energy': [],
    }

    with mp.Pool(num_processes) as pool:
        _ = pool.map(cal_energy, para1_list)
        results = pool.starmap(select_str, para2_list)
    
    for result in results:
        info_total_all['traj_filename'].extend(result['traj_filename'])
        info_total_all['energy'].extend(result['energy'])
        info_total_all['rad_gyr'].extend(result['rad_gyr'])
        info_total_all['rmsd_ref'].extend(result['rmsd_ref'])

    df = pd.DataFrame(info_total_all)
    df.to_csv(os.path.join(args.select_dir, 'traj_info_select.csv'), index=False)