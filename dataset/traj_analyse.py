import os
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import multiprocessing as mp


dirpath = "/cluster/home/shiqian/frame-flow-test1/ATLAS/"
k=2.32*1e-4  # unit(eV/K)
T=298.15  # unit(K)
file_txt = os.path.join(dirpath,'ATLAS_filename.txt')
num_processes = 48

with open(file_txt,'r+') as f:
    file_cont = f.read()
    file_list = file_cont.split("\n")


out_total = {
    'traj_filename':[],
    'rad_gyr': [],
    'rmsd_ref':[],
    'energy':[],
}

def fn(file_md, use_csv = False):
    mdpath = os.path.join(dirpath,file_md)
    filename = file_md

    if not use_csv:
        pdb_filepath = os.path.join(mdpath,filename+".pdb")
        topology_filepath = os.path.join(mdpath,filename+".pdb")

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
    else:
        info = pd.read_csv(os.path.join(mdpath, "traj_info.csv"))
        
    info_array = np.stack([info['rad_gyr'],info['rmsd_ref']],axis=0)  # (2,2500)
    kde = gaussian_kde(info_array)
    density = kde(info_array)  # (2500,)
    G = k*T*np.log(np.max(density)/density)  # (2500,)
    G = (G-np.min(G))/(np.max(G)-np.min(G))
    
    if use_csv:
        info['energy'] = G.tolist()
    else:
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

with mp.Pool(num_processes) as pool:
    _ = pool.map(fn,file_list)

# for file_md in ['16pk_A','7wab_A','7s86_A']:
#     fn(file_md)





