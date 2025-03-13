import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.dihedrals import Ramachandran
import os
import re
import pandas as pd
import matplotlib.pyplot as plt


def ramachandran_eval(all_paths, pdb_file, output_dir):
    angle_results_all = []

    for dirpath in all_paths:
        pdb_path = os.path.join(dirpath,pdb_file)

        u = mda.Universe(pdb_path)
        protein = u.select_atoms('protein')
        # print('There are {} residues in the protein'.format(len(protein.residues)))

        ramachandran = Ramachandran(protein) 
        ramachandran.run()  
        angle_results = ramachandran.results.angles
        # print(angle_results.shape)

        # ramachandran.plot(color='black', marker='.')

        angle_results_all.append(angle_results.reshape([-1,2]))


        # df = pd.DataFrame(angle_results.reshape([-1,2]))
        # df.to_csv(os.path.join(output_dir, os.path.basename(dirpath)+'_'+pdb_file.split('.')[0]+'.csv'), index=False)


    points1 = angle_results_all[0]
    grid_size = 360  # 网格的大小
    x_bins = np.linspace(-180, 180, grid_size)
    y_bins = np.linspace(-180, 180, grid_size)
    result_tmp={}
    for idx in range(len(angle_results_all[1:])):
        idx = idx + 1
        points2 = angle_results_all[idx]

        # 使用2D直方图统计每组点在网格上的分布
        hist1, _, _ = np.histogram2d(points1[:, 0], points1[:, 1], bins=[x_bins, y_bins])
        hist2, _, _ = np.histogram2d(points2[:, 0], points2[:, 1], bins=[x_bins, y_bins])

        # 将直方图转换为布尔值，表示某个网格是否有点落入
        mask1 = hist1 > 0
        mask2 = hist2 > 0

        intersection = np.logical_and(mask1, mask2).sum()
        all_mask2 = mask2.sum()
        val_ratio = intersection / all_mask2
        print(os.path.basename(all_paths[idx]), "val_ratio:", val_ratio)


        result_tmp[os.path.basename(all_paths[idx])] = val_ratio
    result_tmp['file'] = pdb_file

    return result_tmp



if __name__ == "__main__":
    key1 = 'P2DFlow_epoch19'
    all_paths = [
                "/cluster/home/shiqian/frame-flow-test1/valid/evaluate/ATLAS_valid",
                # "/cluster/home/shiqian/frame-flow-test1/valid/evaluate/esm_n_pred",
                "/cluster/home/shiqian/frame-flow-test1/valid/evaluate/alphaflow_pred",
                "/cluster/home/shiqian/frame-flow-test1/valid/evaluate/Str2Str_pred",

                f'/cluster/home/shiqian/frame-flow-test1/valid/evaluate/{key1}',
                
                ]
    output_dir = '/cluster/home/shiqian/frame-flow-test1/valid/evaluate/Ramachandran'
    os.makedirs(output_dir, exist_ok=True)
    results={
            'file':[],
            # 'esm_n_pred':[],
            'alphaflow_pred':[],
            'Str2Str_pred':[],

            key1:[],
            }
    for file in os.listdir(all_paths[0]):
        if re.search('\.pdb',file):

            pdb_file = file
            print(file)
            result_tmp = ramachandran_eval(
                all_paths=all_paths,
                pdb_file=pdb_file,
                output_dir=output_dir
            )
            for key in results.keys():
                results[key].append(result_tmp[key])

    out_total_df = pd.DataFrame(results)
    out_total_df.to_csv(os.path.join(output_dir,f'Ramachandran_plot_validity_{key1}.csv'),index=False)
