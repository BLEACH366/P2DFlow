import os
import re
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align, rms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def cal_PCA(md_pdb_path,ref_path,pred_pdb_path,n_components = 2):
    print("")
    print('filename=',os.path.basename(ref_path))

    u = mda.Universe(md_pdb_path, md_pdb_path)
    u_ref = mda.Universe(ref_path, ref_path)

    aligner = align.AlignTraj(u,
                              u_ref, 
                              select='name CA or name C or name N',
                              in_memory=True).run()

    pc = pca.PCA(u, 
                select='name CA or name C or name N',
                align=False, mean=None,
                # n_components=None,
                n_components=n_components,
                ).run()

    backbone = u.select_atoms('name CA or name C or name N')
    n_bb = len(backbone)
    print('There are {} backbone atoms in the analysis'.format(n_bb))

    for i in range(n_components):
        print(f"Cumulated variance {i+1}: {pc.cumulated_variance[i]:.3f}")

    transformed = pc.transform(backbone, n_components=n_components)

    print(transformed.shape)  # (3000, 2)

    df = pd.DataFrame(transformed,
                    columns=['PC{}'.format(i+1) for i in range(n_components)])


    plt.scatter(df['PC1'],df['PC2'],marker='o')
    plt.show()

    output_dir = os.path.dirname(ref_path)
    output_filename = os.path.basename(ref_path).split('.')[0]

    # df.to_csv(os.path.join(output_dir, f'{output_filename}_md_pca.csv'))
    # plt.savefig(os.path.join(output_dir, f'{output_filename}_md_pca.png'))


    for k,v in pred_pdb_path.items():
        u_pred = mda.Universe(v, v)
        aligner = align.AlignTraj(u_pred,
                            u_ref, 
                            select='name CA or name C or name N',
                            in_memory=True).run()
        pred_backbone = u_pred.select_atoms('name CA or name C or name N')
        pred_transformed = pc.transform(pred_backbone, n_components=n_components)




        target = np.array([4,0])
        min_value = 1e5
        min_idx = -1
        for idx,value in enumerate(pred_transformed):
            if np.linalg.norm(value-target) < min_value:
                min_value = np.linalg.norm(value-target)
                min_idx = idx
        print(min_value)
        print(min_idx)
        raise ValueError




        df = pd.DataFrame(pred_transformed,
                        columns=['PC{}'.format(i+1) for i in range(n_components)])

        plt.scatter(df['PC1'],df['PC2'],marker='o')
        plt.show()

        # output_dir = os.path.dirname(ref_path)
        # output_filename = os.path.basename(ref_path).split('.')[0]
        df.to_csv(os.path.join(output_dir, f'{output_filename}_{k}_pca.csv'))
        plt.savefig(os.path.join(output_dir, f'{output_filename}_{k}_pca.png'))
    plt.clf()


if __name__ == '__main__':
    pred_pdb_path_org={
        'esm_n':'/cluster/home/shiqian/frame-flow-test1/valid/evaluate/esm_n_pred',
        # 'alphaflow':'/cluster/home/shiqian/frame-flow-test1/valid/evaluate/alphaflow_pred',
        # 'Str2Str':'/cluster/home/shiqian/frame-flow-test1/valid/evaluate/Str2Str_pred',
        # 'test1':'/cluster/home/shiqian/frame-flow-test1/valid/evaluate/test_05_1',
    }
    md_pdb_path_org='/cluster/home/shiqian/frame-flow-test1/valid/evaluate/ATLAS_valid'
    ref_path_org='/cluster/home/shiqian/frame-flow-test1/valid/evaluate/crystal'

    # file_list = ['2wsi_A.pdb','2z4u_A.pdb','3pes_A.pdb']

    for file in os.listdir(md_pdb_path_org):
    # for file in file_list:
        if re.search('\.pdb',file):
            pred_pdb_path={
                'esm_n':'',
                # 'alphaflow':'',
                # 'Str2Str':'',
                # 'test1':'',
            }
            for k,v in pred_pdb_path.items():
                pred_pdb_path[k]=os.path.join(pred_pdb_path_org[k],file)
            md_pdb_path = os.path.join(md_pdb_path_org, file)
            ref_path = os.path.join(ref_path_org, file)
            cal_PCA(md_pdb_path,ref_path,pred_pdb_path)







