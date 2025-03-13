import os
import re
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import pca, align, rms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
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

    output_dir = os.path.dirname(md_pdb_path)
    output_filename = os.path.basename(md_pdb_path).split('.')[0]

    df.to_csv(os.path.join(output_dir, f'{output_filename}_md_pca.csv'))
    plt.savefig(os.path.join(output_dir, f'{output_filename}_md_pca.png'))


    for k,v in pred_pdb_path.items():
        u_pred = mda.Universe(v, v)
        aligner = align.AlignTraj(u_pred,
                            u_ref, 
                            select='name CA or name C or name N',
                            in_memory=True).run()
        pred_backbone = u_pred.select_atoms('name CA or name C or name N')
        pred_transformed = pc.transform(pred_backbone, n_components=n_components)

        df = pd.DataFrame(pred_transformed,
                        columns=['PC{}'.format(i+1) for i in range(n_components)])

        plt.scatter(df['PC1'],df['PC2'],marker='o')
        plt.show()

        output_dir = os.path.dirname(v)
        output_filename = os.path.basename(v).split('.')[0]
        df.to_csv(os.path.join(output_dir, f'{output_filename}_{k}_pca.csv'))
        plt.savefig(os.path.join(output_dir, f'{output_filename}_{k}_pca.png'))
    plt.clf()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_pdb_dir", type=str, default="./inference/test/pred_merge_results")
    parser.add_argument("--target_dir", type=str, default="./inference/test/target_dir")
    parser.add_argument("--crystal_dir", type=str, default="./inference/test/crystal_dir")

    args = parser.parse_args()


    pred_pdb_path_org={
        'P2DFlow':args.pred_pdb_dir,
    }
    md_pdb_path_org = args.target_dir
    ref_path_org = args.crystal_dir


    for file in os.listdir(md_pdb_path_org):
        if re.search('\.pdb',file):
            pred_pdb_path={
                'P2DFlow':'',
                # 'alphaflow':'',
                # 'Str2Str':'',
            }
            for k,v in pred_pdb_path.items():
                pred_pdb_path[k]=os.path.join(pred_pdb_path_org[k],file)
            md_pdb_path = os.path.join(md_pdb_path_org, file)
            ref_path = os.path.join(ref_path_org, file)
            cal_PCA(md_pdb_path,ref_path,pred_pdb_path)







