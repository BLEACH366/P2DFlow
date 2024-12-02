import os
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import seaborn as sns
from Bio import SeqIO
import os
import random
import pandas as pd
import multiprocessing as mp

data_dir = "/cluster/home/shiqian/frame-flow-test1/ATLAS"
output_dir = os.path.join(data_dir,"select")
num_processes = 48
file_txt = os.path.join(data_dir,'ATLAS_filename.txt')

os.makedirs(output_dir,exist_ok=True)

def select_str(file_list, select_num = 11):
    info_total = {
            'rad_gyr': [],
            'rmsd_ref':[],
            'traj_filename':[],
            'energy':[],
        }
    count = 0
    for file in file_list:
        count += 1
        print(count, file)
        md_dir = os.path.join(data_dir, file)
        md_csv = pd.read_csv(os.path.join(md_dir, 'traj_info.csv'))
        md_csv = md_csv.sort_values('energy', ascending=True)

        select_filename = []
        temp = 0
        while(len(select_filename)<select_num - 1):
            temp += 1
            if temp > 100000:
                break
            flag = True
            idx = random.randint(0, len(md_csv)-1)
            for info in select_filename:
                if int(info['energy']*10) == int(md_csv.iloc[idx]['energy']*10):
                    flag = False
                    break
            if not flag:
                continue
            else:
                select_filename.append(md_csv.iloc[idx])
        print('select_num:',len(select_filename))
        select_filename.append(md_csv.iloc[0])
        for info in select_filename:
            file = info['traj_filename']
            os.system(f'cp -rf {os.path.join(md_dir,file)} {output_dir}')
        
        for info in select_filename:
            info_total['traj_filename'] += [info['traj_filename']]
            info_total['energy'] += [info['energy']]
            info_total['rad_gyr'] += [info['rad_gyr']]
            info_total['rmsd_ref'] += [info['rmsd_ref']]
    df = pd.DataFrame(info_total,index=[0]*len(info_total['energy']))
    df.to_csv(os.path.join(output_dir, 'traj_info.csv'),index=False)


with open(file_txt,'r+') as f:
    file_cont = f.read()
    file_list = file_cont.split("\n")

select_str(file_list)

# with mp.Pool(num_processes) as pool:
#     _ = pool.map(select_str,file_list)



