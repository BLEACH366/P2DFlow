import os
import multiprocessing as mp

data_dir = "./dataset"
file_txt = os.path.join(data_dir,'ATLAS_filename.txt')
url_base = "https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/"
num_processes = 48

os.makedirs(data_dir, exist_ok=True)

with open(file_txt,'r+') as f:
    file_cont = f.read()
    file_list = file_cont.split("\n")

def fn(file):
    url = url_base + file
    output_filename = file+".zip"
    output_path = os.path.join(data_dir, output_filename)
    unzip_path = os.path.join(data_dir, file)
    os.system(f"curl -X GET {url} -H accept: */* --output {output_path}")
    os.system(f"unzip {output_path} -d {unzip_path}")

with mp.Pool(num_processes) as pool:
    _ = pool.map(fn,file_list)
    print("finished")



