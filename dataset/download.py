import os
import argparse
import multiprocessing as mp


def fn(para):
    file, url_base, data_dir = para
    url = url_base + file
    output_filename = file+".zip"
    output_path = os.path.join(data_dir, output_filename)
    unzip_path = os.path.join(data_dir, file)
    os.system(f"curl -X GET {url} -H accept: */* --output {output_path}")
    os.system(f"unzip {output_path} -d {unzip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename_file", type=str, default="./dataset/ATLAS_filename.txt")
    parser.add_argument("--url_base", type=str, default="https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/")
    parser.add_argument("--output_dir", type=str, default="./dataset/ATLAS_test")

    args = parser.parse_args()

    num_processes = 48

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.filename_file,'r+') as f:
        file_cont = f.read()
        file_list = file_cont.split("\n")
    para_list = [(file, args.url_base, args.output_dir) for file in file_list]

    with mp.Pool(num_processes) as pool:
        _ = pool.map(fn, para_list)
        print("finished")
    
    os.system(f"cp -rf {args.filename_file} {args.output_dir}")
