import shutil
import os
import argparse

def setup_parser():
    parser = argparse.ArgumentParser(
        description='Pack the dependencies of Tailor')
    parser.add_argument('--binary_dir', type=os.path.abspath)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = setup_parser()
    binary_dir = args.binary_dir

    cmd = f"ldd {binary_dir}/Tailor"
    out_string = os.popen(cmd)  # capture the output from the terminal
    data = out_string.readlines()
    data = [[data_.split()[0], data_.split()[2]] for data_ in data if len(data_.split())==4 ]   # parse the libraries
    print(data)

    for data_ in data:
        shutil.copy(data_[1], os.path.join(binary_dir, data_[0]))