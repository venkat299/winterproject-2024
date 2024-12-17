import os 
import argparse
import pickle
from scipy.spatial.distance import pdist, squareform
import numpy as np
import asyncio

import sys
sys.path.append(".")
from src.sol_1_mip_bcp.bcp import build_and_solve
from src.common.helpers import load_pkl_file, save_dataset, get_output_path, background, append_to_pickle



# def get_output_path(opt):
#     dataset_basename, ext = os.path.splitext(os.path.split(opt.dataset)[-1])
#     solver_name = "_".join(os.path.normpath(os.path.splitext(opts.solver)[0]).split(os.sep)[-2:])
#     if opts.o is None:
#         results_dir = os.path.join(opts.results_dir, solver_name)
#         os.makedirs(results_dir, exist_ok=True)

#         out_file = os.path.join(results_dir, "{}-{}{}".format(
#             dataset_basename, solver_name, ext
#         ))
#     else:
#         out_file = opts.o

#     assert opts.f or not os.path.isfile(
#         out_file), "File already exists! Try running with -f option to overwrite."
#     return out_file

@background
def job(idx, instance, filename):
    total_cost, routes, execution_time = build_and_solve(instance)
    print(f'instance id:{idx}, cost:{total_cost}, time:{execution_time}')
    res =[None, None , None, idx]
    if total_cost:
        res = [float(total_cost/10000), routes, execution_time, idx ]
    append_to_pickle(filename, res)
    return res

def eval_dataset(opt):
    # print(opt)
    out_file = get_output_path(opt)
    dt_set = load_pkl_file(opt.dataset)
    # result = []
    # for instance in dt_set:
    #     total_cost, routes, execution_time = build_and_solve(instance)
    #     result.append([total_cost/10000, routes, execution_time ])
    
    loop = asyncio.get_event_loop() # Have a new event loop
    looper = asyncio.gather(*[job(idx, instance, out_file) for idx, instance in enumerate(dt_set)])# Run the loop                        
    result = loop.run_until_complete(looper)     
    save_dataset(result, out_file)

    # print(f"Execution time: {execution_time} seconds")
    # print("Optimal routes:", routes)
    # print("Total cost:", total_cost/10000)

if __name__ == '__main__':
    print("running MIP - BCP solver")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",type=str, help="Filename of the dataset to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--solver', type=str, default='mip_bcb')
    
    opts = parser.parse_args()
    
    eval_dataset(opts)