import os 
import argparse
import pickle
from scipy.spatial.distance import pdist, squareform
import numpy as np
import asyncio


import sys
sys.path.append(".")
from src.sol_4_or_tools.model import solve_cvrp_ortools
from src.common.helpers import load_pkl_file, background, save_dataset, get_output_path, append_to_pickle

def create_data_model(input):
    data = {}
    demands = input[2]
    demands.insert(0,0)  # Depot has no demand
    data["demands"] = demands
    num_vehicles = round(len(input[2])/2)
    data["num_vehicles"] = num_vehicles
    data["vehicle_capacities"] = [int(input[3])]*num_vehicles
    data["depot"] = 0
    

    # Generate random coordinates for customers
    coordinates = input[1]
    coordinates.insert(0,input[0])
    # Calculate distances between customers
    distances = squareform(pdist(coordinates, metric="euclidean"))
    distances = np.int64(np.round(distances, decimals=4)*10000)
    data["distance_matrix"] = distances
    return data



@background
def job(idx, instance, filename):
    dt = create_data_model(instance)
    total_cost, routes, execution_time = solve_cvrp_ortools(dt)
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
    # dt = create_data_model(dt_set[0])
    # total_cost, routes, execution_time = solve_cvrp_ortools(dt)
    loop = asyncio.get_event_loop()                                              # Have a new event loop
    looper = asyncio.gather(*[job(idx, instance, out_file) for idx, instance in enumerate(dt_set)])         # Run the loop                        
    result = loop.run_until_complete(looper)     
    save_dataset(result, out_file)
    

    # print(f"Execution time: {execution_time} seconds")
    # print("Optimal routes:", routes)
    # print("Total cost:", total_cost/10000)

if __name__ == '__main__':
    print("running ortools solver")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",type=str, help="Filename of the dataset to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--solver', type=str, default='mh-ortools')
    
    opts = parser.parse_args()
    
    eval_dataset(opts)