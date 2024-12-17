import sys
sys.path.append(".")
import time
from itertools import cycle
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyomo.opt import SolverFactory
import os
import pickle
from scipy.spatial.distance import pdist, squareform


from constraints import (
    arcs_in,
    arcs_out,
    capacity_constraint,
    comp_vehicle_assignment,
    subtour_elimination,
    vehicle_assignment,
)

from src.sol_1_mip.model import CVRPModel
# from src.common.helpers import transform_data, visualize_tours


def build_model_and_run(data):
    num_nodes = len(data[1])+1
    demands = data[2]
    demands.insert(0,0)  # Depot has no demand

    vehicle_capacity = data[3]
    num_vehicles = 4 #round(len(data[2])/2)

    # Generate random coordinates for customers
    coordinates = data[1]
    coordinates.insert(0,data[0])
    # Calculate distances between customers
    distances = squareform(pdist(coordinates, metric="euclidean"))
    distances = np.int64(np.round(distances, decimals=4)*10000)
    
    print('distances', distances)
    print('num of nodes', num_nodes)
    print('num of vehicles', num_vehicles)
    print('capacity', vehicle_capacity)
    print('demands', demands)
    # return 

    # Instantiate the CVRPModel with data
    cvrp_model = CVRPModel(
        num_nodes, num_vehicles, distances, demands, vehicle_capacity
    )
    model = cvrp_model.get_model()
    env = cvrp_model.get_env()

    # Add constraints from constraints.py to the model
    model.arcs_in = env.Constraint(model.nodes, rule=arcs_in)
    model.arcs_out = env.Constraint(model.nodes, rule=arcs_out)
    model.vehicle_assignment = env.Constraint(
        model.nodes, model.vehicles, rule=vehicle_assignment
    )
    model.comp_vehicle_assignment = env.Constraint(
        model.nodes, model.vehicles, rule=comp_vehicle_assignment
    )
    model.capacity_constraint = env.Constraint(
        model.vehicles, rule=capacity_constraint)
    model.subtour_elimination = env.ConstraintList()

    # Define the objective function
    model.obj = env.Objective(
        expr=sum(
                model.x[i, j, k] * model.distances[i, j]
                for (i, j) in model.edges
                for k in model.vehicles
        ),
        sense=env.minimize,
    )

    # Initialize the solver
    solver = SolverFactory("scip")
    
    sol = solve(model, solver)
    
    tours = cvrp_model.find_tours()
    print(demands)
    print(sol)
    print(tours)

def find_arcs(model):
    arcs = []
    for i, j in model.edges:
        for k in model.vehicles:
            if np.isclose(model.x[i, j, k].value, 1, atol=1e-1):
                arcs.append((i, j))
    return arcs

def find_subtours(arcs):
    G = nx.DiGraph(arcs)
    subtours = list(nx.strongly_connected_components(G))
    return subtours

def eliminate_subtours(model, subtours):
    proceed = False
    for S in subtours:
        if 0 not in S:
            proceed = True
            Sout = {i for i in model.nodes if i not in S}
            for h in S:
                for k in model.vehicles:
                    model.subtour_elimination.add(
                        subtour_elimination(model, S, Sout, h, k)
                    )
    return proceed

def _solve_step(model, solver, verbose=True):
    sol = solver.solve(model)
    arcs = find_arcs(model)
    subtours = find_subtours(arcs)
    if verbose:
        print(f"Current subtours: {subtours}")
    time.sleep(0.1)
    proceed = eliminate_subtours(model, subtours)
    return sol, proceed

def solve(model, solver, verbose=True):
    proceed = True
    while proceed:
        sol, proceed = _solve_step(model, solver, verbose=verbose)
    return sol

def find_tours(model):
    tours = []
    for k in model.vehicles:
        node = 0
        tours.append([0])
        while True:
            for j in model.nodes:
                if (node, j) in model.edges:
                    if np.isclose(model.x[node, j, k].value, 1):
                        node = j
                        tours[-1].append(node)
                        break
            if node == 0:
                break
    return tours



# Visualize the solution
# visualize_tours(tours, coordinates)

def get_output_path(opt):
    dataset_basename, ext = os.path.splitext(os.path.split(opt.dataset)[-1])
    solver_name = "_".join(os.path.normpath(os.path.splitext(opts.solver)[0]).split(os.sep)[-2:])
    if opt.o is None:
        results_dir = os.path.join(opt.results_dir, 'mip_'+solver_name)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}{}".format(
            dataset_basename, solver_name, ext
        ))
    else:
        out_file = opt.o

    assert opt.f or not os.path.isfile(
        out_file), "File already exists! Try running with -f option to overwrite."
    return out_file

def load_pkl_file(file_path):
  with open(file_path, 'rb') as file:
    return pickle.load(file)


def eval_dataset(opt):
    print(opt)
    out_file = get_output_path(opt)
    print(out_file)
    print(opt.dataset)
    dt = load_pkl_file(opt.dataset)
    print(len(dt[0]))
    print(len(dt[0][1]))
    
    build_model_and_run(dt[0])
    

    # save_dataset((results, parallelism), out_file)

if __name__ == '__main__':
    print("running ORTools solver")
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset",type=str, help="Filename of the dataset to evaluate")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('--solver', type=str, default='scip')
    
    opts = parser.parse_args()
    
    eval_dataset(opts)