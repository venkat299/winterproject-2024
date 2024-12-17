import VRPSolverEasy as vrpse
import math
from scipy.spatial.distance import pdist, squareform
import numpy as np


def build_and_solve(input):
    # Initialisation
    model = vrpse.Model()
    model.set_parameters(time_limit=500.0,  print_level=-2)
    #  class src.solver.Parameters(time_limit=300.0, upper_bound=1000000, heuristic_used=False, time_limit_heuristic=20.0, config_file='', solver_name='CLP', print_level=-1, action='solve', cplex_path='')

    # Add vehicle type
    model.add_vehicle_type(
        id=1,
        start_point_id=0,
        end_point_id=0,
        name="VEH1",
        capacity=int(input[3]),
        max_number=round(len(input[2])),
        var_cost_dist=1)

    # Add depot
    model.add_depot(id=0)
    
    # Add customers
    for i in range(1, len(input[2])+1):
        model.add_customer(
            id=i,
            demand= input[2][i-1])
        
    coordinates = input[1]
    coordinates.insert(0,input[0])
    
    # Calculate distances between customers
    distances = squareform(pdist(coordinates, metric="euclidean"))
    distances = np.int64(np.round(distances, decimals=4)*10000)
    
    ls = [(i,j) for i in range(0,len(coordinates)) for j in range(0,len(coordinates))]
    
    for i,j in ls:
        if i==j:
            continue
        model.add_link(
            start_point_id=i,
            end_point_id=j,
            distance=int(distances[i][j]))


    # solve model
    model.solve()
    model.export()

    if model.solution.is_defined():
        routes=[]
        for route in model.solution.routes:
           if len(route.point_ids)>1:
               routes.append(route.point_ids)
        
        return model.statistics.best_lb, routes,  model.statistics.root_time
    else:
        return None, None, None