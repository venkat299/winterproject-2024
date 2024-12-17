import numpy as np 
import hygese as hgs 

# Solver initialization
def build_and_solve(data, timeLimit):
    ap = hgs.AlgorithmParameters(timeLimit=timeLimit)  # seconds
    hgs_solver = hgs.Solver(parameters=ap, verbose=False)

    # Solve
    result = hgs_solver.solve_cvrp(data)
    # print(result.cost)
    # print(result.routes)
    return  result.cost/10000, result.routes, result.time
    
    