from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import time

def solve_cvrp_ortools(data):
    """
    Solves a Capacitated Vehicle Routing Problem (CVRP) using Google OR-Tools.
    
    Returns:
      A tuple containing:
        - A list of routes, where each route is a list of customer indices.
        - The total cost of the solution.
    """
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data["demands"][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data["vehicle_capacities"],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(1)

    # Solve the problem.
    start_time = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    # Extract the solution
    routes = []
    total_cost = 0
    if solution:
        index = routing.Start(0)
        # routes = []
        for vehicle_id in range(data["num_vehicles"]):
            index = routing.Start(vehicle_id)
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                next_index = solution.Value(routing.NextVar(index))
                total_cost += distance_callback(index, next_index)
                index = next_index
                
                route.append(node_index)
            if len(route)>1:
                routes.append(route)
    return total_cost, routes, execution_time
    
    
