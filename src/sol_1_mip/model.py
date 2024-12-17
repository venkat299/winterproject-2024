import pyomo.environ as pyo
import numpy as np

class CVRPModel:
    def __init__(self, num_nodes, num_vehicles, distances, demands, vehicle_capacity):
        self.env = pyo
        self.model = pyo.ConcreteModel()
        self.model.nodes = pyo.Set(initialize=range(num_nodes))
        self.model.edges = pyo.Set(
            initialize=[
                (i, j) for i in self.model.nodes for j in self.model.nodes if i != j
            ]
        )
        self.model.vehicles = pyo.Set(initialize=range(num_vehicles))

        self.model.capacity = pyo.Param(initialize=vehicle_capacity)
        self.model.distances = pyo.Param(
            self.model.edges,
            initialize={(i, j): distances[i, j] for (i, j) in self.model.edges},
        )
        self.model.demands = pyo.Param(
            self.model.nodes, initialize={i: d for (i, d) in enumerate(demands)}
        )

        self.model.x = pyo.Var(
            self.model.edges, self.model.vehicles, within=pyo.Binary
        )
        self.model.y = pyo.Var(
            self.model.nodes, self.model.vehicles, within=pyo.Binary
        )

    def get_model(self):
        return self.model
    
    def get_env(self):
        return self.env
    
    def find_tours(self):
        tours = []
        for k in self.model.vehicles:
            node = 0
            tours.append([0])
            while True:
                for j in self.model.nodes:
                    if (node, j) in self.model.edges:
                        if np.isclose(self.model.x[node, j, k].value, 1):
                            node = j
                            tours[-1].append(node)
                            break
                if node == 0:
                    break
        return tours
