def arcs_in(model, i):
    if i == model.nodes.first():
        return sum(model.x[:, i, :]) == len(model.vehicles)
    else:
        return sum(model.x[:, i, :]) == 1.0

def arcs_out(model, i):
    if i == model.nodes.first():
        return sum(model.x[i, :, :]) == len(model.vehicles)
    else:
        return sum(model.x[i, :, :]) == 1.0

def vehicle_assignment(model, i, k):
    return sum(model.x[:, i, k]) == model.y[i, k]

def comp_vehicle_assignment(model, i, k):
    return sum(model.x[i, :, k]) == model.y[i, k]

def capacity_constraint(model, k):
    return sum(model.y[i, k] * model.demands[i] for i in model.nodes) <= model.capacity

def subtour_elimination(model, S, Sout, h, k):
    nodes_out = sum(model.x[i, j, k] for i in S for j in Sout)
    return model.y[h, k] <= nodes_out