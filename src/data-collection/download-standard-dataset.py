import vrplib
import tqdm
import os
problem_names = vrplib.list_names(low=0, high=20, vrp_type='cvrp') 

instances = ['A', 'B', 'E', 'F', 'M'] # Collect Set A, B, E, F, M datasets
for name in problem_names:
    if 'A' in name:
        instances.append(name)
    elif 'B' in name:
        instances.append(name)
    elif 'E' in name:
        instances.append(name)
    elif 'F' in name:
        instances.append(name)
    elif 'M' in name and 'CMT' not in name:
        instances.append(name)

path_to_save = './data/vrplib/' 

try:
    os.makedirs(path_to_save)
    for instance in tqdm(instances):
        print(instance)
        vrplib.download_instance(instance, path_to_save)
        vrplib.download_solution(instance, path_to_save)
except: # already exist
    pass