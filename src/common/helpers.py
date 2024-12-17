import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import cycle
from scipy.spatial.distance import pdist, squareform
import os 
import pickle
import asyncio

def get_output_path(opt):
    dataset_basename, ext = os.path.splitext(os.path.split(opt.dataset)[-1])
    solver_name = "_".join(os.path.normpath(os.path.splitext(opt.solver)[0]).split(os.sep)[-2:])
    if opt.o is None:
        results_dir = os.path.join(opt.results_dir, solver_name)
        os.makedirs(results_dir, exist_ok=True)

        out_file = os.path.join(results_dir, "{}-{}{}".format(
            dataset_basename, solver_name, ext
        ))
    else:
        out_file = opt.o
    return out_file
        

def load_data():
    np.random.seed(42)

    num_customers = 20
    demands = np.random.randint(1, 10, size=num_customers)
    demands[0] = 0  # Depot has no demand

    vehicle_capacity = 30
    num_vehicles = 4

    # Generate random coordinates for customers
    coordinates = np.random.rand(num_customers, 2)
    # Calculate distances between customers
    distances = squareform(pdist(coordinates, metric="euclidean"))
    distances = np.round(distances, decimals=4)
    print(num_customers, demands, vehicle_capacity, num_vehicles, coordinates, distances)
    return num_customers, demands, vehicle_capacity, num_vehicles, coordinates, distances



def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped

def load_pkl_file(file_path):
  with open(file_path, 'rb') as file:
    return pickle.load(file)

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def load_random_data():
    np.random.seed(42)

    num_customers = 10
    demands = np.random.randint(1, 10, size=num_customers)
    demands[0] = 0  # Depot has no demand

    vehicle_capacity = 15
    num_vehicles = 4

    # Generate random coordinates for customers
    coordinates = np.random.rand(num_customers, 2)
    # Calculate distances between customers
    distances = squareform(pdist(coordinates, metric="euclidean"))
    distances = np.round(distances, decimals=4)
    return num_customers, demands, vehicle_capacity, num_vehicles, coordinates, distances

def transform_data(data):
    num_customers = 10
    demands = np.random.randint(1, 10, size=num_customers)
    demands[0] = 0  # Depot has no demand

    vehicle_capacity = 15
    num_vehicles = 4

    # Generate random coordinates for customers
    coordinates = np.random.rand(num_customers, 2)
    # Calculate distances between customers
    distances = squareform(pdist(coordinates, metric="euclidean"))
    distances = np.round(distances, decimals=4)
    return num_customers, demands, vehicle_capacity, num_vehicles, coordinates, distances


def visualize_tours(tours, coordinates):
    # Choosing colors
    cmap = mpl.colormaps["Dark2"]
    colors = cycle(cmap.colors)

    # Now the figure
    fig, ax = plt.subplots(figsize=[6, 5], dpi=100)
    for r, tour in enumerate(tours):
        c = next(colors)
        t = np.array(tour)
        x = coordinates[t, 0]
        y = coordinates[t, 1]
        ax.scatter(x, y, color=c, label=f"R{r}")
        ax.plot(x, y, color=c)

    ax.legend()
    fig.tight_layout()
    plt.show()


def append_to_pickle(filename, item):
    """Appends an item to a list stored in a pickle file.

    Args:
    filename: The name of the pickle file.
    item: The item to append to the list.
    """
    try:
      # Try to load the existing list from the file
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except EOFError:
        # If the file is empty, create an empty list
        data = []
    except FileNotFoundError:
        # If the file doesn't exist, create an empty list
        data = []

    # Append the new item to the list
    data.append(item)

    # Write the updated list back to the file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
