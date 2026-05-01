#       Name						ID				section
#Sari Murad Abdalghani		     1220982			     4
#Mohammad Shamasneh               1220092			     1

#Instructor:	Dr. Yazan Abu Farha

import random
import math
import copy
import matplotlib.pyplot as plt # type: ignore
from matplotlib.animation import FuncAnimation # type: ignore
import numpy as np # type: ignore


packages_list = []
cars_list = []
states = []
packages = []
cars = []
car_routes=[]



with open("packages.txt", 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        
        package = {
            "name": parts[0],
            "x": float(parts[1]),
            "y": float(parts[2]),
            "weight": float(parts[3]),
            "priority": int(parts[4])
        }
        
        packages_list.append(package)

T = 1000
with open("cars.txt", 'r') as file:
    for line in file:
        parts = line.strip().split(',')
        
        car = {
            "name": parts[0],
            "capacity": float(parts[1])
        }
        
        cars_list.append(car)


# Function to load data from files

def load_data():
    global packages, cars
    packages = []
    cars = []
    
    # Read packages file
    with open("packages.txt", "r") as f:
        for line in f:
            data = line.strip().split(",")
            package = {
                "name": data[0],
                "x": float(data[1]),
                "y": float(data[2]),
                "weight": float(data[3]),
                "priority": int(data[4])
            }
            packages.append(package)
    
    # Read cars file
    with open("cars.txt", "r") as f:
        for line in f:
            data = line.strip().split(",")
            car = {
                "name": data[0],
                "capacity": float(data[1])
            }
            cars.append(car)




################################################################################################################################################
################################################################################################################################################
############################################                                            ########################################################
############################################    The Function of Simulated Annealing     ########################################################
############################################                                            ########################################################
################################################################################################################################################
################################################################################################################################################




# ------------------- Random Initial State -------------------

def randomInitialState() :
 global states
 states =[]
 for car in cars_list :

    point = {
        "name" : car['name'],
        "capacity" : car["capacity"],
        "packages" : [[0,0,0,0]]
            }
    states.append(point)

 packages_list.sort(key=lambda p: p["priority"])
 for package in packages_list :
     point = [package["x"],package["y"],package["weight"],package["priority"],package["name"]]
     test = 100
     while True :
      test -=1
      car = random.choice(states)
      if package['weight'] <= car['capacity'] :
       car['packages'].append(point)
       car["capacity"] -= package["weight"]
       break
      elif test == 0 : 
         print("Cannot distribute package due to capacity limitations.")        
         print(f'Package {point} could not be assigned.')
         test =100
         break
 return states



# ------------------- Distance Calculation -------------------

def calculation(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)



# ------------------- Random Neighbor Generator -------------------

def randomNextState(states):
    for i in range(1000):   
        new_states = copy.deepcopy(states)
        choice = random.random()

        if choice < 0.25:
            car = random.choice(new_states)
            if len(car['packages']) > 2:
                point1, point2 = random.sample(range(1, len(car["packages"])), 2)
                car["packages"][point1], car["packages"][point2] = car["packages"][point2], car["packages"][point1]

        elif choice < 0.5 and len(new_states) >= 2:
            car1, car2 = random.sample(new_states, 2)
            if len(car1["packages"]) <= 1 or len(car2["packages"]) <= 1:
                continue

            idx1 = random.randint(1, len(car1["packages"]) - 1)
            idx2 = random.randint(1, len(car2["packages"]) - 1)

            pkg1 = car1["packages"][idx1]
            pkg2 = car2["packages"][idx2]

            new_cap1 = car1["capacity"] + pkg1[2] - pkg2[2]
            new_cap2 = car2["capacity"] + pkg2[2] - pkg1[2]

            if new_cap1 >= 0 and new_cap2 >= 0:
                car1["packages"][idx1], car2["packages"][idx2] = pkg2, pkg1
                car1["capacity"] = new_cap1
                car2["capacity"] = new_cap2

        elif choice < 0.75 and len(new_states) >= 2:
            car_from, car_to = random.sample(new_states, 2)
            if len(car_from["packages"]) <= 1:
                continue

            idx = random.randint(1, len(car_from["packages"]) - 1)
            pkg = car_from["packages"][idx]

            if car_to["capacity"] >= pkg[2]:
                car_from["packages"].pop(idx)
                car_from["capacity"] += pkg[2]
                car_to["packages"].append(pkg)
                car_to["capacity"] -= pkg[2]

        else:
            car = random.choice(new_states)
            if len(car["packages"]) <= 2:
                continue

            base = [0, 0]
            remaining = car["packages"][1:]
            ordered = []

            while remaining:
                nearest = min(remaining, key=lambda p: calculation(base[0], base[1], p[0], p[1]))
                ordered.append(nearest)
                base = [nearest[0], nearest[1]]
                remaining.remove(nearest)

            car["packages"] = [[0, 0, 0, 0]] + ordered

        return new_states  

    return states 


# ------------------- Objective Function -------------------

def objectiveFunc(states, priority_factor=0.2):
    total_cost = 0
    for state in states:
        points = state["packages"]
        points = [p for p in points if not (p[0] == 0 and p[1] == 0)]
        if not points:
            continue

        cost = 0
        first_x, first_y = 0, 0
        for point in points:
            second_x, second_y = point[0], point[1]
            priority = point[3]   

            distance = calculation(first_x, first_y, second_x, second_y)

            weighted_distance = distance * (1 + priority_factor * priority)

            cost += weighted_distance

            first_x, first_y = second_x, second_y

        cost += calculation(first_x, first_y, 0, 0)
        total_cost += cost

    return total_cost


# ------------------- Probability Calculation -------------------

def calculationProbability(delta_E,T):
    power = delta_E / T
    result = math.exp(power)
    #print(result)
    return result


# ------------------- Final Distance Only -------------------

def finalObjectiveFunc(states):
    total_distance = 0
    for state in states :
       points = state["packages"]
       points = [p for p in points if not (p[0] == 0 and p[1] == 0)]
       if not points :
          continue 
       car_distance = 0
       first_x =0
       first_y =0
       for point in points :
          second_x = point[0]
          second_y = point[1]
          car_distance += calculation(first_x,first_y,second_x,second_y) 
          first_x = second_x
          first_y = second_y

       car_distance += calculation(first_x, first_y, 0, 0)
       total_distance += car_distance


    return total_distance


# ------------------- Visualization -------------------

def animate_paths(states):
    plt.figure(figsize=(12, 10))
    plt.scatter(0, 0, c='red', marker='s', s=150, label='Depot', zorder=5)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    for idx, state in enumerate(states):
        points = state["packages"]
        if len(points) <= 1:
            continue

        color = colors[idx % len(colors)]
        x_coords = [0] + [p[0] for p in points[1:]] + [0]
        y_coords = [0] + [p[1] for p in points[1:]] + [0]

        total_weight = sum(p[2] for p in points[1:])
        route_len = len(points) - 1

        plt.plot(
            x_coords, y_coords, marker='o', linestyle='-',
            color=color, linewidth=2, markersize=8,
            label=f"{state['name']} ({route_len} pkgs, {total_weight:.1f}kg)"
        )

        for p in points[1:]:
            plt.annotate(
                f"Pri:{p[3]}", (p[0], p[1]),
                textcoords="offset points", xytext=(0, 5), ha='center'
            )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Car Routes (Static View)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# ------------------- Simulated Annealing -------------------

def simulatedAnnealing(T_initial=1000, cooling_rate=0.95, max_iter=1000):
    T = T_initial
    current = randomInitialState()

    for i in range(max_iter):
        T *= cooling_rate
        if T < 1 :
            break

        nextState = randomNextState(current)
        delta_E = objectiveFunc(current) - objectiveFunc(nextState)
        #print(objectiveFunc(nextState))
        if delta_E > 0:
            current = nextState
        elif random.random() < calculationProbability(delta_E, T):
            current = nextState

    return current




################################################################################################################################################
################################################################################################################################################
############################################                                            ########################################################
############################################    The Function of Genetic Algorithm       ########################################################
############################################                                            ########################################################
################################################################################################################################################
################################################################################################################################################



# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# Function to evaluate how good a solution
def evaluate_solution(solution):
    total_distance = 0
    global car_routes
    car_routes=[]
    global Dictance
    Dictance=0
    remain_capcity=0
    priority_factor=0

    car_solution = []
    for car_idx in range(len(cars)):
        sel=[]
        for pkg_idx in range(len(solution)):
            if solution[pkg_idx] == car_idx:
                sel.append(pkg_idx)
                
        car_solution.append(sel)
    for car_idx in range(len(cars)):
        point= [0,0]
        routes=[]
        car_capacity=cars[car_idx]["capacity"]
        remaining = car_solution[car_idx].copy()
        nearest=None
        for p in range(len(car_solution[car_idx])):
            min_Dictance = 150

            for pkg in remaining:
                dist = calculate_distance(point[0], point[1],packages[pkg]["x"], packages[pkg]["y"])
                if (dist < min_Dictance  or ((dist-min_Dictance <= 5) and (packages[nearest]["priority"] > packages[pkg]["priority"])) ):
                    min_Dictance = dist
                    nearest = pkg
               
            if nearest in remaining:
                point[0]=packages[nearest]["x"]
                point[1]=packages[nearest]["y"]
                car_capacity -=packages[nearest]["weight"]
                priority_factor+=packages[nearest]["priority"]
                routes.append(nearest)
                total_distance += min_Dictance
                remaining.remove(nearest)
                if car_capacity-packages[pkg]["weight"] < 0:
                    break
        if len(car_solution[car_idx]) > 0:
            total_distance+= calculate_distance(packages[nearest]["x"],packages[nearest]["y"],0,0)
        remain_capcity+=car_capacity
        car_routes.append(routes)
        Dictance=total_distance
    return total_distance + remain_capcity*100 + priority_factor*5


# Function to create random initial solutions
def create_population(size):
    population = []
    
    for N in range(size):
        solution = []
        remaining_capacities = [car["capacity"] for car in cars]
        
        sorted_packages = sorted( range(len(packages)), key=lambda i: ( -packages[i]["priority"],-calculate_distance(0, 0, packages[i]["x"], packages[i]["y"])))

        for pkg_idx in sorted_packages:
            pkg = packages[pkg_idx]
            valid_vehicles = [i for i in range(len(cars)) if remaining_capacities[i] >= pkg["weight"]]
            
            if valid_vehicles:
                car_idx = max(valid_vehicles, key=lambda x: remaining_capacities[x])
                solution.append(car_idx)
                remaining_capacities[car_idx] -= pkg["weight"]
            else:
                car_idx = min(range(len(cars)),  key=lambda x: max(0, remaining_capacities[x] - pkg["weight"]))
                solution.append(car_idx)
        
        population.append(solution)
    
    return population

def select_parent(population):
    num = random.randint(0,len(population)-1)
    parent=population[num]
    return parent

def crossover(parent1, parent2):
    point = random.randint(1, len(parent1)-2)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(solution, mutation_rate):
    if random.random() < mutation_rate:
        n,m= random.sample(range(len(solution)),2)
        temp=solution[n]
        solution[n]=solution[m]
        solution[m]=temp
    return solution


def Genetic_Algorithm():
    population_size = 100
    mutation_rate = 0.1
    generations = 500
    
    population = create_population(population_size)
    best_solution = min(population, key=lambda x: evaluate_solution(x))
    for generation in range(generations):
        new_population = []
        
        for size in range(population_size):
            parent1 = select_parent(population)
            parent2 = select_parent(population)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=lambda x: evaluate_solution(x))
        if evaluate_solution(current_best) < evaluate_solution(best_solution):
            best_solution = current_best
    return best_solution

def plot_car_routes(car_routes):
    plt.figure(figsize=(12, 10))
    plt.scatter(0, 0, c='red', marker='s', s=150, label='Depot', zorder=5)

    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    delivered = set()

    for car_idx, route in enumerate(car_routes):
        if not route:
            continue

        route_coords = [(0, 0)]  
        for pkg_idx in route:
            route_coords.append((packages[pkg_idx]['x'], packages[pkg_idx]['y']))
            delivered.add(pkg_idx)
        route_coords.append((0, 0))  

        x_coords = [pt[0] for pt in route_coords]
        y_coords = [pt[1] for pt in route_coords]

        total_weight = sum(packages[p]['weight'] for p in route)
        plt.plot(
            x_coords, y_coords, marker='o', linestyle='-',
            color=colors[car_idx % len(colors)],
            linewidth=2, markersize=8,
            label=f"Car {cars[car_idx]['name']} ({len(route)} pkgs, {total_weight}/{cars[car_idx]['capacity']}kg)"
        )

        for pkg_idx in route:
            pkg = packages[pkg_idx]
            plt.annotate(
                f"{pkg['name']}\nPri:{pkg['priority']}",
                (pkg['x'], pkg['y']),
                textcoords="offset points",
                xytext=(0, 5), ha='center'
            )

    # Show undelivered packages
    undelivered = [i for i in range(len(packages)) if i not in delivered]
    if undelivered:
        plt.scatter(
            [packages[i]['x'] for i in undelivered],
            [packages[i]['y'] for i in undelivered],
            c='red', marker='x', s=100, label='Undelivered'
        )
        for i in undelivered:
            pkg = packages[i]
            plt.annotate(
                f"{pkg['name']}\nPri:{pkg['priority']}",
                (pkg['x'], pkg['y']),
                textcoords="offset points",
                xytext=(0, 5), ha='center'
            )

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Optimized Package Delivery Routes (Respecting Capacity Limits)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



################################################################################################################################################
################################################################################################################################################
############################################                                            ########################################################
############################################                Main Function               ########################################################
############################################                                            ########################################################
################################################################################################################################################
################################################################################################################################################

load_data()

while True:
    print("\n=== MENU ===")
    print("1. Use Simulated Annealing")
    print("2. Use Genetic Algorithm")
    print("0. Exit")
    choice = input("Enter your choice (0, 1 or 2): ")

    if choice == '0':
        print("Exiting program. Goodbye!")
        break


    if choice == '1':
        best_state = simulatedAnnealing()
        print('\n\n\n')
        print(f"Total Dictance:{finalObjectiveFunc(best_state):.2f}")
        for carr in best_state :
            print(f"Name : {carr['name']}  packages : {[p[4] for p in carr['packages'][1:]]} ")
        animate_paths(best_state)


    elif choice == '2':
        
        solution = Genetic_Algorithm()
        evaluate_solution(solution)
        for i in range(len(cars)):
            if car_routes[i] != []:
                print(f"Name : {cars[i]['name']}: {[packages[p]['name'] for p in car_routes[i]]}")

                
        print(f"Total Dictance: {Dictance:.2f}")
        plot_car_routes(car_routes)

    else:
        print("Invalid choice. Please enter 0, 1, or 2.")


