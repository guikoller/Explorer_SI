##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### Not a complete version of DFS; it comes back prematuraly
### to the base when it enters into a dead end position

import joblib
import os
import random
import math
import csv
import sys
import logging
import concurrent.futures
import os
import random
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod
from bfs import BFS
from a_star import AStar

import numpy  as np
import joblib
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1,clusters=[]):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.received_maps = 0                       # counts the number of explorers' maps
        self.map = Map()                             # explorer will pass the map
        self.victims = {}            # a dictionary of found victims: [vic_id]: ((x,y), [<vs>])
        self.plan = []               # a list of planned actions in increments of x and y
        self.plan_x = 0              # the x position of the rescuer during the planning phase
        self.plan_y = 0              # the y position of the rescuer during the planning phase
        self.plan_visited = set()    # positions already planned to be visited 
        self.plan_rtime = self.TLIM  # the remaing time during the planning phase
        self.plan_walk_time = 0.0    # previewed time to walk during rescue
        self.x = 0                   # the current x position of the rescuer when executing the plan
        self.y = 0                   # the current y position of the rescuer when executing the plan
        self.clusters = clusters     # the clusters of victims this agent should take care of - see the method cluster_victims
        self.sequences = clusters    # the sequence of visit of victims for each cluster 
        
        # A* algorithm to calculate the path between victims
        self.a_star = AStar((0, 0), self.map)

        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)
    
    def save_cluster_csv(self, cluster, cluster_id):
        filename = f"./clusters/cluster{cluster_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for vic_id, values in cluster.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([vic_id, x, y, vs[6], vs[7]])


    def save_sequence_csv(self, sequence, sequence_id):
        filename = f"./clusters/seq{sequence_id}.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])


    def cluster_victims(self):
        # Find the upper and lower limits for x, y, and gravity class
        lower_xlim = sys.maxsize    
        lower_ylim = sys.maxsize
        lower_gclass = sys.maxsize
        upper_xlim = -sys.maxsize - 1
        upper_ylim = -sys.maxsize - 1
        upper_gclass = -sys.maxsize - 1

        for key, values in self.victims.items():
            x, y = values[0]
            gravity_class = values[1][7]
            lower_xlim = min(lower_xlim, x) 
            upper_xlim = max(upper_xlim, x)
            lower_ylim = min(lower_ylim, y)
            upper_ylim = max(upper_ylim, y)
            lower_gclass = min(lower_gclass, gravity_class)
            upper_gclass = max(upper_gclass, gravity_class)

        # K-means clustering
        max_iter = 150
        k = 4

        # Initialize the centroids
        centroids = []
        for i in range(k):
            x = random.uniform(lower_xlim, upper_xlim)
            y = random.uniform(lower_ylim, upper_ylim)
            gravity_class = random.uniform(lower_gclass, upper_gclass)
            centroids.append((x, y, gravity_class))
        
        clusters = [{} for _ in range(k)]
        centroid_changed = True
        iteration = 0

        while (iteration < max_iter) and (centroid_changed):
            centroid_changed = False
            clusters = [{} for _ in range(k)]  # Reset clusters

            # Assign victims to the nearest centroid
            for key, values in self.victims.items():
                x, y = values[0]
                gravity_class = values[1][7]
                distances = [math.sqrt((x - cx)**2 + (y - cy)**2 + (gravity_class - cg)**2) for cx, cy, cg in centroids]
                min_distance_index = distances.index(min(distances))
                clusters[min_distance_index][key] = values

            # Recalculate the centroids
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    avg_x = sum(values[0][0] for values in cluster.values()) / len(cluster)
                    avg_y = sum(values[0][1] for values in cluster.values()) / len(cluster)
                    avg_gravity_class = sum(values[1][7] for values in cluster.values()) / len(cluster)
                    new_centroids.append((avg_x, avg_y, avg_gravity_class))
                else:
                    new_centroids.append((
                        random.uniform(lower_xlim, upper_xlim),
                        random.uniform(lower_ylim, upper_ylim),
                        random.uniform(lower_gclass, upper_gclass)
                    ))

            if new_centroids != centroids:
                centroid_changed = True
                centroids = new_centroids

            iteration += 1

        return clusters

    def predict_severity_and_class(self):
        if os.path.exists('./models/modelo_random_forest.pkl'):
                model = joblib.load('./models/modelo_random_forest.pkl')

        
        regressor = joblib.load('./models/modelo_random_forest.pkl')
        for vic_id, values in self.victims.items():
            qPA = values[1][3]
            pulso = values[1][4]
            freqResp = values[1][5]

            victim_data = pd.DataFrame([{
                'qPA': qPA,
                'pulso': pulso,
                'freqResp': freqResp
            }])

            # try:
            y_pred = model.predict(victim_data.to_numpy())  # Uma classe, ex: [2]
            # severity_value =  random.uniform(0.1, 99.9)          # to be replaced by a regressor 
            severity_value = regressor.predict(victim_data.to_numpy())[0]
            severity_class = int(y_pred[0])
            values[1].extend([severity_value, severity_class])  # append to the list of vital signals; values is a pair( (x,y), [<vital signals list>] )

    def create_population(self, sequence, pop_size):
        logging.debug("Creating initial population")
        population = []

        # Add a greedy individual
        population.append(self.greedy_individual(sequence))

        # Add individuals sorted by different heuristics
        by_x = dict(sorted(sequence.items(), key=lambda item: item[1][0][0]))
        by_y = dict(sorted(sequence.items(), key=lambda item: item[1][0][1]))
        by_gravity = dict(sorted(sequence.items(), key=lambda item: item[1][1][6], reverse=True))  # highest gravity first
        by_priority = dict(sorted(sequence.items(), key=lambda item: 5 - item[1][1][7], reverse=True))  # highest priority first

        population.extend([by_x, by_y, by_gravity, by_priority])

        # Fill the rest randomly
        while len(population) < pop_size:
            individual = list(sequence.items())
            random.shuffle(individual)
            population.append(dict(individual))

        logging.debug(f"Initial population created with {len(population)} individuals.")
        return population

    
    def greedy_individual(self, sequence):
        """Create an individual using a greedy approach with some randomness."""
        unvisited = list(sequence.items())
        current_position = (0, 0)
        individual = {}
        a_star = AStar((0, 0), self.map)

        while unvisited:
            # Find the nearest unvisited victims
            distances = [
                (item, a_star.get_shortest_cost(current_position, item[1][0]) / (item[1][1][6] + 1))  # prioritize severity
                for item in unvisited
            ]
            distances.sort(key=lambda x: x[1])

            # Introduce randomness: select one of the nearest victims
            nearest_victims = distances[:3]  # Consider the 3 nearest victims
            selected_victim = random.choice(nearest_victims)
            
            individual[selected_victim[0][0]] = selected_victim[0][1]
            current_position = selected_victim[0][1][0]
            unvisited.remove(selected_victim[0])

        return individual
    
    def calculate_score(self, individual):
        a_astar = AStar((0, 0), self.map)

        total_time = 0
        weighted_gravity_score = 0
        weighted_class_score = 0
        walking_time = 0
        time_limit = self.TLIM - 100  # buffer for return

        keys = list(individual.keys())
        start = (0, 0)

        for i, vic_id in enumerate(keys):
            goal = individual[vic_id][0]
            vs = individual[vic_id][1]

            gravity = vs[6]           # [0–100]
            class_priority = 5 - vs[7]  # class: 1 → 4, 2 → 3, ..., 4 → 1

            # Normalize both scores to [0, 1]
            norm_gravity = gravity / 100
            norm_priority = class_priority / 4

            # Path cost
            cost = a_astar.get_shortest_cost(start, goal)
            if cost == -1:
                return float('inf')  # invalid path

            cost += self.COST_FIRST_AID
            total_time += cost
            walking_time += cost

            # Give more weight to early rescues of high-priority victims
            position_weight = (len(keys) - i) / len(keys)

            weighted_gravity_score += norm_gravity * position_weight
            weighted_class_score += norm_priority * position_weight

            start = goal

        # Normalize walking time (if needed, depending on max possible time in your map)
        norm_walk_time = walking_time / max(self.TLIM, 1)

        # Penalize plans that exceed the time limit
        overtime_penalty = 0
        if total_time > time_limit:
            overtime_penalty = (total_time - time_limit) * 10

        # Weighted score formula
        score = (
            weighted_gravity_score * 50 +
            weighted_class_score * 50 -
            norm_walk_time * 40 +
            overtime_penalty
        )

        logging.debug(f"Score for individual: {score}")
        logging.debug(f"Time for individual: {total_time}")
        return score
    
    def select_bests(self, scores, population):
        logging.debug("Selecting best individuals")
        # Sort the population based on the scores in ascending order
        sorted_population = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0], reverse=True)]
        # Select the top half of the sorted population
        selected_population = sorted_population[:len(population) // 2]
        # logging.debug(f"Selected best individuals: {selected_population}")
        return selected_population
    
    def reproduce(self, selecteds):
        logging.debug("Reproducing new generation")
        children = []
        num_selected = len(selecteds)
        mutation_rate = 0.35  # Slightly higher for exploration

        for i in range(num_selected):
            parent1 = list(selecteds[i].items())
            parent2 = list(selecteds[(i + 1) % num_selected].items())

            # Order Crossover (OX)
            start, end = sorted(random.sample(range(len(parent1)), 2))
            child = [None] * len(parent1)
            child[start:end] = parent1[start:end]
            parent2_items = [item for item in parent2 if item not in child]
            child = [item if item is not None else parent2_items.pop(0) for item in child]
            child = dict(child)

            # Swap Mutation
            if random.random() < mutation_rate:
                keys = list(child.keys())
                idx1, idx2 = random.sample(range(len(keys)), 2)
                keys[idx1], keys[idx2] = keys[idx2], keys[idx1]
                child = {key: child[key] for key in keys}

            # Inversion Mutation (new)
            if random.random() < 0.15:
                keys = list(child.items())
                start, end = sorted(random.sample(range(len(keys)), 2))
                keys[start:end] = reversed(keys[start:end])
                child = dict(keys)

            children.append(child)

        return children
    
    def select_the_best(self, population):
        logging.debug("Selecting the best individual from the population")
        best = min(population, key=self.calculate_score)
        # logging.debug(f"Best individual selected: {best}")
        return best

    def sequencing(self):
        """ Genetic Algorithm with Elitism, Early Stopping, and Multiprocessing """
        from concurrent.futures import ThreadPoolExecutor

        pop_size = 20
        gen_size = 10
        early_stop_patience = 5
        new_sequences = []

        for seq in self.sequences:
            population = self.create_population(seq, pop_size)
            best_individual = None
            best_score = float('inf')
            no_improvement = 0

            for gen in range(gen_size):
                logging.info(f"Generation {gen}")

                # Use multiprocessing for faster scoring
                with ThreadPoolExecutor() as executor:
                    scores = list(executor.map(self.calculate_score, population))

                # Find best in generation
                min_gen_score = min(scores)
                gen_best = population[scores.index(min_gen_score)]

                # Update best overall if improved
                if min_gen_score < best_score:
                    best_score = min_gen_score
                    best_individual = gen_best
                    no_improvement = 0
                    logging.info(f"New best score: {best_score:.2f}")
                else:
                    no_improvement += 1
                    logging.info(f"No improvement: {no_improvement} consecutive generations")

                # Early stopping condition
                if no_improvement >= early_stop_patience:
                    logging.info(f"Early stopping at generation {gen} (patience {early_stop_patience})")
                    break

                # Selection, reproduction, elitism
                selecteds = self.select_bests(scores, population)
                children = self.reproduce(selecteds)
                population = [best_individual] + selecteds + children[:-1]  # apply elitism

            new_sequences.append(best_individual)

        self.sequences = new_sequences

    def planner(self):
        """ A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
            after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis."""

        # The planner uses the A* algorithm to calculate the path between victims
        a_astar = AStar((0,0),self.map)

        # for each victim of the first sequence of rescue for this agent, we're going go calculate a path
        # starting at the base - always at (0,0) in relative coords
        
        if not self.sequences:   # no sequence assigned to the agent, nothing to do
            return

        # we consider only the first sequence (the simpler case)
        # The victims are sorted by x followed by y positions: [vic_id]: ((x,y), [<vs>]

        sequence = self.sequences[0]
        start = (0,0) # always from starting at the base

        for vic_id in sequence:
            goal = sequence[vic_id][0]
            # print(f"{self.NAME} Plan: from {start} to {goal}")
            # plan, time = bfs.search(start, goal, self.plan_rtime)

            # remaining time is not necessary here
            plan_back, plan_back_cost = a_astar.calc_plan(goal, (0,0))
            # calculate next move based on time available (-1 is used for aditional time)
            plan, time = a_astar.calc_plan(start, goal, self.plan_rtime - plan_back_cost - 5)
            # if plan and time != -1:
            #     self.plan += plan
            #     self.plan_rtime = self.plan_rtime - time - 1
            #     print("Remaining time for the rescuer: ", self.plan_rtime)
            #     start = goal
            # else:
            #     print(f"{self.NAME} Plan fail - no path between {start} and {goal}")
            #     break
            if time == -1:
                print(f"{self.NAME} Plan fail - no path between {start} and {goal}")
                break
            self.plan += plan
            self.plan_rtime = self.plan_rtime - time - 5
            print("Remaining time for the rescuer: ", self.plan_rtime)
            start = goal

        # Plan to come back to the base
        goal = (0,0)
        plan_back, plan_back_cost = a_astar.calc_plan(start, goal, self.plan_rtime)
        self.plan = self.plan + plan_back
        self.plan_rtime = self.plan_rtime - plan_back_cost
        print("Remaining time for the rescuer: ", self.plan_rtime)
        print("Time to get back to the base: ", plan_back_cost)
    
    def sync_explorers(self, explorer_map, victims):
        """ This method should be invoked only to the master agent

        Each explorer sends the map containing the obstacles and
        victims' location. The master rescuer updates its map with the
        received one. It does the same for the victims' vital signals.
        After, it should classify each severity of each victim (critical, ..., stable);
        Following, using some clustering method, it should group the victims and
        and pass one (or more)clusters to each rescuer """

        self.received_maps += 1

        print(f"{self.NAME} Map received from the explorer")
        self.map.update(explorer_map)
        self.victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            print(f"{self.NAME} all maps received from the explorers")
            #self.map.draw()
            #print(f"{self.NAME} found victims by all explorers:\n{self.victims}")

            #TODO: predict the severity and the class of victims' using a classifier
            self.predict_severity_and_class()

            #cluster the victims possibly using the severity and other criteria
            # Here, there 4 clusters
            clusters_of_vic = self.cluster_victims()

            for i, cluster in enumerate(clusters_of_vic):
                self.save_cluster_csv(cluster, i+1)    # file names start at 1

            # Instantiate the other rescuers
            rescuers = [None] * 4
            rescuers[0] = self                    # the master rescuer is the index 0 agent

            # Assign the cluster the master agent is in charge of 
            self.clusters = [clusters_of_vic[0]]  # the first one

            # Instantiate the other rescuers and assign the clusters to them
            for i in range(1, 4):    
                #print(f"{self.NAME} instantianting rescuer {i+1}, {self.get_env()}")
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                # each rescuer receives one cluster of victims
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters_of_vic[i]]) 
                rescuers[i].map = self.map     # each rescuer have the map

            
            # Calculate the sequence of rescue for each agent
            # In this case, each agent has just one cluster and one sequence
            self.sequences = self.clusters         

            # print("Victims--->", self.victims)

            # For each rescuer, we calculate the rescue sequence 
            for i, rescuer in enumerate(rescuers):
                rescuer.sequencing()         # the sequencing will reorder the cluster
                
                for j, sequence in enumerate(rescuer.sequences):
                    if j == 0:
                        self.save_sequence_csv(sequence, i+1)              # primeira sequencia do 1o. cluster 1: seq1 
                    else:
                        self.save_sequence_csv(sequence, (i+1)+ j*10)      # demais sequencias do 1o. cluster: seq11, seq12, seq13, ...

            
                rescuer.planner()            # make the plan for the trajectory
                rescuer.set_state(VS.ACTIVE) # from now, the simulator calls the deliberation method 
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)
        # print(f"{self.NAME} pop dx: {dx} dy: {fdy} ")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            # if(self.NAME == "RESC_4"):
                # print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")
            # print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")

            # check if there is a victim at the current position
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
                    #if self.first_aid(): # True when rescued
                        #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")                    
        else:
            # if(self.NAME == "RESC_4"):
                # print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.y}) + ({dx},{dy})")
            pass
            
        return True


