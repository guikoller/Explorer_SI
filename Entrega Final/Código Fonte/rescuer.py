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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, nb_of_explorers=1,clusters=[]):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        self.nb_of_explorers = nb_of_explorers
        self.received_maps = 0
        self.map = Map()
        self.victims = {}
        self.plan = []
        self.x = 0
        self.y = 0
        self.clusters = clusters
        self.sequences = clusters
        self.distance_cache = {}

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

    def save_prediction_csv(self):
        filename = f"./predictions/file_predict.txt"
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in self.victims.items():
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
        """
        Prevê o valor e a classe de gravidade para cada vítima usando:
        - Regressor: Rede Neural (com dados escalados)
        - Classificador: Random Forest (com dados brutos)
        """
        # --- Carregamento dos Modelos e do Scaler Necessário ---

        # Carrega o modelo de CLASSIFICAÇÃO (Random Forest)
        # Este modelo NÃO precisa de um scaler.
        if os.path.exists('./models/trained_models/model_CART_classifier.pkl'):
            classifier = joblib.load('./models/trained_models/model_CART_classifier.pkl')
        else:
            print("Erro: Modelo classificador (model_CART_classifier.pkl) não encontrado.")
            return

        # Carrega o modelo de REGRESSÃO (Rede Neural) e seu scaler
        # Este modelo PRECISA do scaler com o qual foi treinado.
        if os.path.exists('./models/trained_models/model_neural_network_regressor.pkl'):
            regressor = joblib.load('./models/trained_models/model_neural_network_regressor.pkl')
            scaler_reg = joblib.load('./models/trained_models/scaler_regressor.pkl') # O scaler é essencial
        else:
            print("Erro: Modelo regressor ou seu scaler não encontrado.")
            return

        # --- Loop de Previsão ---
        
        for vic_id, values in self.victims.items():
            # Extrai os sinais vitais da estrutura de dados
            qPA = values[1][3]
            pulso = values[1][4]
            freqResp = values[1][5]

            # Cria um DataFrame para garantir a ordem e o formato corretos
            victim_data = pd.DataFrame([{
                'qPA': qPA,
                'pulso': pulso,
                'freqResp': freqResp
            }])

            # --- Previsão da CLASSE (Random Forest) ---
            # Usa os dados BRUTOS (não escalados), como no treinamento.
            severity_class = int(classifier.predict(victim_data.to_numpy())[0])
            # Se suas classes no ambiente são (1, 2, 3, 4), pode ser necessário somar 1.
            # Ex: severity_class = int(classifier.predict(victim_data.to_numpy())[0]) + 1

            # Adiciona 1 para converter a previsão de volta para a escala original [1, 2, 3, 4]
            severity_class = severity_class + 1

            # --- Previsão do VALOR (Rede Neural) ---
            # 1. Escala os dados usando o scaler específico do regressor
            victim_data_scaled = scaler_reg.transform(victim_data)

            # 2. Faz a previsão do valor contínuo com os dados escalados
            severity_value = regressor.predict(victim_data_scaled)[0]
            
            # Anexa os valores previstos à lista de sinais vitais da vítima
            values[1].extend([severity_value, severity_class])

    def create_population(self, sequence, pop_size):
        population = []
        population.append(self.greedy_individual(sequence))
        population.append(dict(sorted(sequence.items(), key=lambda item: item[1][0][0])))
        population.append(dict(sorted(sequence.items(), key=lambda item: item[1][0][1])))
        population.append(dict(sorted(sequence.items(), key=lambda item: item[1][1][6], reverse=True)))
        population.append(dict(sorted(sequence.items(), key=lambda item: 5 - item[1][1][7], reverse=True)))
        while len(population) < pop_size:
            individual_list = list(sequence.items())
            random.shuffle(individual_list)
            population.append(dict(individual_list))
        return population
    
    def greedy_individual(self, sequence):
        unvisited = list(sequence.items())
        current_position, individual = (0, 0), {}
        while unvisited:
            distances = [(item, self.distance_cache.get((current_position, item[1][0]), float('inf')) / (item[1][1][6] + 1)) for item in unvisited]
            distances.sort(key=lambda x: x[1])
            selected_victim = random.choice(distances[:1]) # Pega o melhor
            individual[selected_victim[0][0]] = selected_victim[0][1]
            current_position = selected_victim[0][1][0]
            unvisited.remove(selected_victim[0])
        return individual
    
    def calculate_score(self, individual):
        total_time, weighted_gravity_score, weighted_class_score, walking_time = 0, 0, 0, 0
        time_limit = self.TLIM - 100
        keys, start_pos = list(individual.keys()), (0, 0)
        for i, vic_id in enumerate(keys):
            goal_pos = individual[vic_id][0]
            vs = individual[vic_id][1]
            gravity, class_priority = vs[6], 5 - vs[7]
            norm_gravity, norm_priority = gravity / 100.0, class_priority / 4.0
            cost = self.distance_cache.get((start_pos, goal_pos), float('inf'))
            if cost == float('inf'): return float('inf')
            walking_time += cost
            total_time += cost + self.COST_FIRST_AID
            position_weight = (len(keys) - i) / len(keys)
            weighted_gravity_score += norm_gravity * position_weight
            weighted_class_score += norm_priority * position_weight
            start_pos = goal_pos
        return_cost = self.distance_cache.get((start_pos, (0,0)), float('inf'))
        if return_cost == float('inf'): return float('inf')
        total_time += return_cost
        walking_time += return_cost
        overtime_penalty = max(0, total_time - time_limit) * 10
        score = (walking_time + overtime_penalty - (weighted_gravity_score * 50) - (weighted_class_score * 50))
        return score
    
    def select_bests(self, scores, population):
        return [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0])][:len(population) // 2]
    
    def reproduce(self, selecteds):
        children = []
        for i in range(len(selecteds)):
            parent1_list = list(selecteds[i].items())
            parent2_list = list(selecteds[(i + 1) % len(selecteds)].items())
            start, end = sorted(random.sample(range(len(parent1_list)), 2))
            p1_slice = parent1_list[start:end]
            child_list = [None] * len(parent1_list)
            child_list[start:end] = p1_slice
            ids_from_p1 = {item[0] for item in p1_slice}
            p2_items_to_add = [item for item in parent2_list if item[0] not in ids_from_p1]
            p2_idx = 0
            for j in range(len(child_list)):
                if child_list[j] is None: child_list[j] = p2_items_to_add[p2_idx]; p2_idx += 1
            child = dict(child_list)
            if random.random() < 0.35:
                keys = list(child.keys()); idx1, idx2 = random.sample(range(len(keys)), 2)
                keys[idx1], keys[idx2] = keys[idx2], keys[idx1]; child = {key: child[key] for key in keys}
            if random.random() < 0.15:
                keys = list(child.keys()); start_mut, end_mut = sorted(random.sample(range(len(keys)), 2))
                sub_list = keys[start_mut:end_mut]; sub_list.reverse(); keys[start_mut:end_mut] = sub_list
                child = {key: child[key] for key in keys}
            children.append(child)
        return children

    def select_the_best(self, population, scores):
        if not scores: return None, float('inf')
        min_score = min(scores)
        return population[scores.index(min_score)], min_score

    def sequencing(self):
        pop_size, gen_size, early_stop_patience = 50, 100, 15
        new_sequences = []
        for seq in self.sequences:
            if not seq: continue
            logging.info(f"[{self.NAME}] Pre-calculating A* distances for {len(seq)} victims...")
            points_of_interest = {'base': (0, 0), **{vic_id: values[0] for vic_id, values in seq.items()}}
            a_star_calculator = AStar((0, 0), self.map)
            self.distance_cache = {(p1, p2): a_star_calculator.get_shortest_cost(p1, p2) for p1 in points_of_interest.values() for p2 in points_of_interest.values() if p1 != p2}
            logging.info(f"[{self.NAME}] Distance cache created.")
            
            population = self.create_population(seq, pop_size)
            best_individual, best_score, no_improvement_count = population[0], float('inf'), 0
            
            for gen in range(gen_size):
                scores = [self.calculate_score(ind) for ind in population]
                gen_best_individual, gen_best_score = self.select_the_best(population, scores)
                
                if gen_best_individual and gen_best_score < best_score:
                    best_score, best_individual, no_improvement_count = gen_best_score, gen_best_individual, 0
                    logging.info(f"[{self.NAME}] Generation {gen}: New best score = {best_score:.2f}")
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= early_stop_patience:
                    logging.info(f"[{self.NAME}] Early stopping at generation {gen}.")
                    break
                
                selecteds = self.select_bests(scores, population)
                children = self.reproduce(selecteds)
                population = [best_individual] + children + selecteds[:-1]

            if best_individual: new_sequences.append(best_individual)
        self.sequences = new_sequences

    def planner(self):
        if not self.sequences: return
        a_astar = AStar((0,0), self.map)
        sequence = self.sequences[0]
        
        complete_plan, total_cost = [], 0
        start_pos = (0,0)
        
        for vic_id in sequence:
            goal_pos = sequence[vic_id][0]
            plan_segment, time = a_astar.calc_plan(start_pos, goal_pos)
            if not plan_segment:
                logging.error(f"[{self.NAME}] Planner failed: No path from {start_pos} to {goal_pos}")
                return
            complete_plan += plan_segment
            total_cost += time + self.COST_FIRST_AID
            start_pos = goal_pos
        
        plan_back, time_back = a_astar.calc_plan(start_pos, (0,0))
        if not plan_back:
            logging.error(f"[{self.NAME}] Planner failed: No path back to base from {start_pos}")
            return

        complete_plan += plan_back
        total_cost += time_back

        if total_cost < self.TLIM:
            self.plan = complete_plan
            logging.info(f"[{self.NAME}] Plan created successfully. Total cost: {total_cost:.2f}")
        else:
            logging.warning(f"[{self.NAME}] Optimal plan discarded, time insufficient. Cost: {total_cost:.2f}, Time Limit: {self.TLIM}")
            self.plan = []

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

            self.save_prediction_csv()

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