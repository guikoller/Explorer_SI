# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
# from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

from numpy.random import choice
from a_star import AStar
import heapq

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class EdgeManager:
    def __init__(self):
        self.edges = {}

    def add_edge(self, node1, node2, cost):
        if node1 not in self.edges:
            self.edges[node1] = {}
        if node2 not in self.edges:
            self.edges[node2] = {}
        self.edges[node1][node2] = cost
        self.edges[node2][node1] = cost

    def check_edge(self, node1, node2):
        return node1 in self.edges and node2 in self.edges[node1]

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 3             # the maximum degree of difficulty to enter into a cell
    
    def __init__(self, env, config_file, resc, priorities_vector):
        """ Construtor do agente
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.visited = set()
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        self.visited.add((self.x, self.y))
        self.is_coming_back = False
        self.back_plan = []        # the plan to come back to the base
        self.back_plan_cost = 0    # the cost of the plan to come back to the base
        self.a_start = AStar((0,0))
        self.movements = priorities_vector

    def get_next_position(self):
            obstacles = self.check_walls_and_lim()
            cost = self.map.get_difficulty((self.x, self.y))

            for movement in self.movements:
                dx, dy = Explorer.AC_INCR[movement]

                if obstacles[movement] == VS.CLEAR and (self.x + dx, self.y + dy) not in self.visited:
                    return (dx, dy)
                
                if obstacles[movement] == VS.CLEAR and (self.x + dx, self.y + dy) in self.visited and not self.a_start.check_edge((self.x, self.y), (self.x + dx, self.y + dy)):
                    cost_neighbor = self.map.get_difficulty((self.x + dx, self.y + dy))

                    if dx == 0 or dy == 0:
                        cost = cost * self.COST_LINE
                        cost_neighbor = cost_neighbor * self.COST_LINE

                    else:
                        cost = cost * self.COST_DIAG
                        cost_neighbor = cost_neighbor * self.COST_DIAG

                    self.a_start.add_edge((self.x, self.y), (self.x + dx, self.y + dy), cost, cost_neighbor)
                    
            direction = random.randint(0, 7)
            return Explorer.AC_INCR[direction]
        
    def explore(self):
            dx, dy = self.get_next_position()

            # checks whether the agent should backtrack due to all neighbors being visited
            # if all neighbors are visited or bumps into a wall or limit, return to the previous position
            if all([(self.x + incr[0], self.y + incr[1]) in self.visited 
                    or self.check_walls_and_lim()[i] == VS.WALL 
                    or self.check_walls_and_lim()[i] == VS.END for i, incr in Explorer.AC_INCR.items()]):
                dx, dy = self.walk_stack.pop()
                dx = -1 * dx
                dy = -1 * dy
            
            # Moves the body to another position
            rtime_bef = self.get_rtime()    # previous remaining time
            result = self.walk(dx, dy)      # walk to the new position
            rtime_aft = self.get_rtime()    # remaining time after the walk


            # Should never bump, but for safe functionning let's test
            if result == VS.BUMPED:
                # update the map with the wall
                self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
                
            if result == VS.EXECUTED:
                # store the new step (displacement) in the stack
                if (self.x + dx,self.y + dy) not in self.visited:
                    self.walk_stack.push((dx, dy))

                self.visited.add((self.x + dx,self.y + dy))
                # check for victim, returns -1 if there is no victim or the sequential
                # the sequential number of a found victim

                prev_x = self.x
                prev_y = self.y
                prev_diff = self.map.get_difficulty((prev_x, prev_y))

                # update the agent's position relative to the origin
                self.x += dx
                self.y += dy

                # update the walk time
                self.walk_time = self.walk_time + (rtime_bef - rtime_aft)

                # Check for victims
                seq = self.check_for_victim()
                if seq != VS.NO_VICTIM and seq not in self.victims:
                    vs = self.read_vital_signals()
                    # add the victim to the dictionary (vs[0] = victim id)
                    self.victims[vs[0]] = ((self.x, self.y), vs)

                # Calculates the difficulty (cost) of the visited cell
                difficulty = (rtime_bef - rtime_aft)
                if dx == 0 or dy == 0:
                    prev_diff = prev_diff * self.COST_LINE
                    self.a_start.add_edge((prev_x, prev_y), (self.x, self.y), difficulty, prev_diff)
                    difficulty = difficulty / self.COST_LINE
                else:
                    prev_diff = prev_diff * self.COST_DIAG
                    self.a_start.add_edge((prev_x, prev_y), (self.x, self.y), difficulty, prev_diff)
                    difficulty = difficulty / self.COST_DIAG

                # Update the map with the new cell
                self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
                
            return

    def come_back(self):
        dx, dy = self.walk_stack.pop()

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            
    def deliberate(self) -> bool:
        time_tolerance = 2* self.COST_DIAG * Explorer.MAX_DIFFICULTY + self.COST_READ

        # keeps exploring while there is enough time
        if  self.back_plan_cost + time_tolerance < self.get_rtime():
            self.explore()

            self.back_plan_cost = self.a_start.get_shortest_cost((self.x, self.y), (0,0))
            
            return True

        if not self.is_coming_back:
            self.is_coming_back = True
            self.back_plan, self.back_plan_cost = self.a_start.calc_backtrack((self.x, self.y))
            # updates walk_stack with the back_plan
            self.walk_stack = Stack()
            for action in self.back_plan[::-1]:
                self.walk_stack.push(action)

        # no more come back walk actions to execute or already at base
        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # prints position and time of the explorer when finishes
            print(f"{self.NAME}: at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
            # finishes the execution of this agent
            return False

        # proceed to the base
        self.come_back()
        return True

