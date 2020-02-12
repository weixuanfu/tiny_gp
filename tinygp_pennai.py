# tiny genetic programming plus, copyright moshe sipper, www.moshesipper.com
# graphic output, dynamic progress display, bloat-control option 
# need to install https://pypi.org/project/graphviz/

# cmd line: python tinygp_pennai.py [dataset.tsv] [popsize] [gencount]

from random import random, randint, seed
from pandas import read_csv
from pathlib import Path # correctly use folders on both Windows and Linux ('/' vs '\')
from statistics import mean
from copy import deepcopy
from IPython.display import Image, display
from graphviz import Digraph, Source 
from sys import exit, argv
from inspect import getfullargspec
from math import exp, sqrt, log, sin, cos, tan
from os import environ

MIN_DEPTH       = 2     # minimal initial random tree depth
MAX_DEPTH       = 5     # maximal initial random tree depth
TOURNAMENT_SIZE = 5     # size of tournament for tournament selection
XO_RATE         = 0.8   # crossover rate 
PROB_MUTATION   = 0.2   # per-node mutation probability 
BLOAT_CONTROL   = True # True adds bloat control to fitness function

popsize     = 10 # population size
generations = 10 # maximal number of generations to run evolution

def arity(func): return len(getfullargspec(func)[0])
def running_in_spyder(): return any('SPYDER' in name for name in environ)

def f_add2(x,y): return x+y
def f_sub2(x,y): return x-y
def f_mul2(x,y): return x*y
def f_div2(x,y): return x/sqrt(1+y*y)
def f_sqrt1(x):  return sqrt(abs(x))
def f_log1(x):   return 0 if abs(x)<0.0001 else log(abs(x))
def f_neg1(x):   return -x
def f_inv1(x):   return 0 if abs(x)<0.0001 else 1/x
def f_abs1(x):   return abs(x)
def f_max2(x,y): return max(x, y)
def f_min2(x,y): return min(x, y)
def f_sin1(x):   return sin(x)
def f_cos1(x):   return cos(x)
def f_tan1(x):   return tan(x)
def f_sig1(x):   return 0 if abs(x)>100 else 1/(1+exp(-x))

FUNCTIONS = [f_add2, f_sub2, f_mul2, f_div2, f_sqrt1, f_log1, f_neg1, f_inv1, f_abs1, f_max2, f_min2, f_sin1, f_cos1, f_tan1, f_sig1]

terminals = [] 
def set_terminals(dataset):
    global terminals
    terminals = list(dataset.columns.values)[:-1]
    print("Terminal set:",terminals)

def read_dataset(dsfile):
    return read_csv(Path(dsfile), delimiter = '\t')

class GPTree:
    def __init__(self, data = None, left = None, right = None):
        self.data  = data
        self.left  = left
        self.right = right
        
    def node_label(self): # return string label
        if (self.data in FUNCTIONS):
            return self.data.__name__
        else: 
            return str(self.data)
    
    def draw(self, dot, count): # dot & count are lists in order to pass "by reference" 
        node_name = str(count[0])
        dot[0].node(node_name, self.node_label())
        if self.left:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.left.draw(dot, count)
        if self.right:
            count[0] += 1
            dot[0].edge(node_name, str(count[0]))
            self.right.draw(dot, count)
        
    def draw_tree(self, fname, footer):
        dot = [Digraph()]
        dot[0].attr(kw='graph', label = footer)
        count = [0]
        self.draw(dot, count)
        if running_in_spyder():
            Source(dot[0], filename = fname + ".gv", format="png").render()
            display(Image(filename = fname + ".gv.png"))

    def compute_tree(self, datarow): 
        if (self.data in FUNCTIONS): 
            a = arity(self.data)
            if a == 1:
                return self.data(self.left.compute_tree (datarow))
            elif a == 2:                                 
                return self.data(self.left.compute_tree (datarow),\
                                 self.right.compute_tree(datarow))
        elif (self.data in terminals): return datarow[self.data]
        else: exit("GPTree compute_tree: unknown tree node")
            
    def random_tree(self, grow, max_depth, depth = 0): # create random tree using either grow or full method
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        elif depth >= max_depth:   
            self.data = terminals[randint(0, len(terminals)-1)]
        else: # intermediate depth, grow
            if random () > 0.5: 
                self.data = terminals[randint(0, len(terminals)-1)]
            else:
                self.data = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        if self.data in FUNCTIONS:
            a = arity(self.data)
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1) 
            if a == 2:                
                self.right = GPTree()
                self.right.random_tree(grow, max_depth, depth = depth + 1)
            if (a > 2): exit("random_tree: error in arity")    

    def mutation(self):
        if random() < PROB_MUTATION: # mutate at this node
            self.random_tree(grow = True, max_depth = 2)
        elif self.left: self.left.mutation()
        elif self.right: self.right.mutation() 
        
    def size(self): # tree size in nodes
        if self.data in terminals: return 1
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self): # count is list in order to pass "by reference"
        t = GPTree()
        t.data = self.data
        if self.left:  t.left  = self.left.build_subtree()
        if self.right: t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # note: count is list, so it's passed "by reference"
        count[0] -= 1            
        if count[0] <= 1: 
            if not second: # return subtree rooted here
                return self.build_subtree()
            else: # glue subtree here
                self.data  = second.data
                self.left  = second.left
                self.right = second.right
        else:  
            ret = None              
            if self.left  and count[0] > 1: ret = self.left.scan_tree(count, second)  
            if self.right and count[0] > 1: ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): # xo 2 trees at random nodes
        if random() < XO_RATE:
            second = other.scan_tree([randint(1, other.size())], None) # 2nd random subtree
            self.scan_tree([randint(1, self.size())], second) # 2nd subtree "glued" inside 1st tree
# end class GPTree
                   
def init_population(): # ramped half-and-half
    pop = []
    for md in range(3, MAX_DEPTH + 1):
        for i in range(int(popsize/6)):
            t = GPTree()
            t.random_tree(grow = True, max_depth = md) # grow
            pop.append(t) 
        for i in range(int(popsize/6)):
            t = GPTree()
            t.random_tree(grow = False, max_depth = md) # full
            pop.append(t) 
    return pop

def error(individual, dataset):
    i_target = dataset.shape[1] - 1
    return mean([float(abs(individual.compute_tree(dataset.iloc[i,:]) - dataset.iloc[i, i_target]))\
                 for i in range(dataset.shape[0])])

def fitness(individual, dataset): 
    if BLOAT_CONTROL:
        return 1 / (1 + error(individual, dataset) + 0.01*individual.size())
    else:
        return 1 / (1 + error(individual, dataset))
                
def selection(population, fitnesses): # select one individual using tournament selection
    tournament = [randint(0, len(population)-1) for i in range(TOURNAMENT_SIZE)] # select tournament contenders
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    return deepcopy(population[tournament[tournament_fitnesses.index(max(tournament_fitnesses))]])             

def main(): 
    if len(argv) < 4:
        exit('format: python tinygp_pennai.py [dataset.tsv] [popsize] [gencount]')
        
    dsfile = argv[1]
    global popsize, generations
    popsize = int(argv[2])
    generations = int(argv[3])
    
    print('dataset:', dsfile, '\npop:', popsize, '\ngens:', generations)
       
    # init stuff
    seed() # init internal state of random number generator
    dataset = read_dataset(dsfile)
    set_terminals(dataset)
    population= init_population() 
    best_of_run = None
    best_of_run_error = 1e20 
    best_of_run_gen = 0
    fitnesses = [fitness(ind, dataset) for ind in population]

    # go evolution!
    for gen in range(generations):        
        nextgen_population=[]
        for i in range(popsize):
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)
            parent1.crossover(parent2)
            parent1.mutation()
            nextgen_population.append(parent1)
        population=nextgen_population
        fitnesses = [fitness(ind, dataset) for ind in population]
        errors = [error(ind, dataset) for ind in population]
        if min(errors) < best_of_run_error:
            best_of_run_error = min(errors)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[errors.index(min(errors))])
            if running_in_spyder():
                print("________________________")
                best_of_run.draw_tree(dsfile + '_p' + str(popsize) + '_g' + str(generations),\
                                      "gen: " + str(gen) + ", error: " + str(round(best_of_run_error,3)))
        if best_of_run_error <= 1e-5: break
    
    if running_in_spyder():
        endrun = "_________________________________________________\nEND OF RUN (bloat control was "
        endrun += "ON)" if BLOAT_CONTROL else "OFF)"
        print(endrun,'\n\n')

    s = "best_of_run attained at gen " + str(best_of_run_gen) + " and has error=" + str(round(best_of_run_error,3))
    print(s)
    best_of_run.draw_tree("best_of_run",s)
    
if __name__== "__main__":
  main()