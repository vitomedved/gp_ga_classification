import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# custom funckija koja sluzi kao operator u stablu
def protected_div(arg1, arg2):
    if arg2 == 0:
        return 0
    return float(arg1 / arg2)

# custom funckija koja sluzi kao operator u stablu
def if_then_else(condition, out1, out2):
    if condition < 0:
        return out1
    return out2

# uzmi dataset i broj argumenata koje cemo imati
file = open("dataset.data")
lines = file.readlines()

# ovdje uzimam jednu liniju kako bih mogao izvaditi duljinu argumenata
arr = lines[0][:-1].split(",")
# num of args je jednak indexu zadnjeg stupca koji pogadamo tako da pazi na to
numOfArgs = len(arr[:-1])

# definiraj kako ce stablo izgledati
pset = gp.PrimitiveSet("MAIN", numOfArgs)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(if_then_else, 3)

# https://deap.readthedocs.io/en/master/api/creator.html
# creator radi klasu imena arg0, bazna klasa je arg1, a arg2 sluzi kao parametri koje zelimo inicijalizirati u toj novoj klasi
# ovdje radim dvije klase, FitnessMin i Individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# https://deap.readthedocs.io/en/master/api/base.html#toolbox
# napravimo objekt toolboxa koji sadrzi evolutionary operatore
# sa register u taj objekt registriramo funkcije po zelji na sljedeci nacin:
#   arg0 = alias funkcije, arg1 = funkcija koju zelimo registrirati, arg2 = 1 ili vise argumenata koji ce biti defaultni kod poziva te funkcije
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# funkcija koja ce evaluirati tocnost stabla
def evalTrues(individual, dataLines):
    # pretvori tree expresion u kod koji python moze izvrsiti, tj. u funkciju koju onda smao pozivamo
    func = toolbox.compile(expr=individual)

    # counteri koji broje koliko je tocno pogodenih linija podataka, a koliko je ukupno linija
    # TODO: ukupni broj linija se vjerojatno moze dobiti preko len(dataLines), ali naravno ako kod radi, ne treba ga mijenjati
    correct = 0
    guesses = 0

    # idi kroz svaku liniju u datasetu
    # nisam siguran kako da odaberem pola za "train", a pola za "test" pa je ovaj zakomentirani dio dio koji kaze da se ide samo do pola dataseta, ne kroz cijeli
    for line in dataLines:#[:len(dataLines) / 2]:
        # parsiraj liniju u array podataka
        currentLine = line.strip().split(",")

        # razdvoji liniju na parametre koje moramo pogadati i stvarni rezultat
        actualResult = currentLine[-1]
        currentLine = map(float, currentLine[:-1])

        #dobij neku vrijednost iz klasifikatora
        guess = func(*currentLine)

        # logika ovdje je da pretpostavljam da je 0 threshold, ako klasifikator vrati >0, to znaci da je rekao da je rezultat 'g', tj. 'good'
        # to je hardkodirana vrijednost pa se kod razlicitih dataseta ovo mora rucno promijeniti u taj zadnji element (1 ili 0)
        # ovo naravno vrijedi samo za klasifikaciju, a ne regresiju
        if(guess > 0 and actualResult == 'g') or (guess <= 0 and actualResult == 'b'):
            correct += 1

        guesses += 1

    # vrati accuracy
    result = float(correct) / float(guesses)
    
    # ovaj "result," je dosta zanimljiv jer ima zarez nakon varijable, to je zato jer moram vratiti tupple, inace program ne radi :)))))
    return result,

# ovdje opet ista prica kao i gore, registriram neke funkcije sa aliasima i defaultnim parametrima
# neke od ovih funkcija treba postaviti prije pozivanja evolucijskog algoritma, za tocne info o tome koje funkcije treba postaviti, checkiraj dokumentaciju, link je dolje negdje
# isto tako provjeri: https://deap.readthedocs.io/en/master/api/tools.html gdje su navedeni svi tools-i koje koristimo poput gp.selTournament, gp.cxOnePoint, itd. 
# TODO: umjesto tournament selecetion koristi NSGA2 ili SPEA2

# evaluation
toolbox.register("evaluate", evalTrues, dataLines=lines)
# selection - algoritam kojim odabiremo rezultate
# TODO: koristiti neki drugi selection algoritam koji se koristi u radu
toolbox.register("select", tools.selTournament, tournsize=3)
# crossover - napravi crossover izmedu dva stabla na jednoj tocki svakog stabla
toolbox.register("mate", gp.cxOnePoint)
# initialization - koristimo stabla gdje svaki list ima jednaku dubinu izmedu min-max, u biti odreduje dubinu stabla na neki nacin
# moze se i koristiti gp.genGrow gdje stabla mogu imati razlicite dubine, a moze se i koristiti genHalfAndHalf gdje se mijesa geFull i genGrow
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
# mutiranje - uzme neku random tocku na stablu i zamijeni je sa stablom koje ce se generirati iz expr argumenta
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# "dekorira" funkcije koje su vec registrirane
# ako se ne varam, ovo kaze da visina stabla ne bude veca od max_value, nisam siguran tho
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    # koristimo zbog pseudo-random generiranj brojeva, tj. trebamo dati neku drukciji seed svaki put kako bi osigurali
    # da nam random funkcija vraca random rjesenja svaki put
    random.seed(318)

    # velicina populacije koju zelimo imati i optimizirati
    pop = toolbox.population(n=20)

    # lista od x najboljih rjesenja, dolje ima link za referencu
    hof = tools.HallOfFame(1)
    
    # napravimo objekt statistike i registriramo nutra funkcije statistike koje zelimo imati (u ovom slucaju za fitness)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)


    # https://deap.readthedocs.io/en/master/api/algo.html
    # jednostavni evolucijski algoritam
    # arg0 = populacija koju optimiziramo, arg1 = toolbox sa svim funkcijama koje trebamo imati definirane, arg2 = vjerojatnost za crossover stabla,
    # arg3 = vjerojatnost mutiranja stablam arg4 = broj generacija, arg5 = objekt u koji se sprema statistika, 
    # arg6 = objekt koji sadrzi najbolja drva koja su spremna za sjecu i pretvaranje u namjestaj, arg7 = hocu li loggati statistiku ili ne
    pop, log = algorithms.eaSimple(pop, toolbox, 0.2, 0.4, 40, stats=mstats,
                                   halloffame=hof, verbose=True)

    return pop, log, hof

# za return i parametre funkcije pogledaj link na liniji 114 (nadajmo se da se ta linija nece previse mijenjati)
# pop = optimizirana populacija nakon x generacija
# log = LogBook klasa, u biti log svake generacije
# hof = najbolji klasifikatori svake generacije
pop, log, hof = main()

print(len(pop), len(log), len(hof))

'''
hof ce vratiti listu koja sadrzi najbolje klasifikatore (hof[0] vraca onaj najbolji klasifikator)
https://deap.readthedocs.io/en/0.7-a1/halloffame.html

'''
# spremi cvorove, bridove i nazive iz najboljeg stabla
nodes, edges, labels = gp.graph(hof[0])

# crtanje grafa i spremanje u pdf, ne da mi se vise komentirati, mislim da je intuitivno na dalje
import pygraphviz as pgv

g = pgv.AGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
g.layout(prog="dot")

for i in nodes:
    n = g.get_node(i)
    n.attr["label"] = labels[i]

g.draw("tree.pdf")