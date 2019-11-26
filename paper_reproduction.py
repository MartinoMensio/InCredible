# -*- coding: latin-1 -*-
from __future__ import division
import os
from os import listdir
from os.path import isfile, isdir, join
import time
import re
import json
import math
import numpy as np
from random import shuffle
import pickle
import math
import string
import random
from types import *
import matplotlib.pyplot as plt
import collections
import pandas as pd
import nltk
import graphDoc
import sys
import getopt
import glob
from multiprocessing.pool import Pool
from tqdm import tqdm
from collections import defaultdict

import functions
import plot

__author__ = 'Dimitrios Bountouridis'

def clique_processing(clique_id, clique_data):
    contents = clique_data["contents"]
    publications = clique_data["publications"]
    titles = clique_data["sentences"]
        
    print("Initialize object...")
    gDoc=graphDoc.graphDocuments(contents,publications,titles)   # initialize object with documents and publication classes e.g. cnn, fox
    
    print("Extracting sentence structure...")
    gDoc.sentenceProcess(withGA=True, output=f"temp/sentences_{clique_id}.pkl")
    
    print("Computing sentence similarities...")
    gDoc.computeSentenceDistances(similarityFunction = "cosine")

    print("Keeping only the most important sentence-to-sentence similarities (thresholding)...")
    gDoc.reduceSentenceSimilarityFrame(pA=85,pB=93)

    print("Create graph... (no plotting)")
    gDoc.computeNetwork(plot=False,cliqueEdges=[])

    print("Clique finder in the graph...")
    gDoc.cliqueFinder(output=f"temp/cliques_GA/cliques_{clique_id}.json",orderby="median tf-idf score")

def fn_wrap(arg):
    # print(type(arg[0]), type(arg[1]))
    try:
        clique_processing(arg[0], arg[1])
        return arg[0]
    except Exception as e:
        print(e)
        raise ValueError(arg[0])

def all_clique_processing(jsonFile):
    # Find cross-referenced pieces if information 
    print("Reading the documents")
    cliqueOfArticles = functions.readJsonFile(jsonFile)
    items = cliqueOfArticles.items()
    
    pool = Pool(processes=8)
    for clique_id in tqdm(pool.imap_unordered(fn_wrap, items), desc='outer loop'):
        print(clique_id, 'done')


def stats(jsonFile):
    # Computes some stats about initial cliques
    print('computing stats about input')
    cliqueOfArticles = functions.readJsonFile(jsonFile)
    size_acc = 0
    size_distrib = []
    clique_sim_distrib = []
    per_outlet = defaultdict(lambda: 0)
    for k, v in cliqueOfArticles.items():
        publications = v['publications']
        size_acc += len(publications)
        size_distrib.append(float(len(publications)))
        clique_sim_distrib.append(v['score'])
        for o in publications:
            per_outlet[o] += 1
    
    print('total articles', size_acc)
    print('#cliques', len(cliqueOfArticles))
    print('average clique len', size_acc / len(cliqueOfArticles))
    print(size_distrib)
    plot.plot_distribution(size_distrib, 'temp/fig_clique_size_distrib.png')
    plot.plot_distribution(clique_sim_distrib, 'temp/fig_clique_sim_distrib.png')
    print(per_outlet)

def results_from_cliques(cliques_results_data, threshold = 0.25):
    # cliques_results_data is {clique_id: clique_results} where results is the file produced by gDoc.cliqueFinder
    # TODO three bins of similarity based on the 'median tf-idf score' of each POI
    outlets_poi_stats = defaultdict(lambda: defaultdict(lambda: 0))
    poi_averages = []
    # quantile computation: split in 3 almost equal sized bins
    for clique_id, clique_results in cliques_results_data.items():
        for poi_id, poi_results in clique_results.items():
            poi_average_similarity = poi_results['average clique similarity']
            if poi_average_similarity > threshold:
                poi_averages.append(poi_average_similarity)
    # quantiles computation, q12 = 0.5, q23=0.75
    q12 = np.quantile(np.array(poi_averages), q=1./3)
    q23 = np.quantile(np.array(poi_averages), q=2./3)
    print('q12', q12, 'q23', q23)

    for clique_id, clique_results in cliques_results_data.items():
        # first collect all the outlets that refer to this story
        all_outlets_publishing_about_this = set()
        for poi_id, poi_results in clique_results.items():
            publications = poi_results['publications']
            all_outlets_publishing_about_this.update(publications)

        # and then see for each POI if they covered it or not
        for poi_id, poi_results in clique_results.items():
            publications = poi_results['publications']
            poi_average_similarity = poi_results['average clique similarity']
            # apply threshold of POI similarity
            if poi_average_similarity > threshold:
                for o in all_outlets_publishing_about_this:
                    if o in publications and len(publications) > 1:
                        # this is corroborated POI
                        label = 'corroborated'
                    else:
                        # this POI is omitted
                        label = 'omitted'
                    which_bin = 1 if poi_average_similarity < q12 else (2 if poi_average_similarity < q23 else 3)
                    outlets_poi_stats[o][f'{label}_{which_bin}'] += 1
    #print(outlets_poi_stats.items())
    plot.plot_distribution(poi_averages, 'temp/fig_POI_sim_distrib.png')
    plot.figure_3(outlets_poi_stats, 'temp/fig_3.png')
    return outlets_poi_stats

def load_clique_results(glob_expression):
    file_matches = glob.glob(glob_expression)
    result = {}
    for f_name in file_matches:
        f_content = functions.readJsonFile(f_name)
        result[f_name] = f_content
    return result
    

   
def main(recompute=False):
    jsonFile = 'Data/dataset.json'
    stats(jsonFile)
    if recompute:
        all_clique_processing(jsonFile)
    cliques_results_data = load_clique_results('temp/cliques_GA/cliques_*')
    results_from_cliques(cliques_results_data)
    
    


if __name__ == "__main__":
   main()           
            
