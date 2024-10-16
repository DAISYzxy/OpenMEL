import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import faiss
import heapq
import re
from difflib import SequenceMatcher
from collections import Counter
   

def similar_strings(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


def to_np_arrary(s):
    numbers_str = re.findall(r"[-+]?\d*\.\d+e[+-]\d+|[-+]?\d*\.\d+|[-+]?\d+", s)
    numbers_float = [float(num) for num in numbers_str]
    array = np.array(numbers_float)
    
    return array


def below_threshold_index_lst(D, threshold=0.6):
    indices_below = []
    for row in D:
        index = next((i for i, x in enumerate(row) if x < threshold), None)
        indices_below.append(index)
    return indices_below


def process_mention(origin_Dme, origin_Ime, below_index):
    if below_index == None:
        below_index = len(origin_Ime) - 1
    D_me = list(origin_Dme)[:below_index+1]
    I_me = list(origin_Ime)[:below_index+1]
    return D_me, I_me


def process_entity(I_me, origin_Dee, origin_Iee, below_index_lst):
    D_ee = dict()
    I_ee = dict()
    for idx in range(len(origin_Dee)):
        tmp_Dee = origin_Dee[idx]
        tmp_idx = I_me[idx]
        tmp_Iee = origin_Iee[idx]
        below_idx = below_index_lst[idx]
        if below_idx == None:
            below_idx = len(tmp_Dee) - 1
        D_ee[tmp_idx] = list(tmp_Dee)[:below_idx+1]
        I_ee[tmp_idx] = list(tmp_Iee)[:below_idx+1]
    return D_ee, I_ee




class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, weight, u, v):
        heapq.heappush(self.heap, (-weight, u, v))

    def pop(self):
        weight, u, v = heapq.heappop(self.heap)
        return (-weight, u, v)


def find(parent, i):
    if i not in parent:
        parent[i] = i
    if parent[i] == i:
        return i
    else:
        parent[i] = find(parent, parent[i])
        return parent[i]


def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if xroot != yroot:
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1


def maximum_spanning_tree(m, I_me, D_me, I_ee, D_ee, K):
    results = [m]
    edge_list = []
    H = MaxHeap()
    parent = {}
    rank = {}

    # Initialize heap with edges from m to its nearest entities
    for i in range(len(I_me)):
        H.push(D_me[i], m, I_me[i])

    # Initialize Union-Find structure for the start node
    parent[m] = m
    rank[m] = 0

    while len(results) < K and len(H.heap) > 0:
        weight, u, v = H.pop()
        
        # Ensure the nodes are initialized in Union-Find structure
        if v not in parent:
            parent[v] = v
            rank[v] = 0
        
        if find(parent, u) != find(parent, v):
            union(parent, rank, u, v)
            results.append(v)
            edge_list.append((u, v, weight))

            # Push edges from v to its nearest entities
            if v in I_ee:
                for j in range(len(I_ee[v])):
                    w = I_ee[v][j]
                    if w not in results:
                        H.push(D_ee[v][j], v, w)

    return results, edge_list






