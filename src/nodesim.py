import numpy as np
import networkx as nx
import random


class Graph():
	def __init__(self, nx_G, a, b):
		self.G = nx_G
		self.a = a
		self.b = b
		
	def nodesim_walk(self, walk_length, start_node):
		'''
		Simulate nodesim random walk starting from a given node.
		'''
		G = self.G
		probabilities=self.probabilities
		neighbors=self.neighbors
		walk = [start_node]
		while len(walk) < walk_length:
			cur = walk[-1]
			nextnode=random.choices(list(neighbors[cur]), list(probabilities[cur]))[0]
			walk.append(nextnode)
		return walk

	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly run random walks from each node.
		'''
		G = self.G
		walks = []
		nodes = list(G.nodes())
		print('Walk iteration:')
		for walk_iter in range(num_walks):
			print(str(walk_iter+1), '/', str(num_walks))
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.nodesim_walk(walk_length=walk_length, start_node=node))
		return walks

	def compute_edge_probs(self):
		'''
		Compute transition probabilities for nodesim random walks.
		'''
		def compute_jc(u, v):
			union_size = len(set(G[u]) | set(G[v]))
			if union_size == 0:
				return 0
			return len(list(nx.common_neighbors(G, u, v))) / union_size
			
		a=self.a
		b=self.b	
		G = self.G
		
		probs={}
		nghs={}
		for node in G.nodes():
			nghbrs=[]
			pr=[]
			for ngh in G.neighbors(node):
				nghbrs.append(ngh)
				pval=compute_jc(node, ngh) + 1.0/G.degree(node)
				if G.nodes[node]['community'] == G.nodes[ngh]['community']:
					pr.append(a*pval)
				else:
					pr.append(b*pval)
				
			s=sum(pr)
			pr=[x / s for x in pr]		
			probs[node]=pr
			nghs[node]=nghbrs
				
		self.probabilities=probs
		self.neighbors=nghs
		return	


