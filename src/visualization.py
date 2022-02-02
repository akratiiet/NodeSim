import numpy as np
from gensim.models import KeyedVectors
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import colors as mcolors

def colorlist():
	#generate a list of named color.
	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
	by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name) for name, color in colors.items())
	sorted_names = [name for hsv, name in by_hsv]
	random_colors=list(np.random.permutation(sorted_names)) 
	return random_colors

def plot_embedding_2d(G, emb):
	'''This method visulaizes the 2-dimensional embedding of the network. The method can visulaize the network having maximum 156 communities as the number of named colors in matplotlib is 156.'''
	random_colors=colorlist() 	
	for each in G.nodes:
		plt.plot(emb[str(each)][0], emb[str(each)][1], color=random_colors[G.nodes[each]['community']], marker='o', markersize=35)
		plt.annotate(str(each), (emb[str(each)][0]-0.01, emb[str(each)][1]-0.01), size=18)
	plt.xticks([])
	plt.yticks([])
	plt.show()

def plot_embedding_3d(G, emb):
	'''This method visulaizes the 3-dimensional embedding of the network. The method can visulaize the network having 156 communities as the number of named colors in matplotlib is 156.'''
	random_colors=colorlist()
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	for each in G.nodes:		
		ax.scatter(emb[str(each)][0], emb[str(each)][1], emb[str(each)][2], facecolor=(0,0,0,0), edgecolors=random_colors[G.nodes[each]['community']], marker='o' , s=100, linewidths=3)
	ax.set_xticks([])
	ax.set_yticks([])
	ax.set_zticks([])
	ax.grid(False)
	plt.show()
	
def plot_network(G):
	'''This method plots the given network G.'''
	node_sizes=[]
	li=[]
	cc=colorlist()
	for each in G.nodes():
		node_sizes.append(800)
		li.append(cc[G.nodes[each]['community']])
	pos = nx.spring_layout(G)
	nx.draw_networkx_nodes(G, pos, node_color=li,node_size=node_sizes,font_size=20,with_labels=True)
	nx.draw_networkx_labels(G, pos,font_size=20)
	nx.draw_networkx_edges(G, pos, edgelist=list(G.edges()), edge_color='black')
	ax = plt.gca()
	ax.set_axis_off()
	plt.show()


emb = KeyedVectors.load_word2vec_format("Output/karate_2d.emb",  binary=False)
G=nx.read_gpickle("Input/karate.gpickle")		
plot_embedding_2d(G, emb)	
plot_network(G)
	
