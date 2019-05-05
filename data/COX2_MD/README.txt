README for dataset COX2_MD


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs

(1) 	DS_A.txt (m lines) 
	sparse (block diagonal) adjacency matrix for all graphs,
	each line corresponds to (row, col) resp. (node_id, node_id)

(2) 	DS_graph_indicator.txt (n lines)
	column vector of graph identifiers for all nodes of all graphs,
	the value in the i-th line is the graph_id of the node with node_id i

(3) 	DS_graph_labels.txt (N lines) 
	class labels for all graphs in the dataset,
	the value in the i-th line is the class label of the graph with graph_id i

(4) 	DS_node_labels.txt (n lines)
	column vector of node labels,
	the value in the i-th line corresponds to the node with node_id i

There are OPTIONAL files if the respective information is available:

(5) 	DS_edge_labels.txt (m lines; same size as DS_A_sparse.txt)
	labels for the edges in DD_A_sparse.txt 

(6) 	DS_edge_attributes.txt (m lines; same size as DS_A.txt)
	attributes for the edges in DS_A.txt 

(7) 	DS_node_attributes.txt (n lines) 
	matrix of node attributes,
	the comma seperated values in the i-th line is the attribute vector of the node with node_id i

(8) 	DS_graph_attributes.txt (N lines) 
	regression values for all graphs in the dataset,
	the value in the i-th line is the attribute of the graph with graph_id i


=== Description of the dataset === 

Dataset derived from the chemical compound dataset COX2 which comes with 
3D coordinates. We generated complete graphs from the compounds, where 
edges are attributed with distances and labeled with the chemical bond 
type (single, double, triple or aromatic). Vertex labels correspond to 
atom types. Explicit hydrogen atoms have been removed. Chemical data was 
processed using the Chemistry Development Kit (v1.4).
A filtering technique removing highly similar structures ("coverage-based
reduction") was applied as in the original publication.

Node labels:

  0  C
  1  N
  2  F
  3  S
  4  O
  5  Cl
  6  Br

Edge labels:

  0  aromatic
  1  no chemical bond
  2  single
  3  double
  4  triple

Edge attributes:

  The distance between atoms.


=== Previous Use of the Dataset ===

This dataset was used in:

Kriege, N., Mutzel, P.: Subgraph matching kernels for attributed graphs. In: Proceedings
of the 29th International Conference on Machine Learning (ICML-2012) (2012).

Mahé, P.; Ralaivola, L.; Stoven, V. & Vert, J.-P. The pharmacophore kernel
for virtual screening with support vector machines. J Chem Inf Model, 2006,
46, 2003-2014


=== References ===

Sutherland, J. J.; O'Brien, L. A. & Weaver, D. F. Spline-fitting with a
genetic algorithm: a method for developing classification structure-activity
relationships. J. Chem. Inf. Comput. Sci., 2003, 43, 1906-1915
