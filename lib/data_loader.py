# -*- coding: utf-8 -*-

from graph import Graph,wl_labeling
import networkx as nx
from utils import per_section,indices_to_one_hot
from collections import defaultdict
import numpy as np
import math
class NotImplementedError(Exception):
    pass

"""
All methods for loading the data
"""

def load_local_data(data_path,name,one_hot=False,attributes=True,use_node_deg=False,wl=0):
    """ Load local datasets    
    Parameters
    ----------
    data_path : string
                Path to the data. Must link to a folder where all datasets are saved in separate folders
    name : string
           Name of the dataset to load. 
           Choices=['mutag','ptc','nci1','imdb-b','imdb-m','enzymes','protein','protein_notfull','bzr','cox2','synthetic','aids','cuneiform'] 
    one_hot : integer
              If discrete attributes must be one hotted it must be the number of unique values.
    attributes :  bool, optional
                  For dataset with both continuous and discrete attributes. 
                  If True it uses the continuous attributes (corresponding to "Node Attr." in [5])
    use_node_deg : bool, optional
                   Wether to use the node degree instead of original labels. 
    wl : integer, optional
         For dataset with discrete attributes.
         Relabels the graph with a Weisfeler-Lehman procedure. wl is the number of iteration of the procedure
         See wl_labeling in graph.py
    Returns
    -------
    X : array
        array of Graph objects created from the dataset
    y : array
        classes of each graph    
    References
    ----------    
    [5] Kristian Kersting and Nils M. Kriege and Christopher Morris and Petra Mutzel and Marion Neumann 
        "Benchmark Data Sets for Graph Kernels"
    """
    if name=='mutag':
        path=data_path+'/MUTAG_2/'
        dataset=build_MUTAG_dataset(path,one_hot=one_hot)
    if name=='ptc':
        path=data_path+'/PTC_MR/'
        dataset=build_PTC_dataset(path,one_hot=one_hot)
    if name=='nci1':
        path=data_path+'/NCI1/'
        if one_hot==True:
            raise NotImplementedError
        dataset=build_NCI1_dataset(path)
    if name=='imdb-b':
        path=data_path+'/IMDB-BINARY/'
        dataset=build_IMDB_dataset(path,s='BINARY',use_node_deg=use_node_deg)
    if name=='imdb-m':
        path=data_path+'/IMDB-MULTI/'
        dataset=build_IMDB_dataset(path,s='MULTI',use_node_deg=use_node_deg)
    if name=='enzymes':
        path=data_path+'/ENZYMES_2/'
        if attributes:
            dataset=build_ENZYMES_dataset(path,type_attr='real')
        else:
            dataset=build_ENZYMES_dataset(path)
    if name=='protein':
        path=data_path+'/PROTEINS_full/'
        if attributes:
            dataset=build_PROTEIN_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_PROTEIN_dataset(path)
    if name=='protein_notfull':
        path=data_path+'/PROTEINS/'
        if attributes:
            dataset=build_PROTEIN2_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_PROTEIN2_dataset(path)
    if name=='bzr':
        path=data_path+'/BZR/'
        if attributes:
            dataset=build_BZR_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_BZR_dataset(path)
    if name=='cox2':
        path=data_path+'/COX2/'
        if attributes:
            dataset=build_COX2_dataset(path,type_attr='real',use_node_deg=use_node_deg)
        else:
            dataset=build_COX2_dataset(path)
    if name=='synthetic':
        path=data_path+'/SYNTHETIC/'
        if attributes:
            dataset=build_SYNTHETIC_dataset(path,type_attr='real')
        else:
            dataset=build_SYNTHETIC_dataset(path)
    if name=='aids':
        path=data_path+'/AIDS/'
        if attributes:
            dataset=build_AIDS_dataset(path,type_attr='real')
        else:
            dataset=build_AIDS_dataset(path)
    if name=='cuneiform':
        path=data_path+'/Cuneiform/'
        if attributes:
            dataset=build_Cuneiform_dataset(path,type_attr='real')
        else:
            dataset=build_Cuneiform_dataset(path)
    if name=='letter_high':
        path=data_path+'/Letter-high/'
        if attributes:
            dataset=build_LETTER_dataset(path,type_attr='real',name='high')
        else:
            dataset=build_LETTER_dataset(path,name='med')
    if name=='letter_med':
        path=data_path+'/Letter-med/'
        if attributes:
            dataset=build_LETTER_dataset(path,type_attr='real',name='med')
        else:
            dataset=build_LETTER_dataset(path,name='med') 
    if name=='fingerprint':
        path=data_path+'/Fingerprint/'
        dataset=build_Fingerprint_dataset(path,type_attr='real')
    X,y=zip(*dataset)
    if wl!=0:
        X=label_wl_dataset(X,h=wl)
    return np.array(X),np.array(y)
    
def build_noisy_circular_graph(N=20,mu=0,sigma=0.3,with_noise=False,structure_noise=False,p=None):
    g=Graph()
    g.add_nodes(list(range(N)))
    for i in range(N):
        noise=float(np.random.normal(mu,sigma,1))
        if with_noise:
            g.add_one_attribute(i,math.sin((2*i*math.pi/N))+noise)
        else:
            g.add_one_attribute(i,math.sin(2*i*math.pi/N))
        g.add_edge((i,i+1))
        if structure_noise:
            randomint=np.random.randint(0,p)
            if randomint==0:
                if i<=N-3:
                    g.add_edge((i,i+2))
                if i==N-2:
                    g.add_edge((i,0))
                if i==N-1:
                    g.add_edge((i,1))
    g.add_edge((N,0))
    noise=float(np.random.normal(mu,sigma,1))
    if with_noise:
        g.add_one_attribute(N,math.sin((2*N*math.pi/N))+noise)
    else:
        g.add_one_attribute(N,math.sin(2*N*math.pi/N))
    return g

#%%

def label_wl_dataset(X,h):
    X2=[]
    for x in X:
        x2=Graph()
        x2.nx_graph=wl_labeling(x.nx_graph,h=2)
        X2.append(x2)
    return X2

#%%

def histog(X,bins=10):
    node_length=[]
    for graph in X:
        node_length.append(len(graph.nodes()))
    return np.array(node_length),{'histo':np.histogram(np.array(node_length),bins=bins),'med':np.median(np.array(node_length))
            ,'max':np.max(np.array(node_length)),'min':np.min(np.array(node_length))}

def node_labels_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=int(elt)
            k=k+1
    return node_dic

def node_attr_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=[float(x) for x in elt.split(',')]
            k=k+1
    return node_dic

def graph_label_list(path,name):
    graphs=[]
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            graphs.append((k,int(elt)))
            k=k+1
    return graphs
    
def graph_indicator(path,name):
    data_dict = defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            data_dict[int(elt)].append(k)
            k=k+1
    return data_dict

def compute_adjency(path,name):
    adjency= defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        for elt in sections[0]:
            adjency[int(elt.split(',')[0])].append(int(elt.split(',')[1]))
    return adjency


def all_connected(X):
    a=[]
    for graph in X:
        a.append(nx.is_connected(graph.nx_graph))
    return np.all(a)

#%% TO FACTORIZE !!!!!!!!!!!
def build_NCI1_dataset(path):
    node_dic=node_labels_dic(path,'NCI1_node_labels.txt')
    node_dic2={}
    for k,v in node_dic.items():
        node_dic2[k]=v-1
    node_dic=node_dic2
    graphs=graph_label_list(path,'NCI1_graph_labels.txt')
    adjency=compute_adjency(path,'NCI1_A.txt')
    data_dict=graph_indicator(path,'NCI1_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_PROTEIN_dataset(path,type_attr='label',use_node_deg=False):
    if type_attr=='label':
        node_dic=node_labels_dic(path,'PROTEINS_full_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'PROTEINS_full_node_attributes.txt')
    graphs=graph_label_list(path,'PROTEINS_full_graph_labels.txt')
    adjency=compute_adjency(path,'PROTEINS_full_A.txt')
    data_dict=graph_indicator(path,'PROTEINS_full_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data
    
def build_PROTEIN2_dataset(path,type_attr='label',use_node_deg=False):
    if type_attr=='label':
        node_dic=node_labels_dic(path,'PROTEINS_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'PROTEINS_node_attributes.txt')
    graphs=graph_label_list(path,'PROTEINS_graph_labels.txt')
    adjency=compute_adjency(path,'PROTEINS_A.txt')
    data_dict=graph_indicator(path,'PROTEINS_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_MUTAG_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'MUTAG_graph_labels.txt')
    adjency=compute_adjency(path,'MUTAG_A.txt')
    data_dict=graph_indicator(path,'MUTAG_graph_indicator.txt')
    node_dic=node_labels_dic(path,'MUTAG_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],7)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data


def build_IMDB_dataset(path,s='MULTI',use_node_deg=False):
    graphs=graph_label_list(path,'IMDB-'+s+'_graph_labels.txt')
    adjency=compute_adjency(path,'IMDB-'+s+'_A.txt')
    data_dict=graph_indicator(path,'IMDB-'+s+'_graph_indicator.txt')
    #node_dic=node_labels_dic(path,'MUTAG_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            #g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))
        
    

    return data

def build_PTC_dataset(path,one_hot=False):
    graphs=graph_label_list(path,'PTC_MR_graph_labels.txt')
    adjency=compute_adjency(path,'PTC_MR_A.txt')
    data_dict=graph_indicator(path,'PTC_MR_graph_indicator.txt')
    node_dic=node_labels_dic(path,'PTC_MR_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],18)
                g.add_one_attribute(node,attr)
            else:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data


def build_ENZYMES_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'ENZYMES_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'ENZYMES_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'ENZYMES_node_attributes.txt')
    adjency=compute_adjency(path,'ENZYMES_A.txt')
    data_dict=graph_indicator(path,'ENZYMES_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_BZR_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'BZR_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'BZR_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'BZR_node_attributes.txt')
    adjency=compute_adjency(path,'BZR_A.txt')
    data_dict=graph_indicator(path,'BZR_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data

def build_COX2_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'COX2_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'COX2_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'COX2_node_attributes.txt')
    adjency=compute_adjency(path,'COX2_A.txt')
    data_dict=graph_indicator(path,'COX2_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            if not use_node_deg:
                g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        data.append((g,i[1]))

    return data
    
def build_SYNTHETIC_dataset(path,type_attr='label'):
    graphs=graph_label_list(path,'SYNTHETIC_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'SYNTHETIC_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'SYNTHETIC_node_attributes.txt')
    adjency=compute_adjency(path,'SYNTHETIC_A.txt')
    data_dict=graph_indicator(path,'SYNTHETIC_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data
    
def build_AIDS_dataset(path,type_attr='label'):
    graphs=graph_label_list(path,'AIDS_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'AIDS_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'AIDS_node_attributes.txt')
    adjency=compute_adjency(path,'AIDS_A.txt')
    data_dict=graph_indicator(path,'AIDS_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data
    
def build_Cuneiform_dataset(path,type_attr='label'):
    graphs=graph_label_list(path,'Cuneiform_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'Cuneiform_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'Cuneiform_node_attributes.txt')
    adjency=compute_adjency(path,'Cuneiform_A.txt')
    data_dict=graph_indicator(path,'Cuneiform_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data
    
def build_LETTER_dataset(path,type_attr='label',name='med'):
    graphs=graph_label_list(path,'Letter-'+name+'_graph_labels.txt')
    if type_attr=='label':
        node_dic=node_labels_dic(path,'Letter-'+name+'_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        node_dic=node_attr_dic(path,'Letter-'+name+'_node_attributes.txt')
    adjency=compute_adjency(path,'Letter-'+name+'_A.txt')
    data_dict=graph_indicator(path,'Letter-'+name+'_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data

def build_Fingerprint_dataset(path,type_attr='real'):
    graphs=graph_label_list(path,'Fingerprint_graph_labels.txt')
    node_dic=node_attr_dic(path,'Fingerprint_node_attributes.txt')
    adjency=compute_adjency(path,'Fingerprint_A.txt')
    data_dict=graph_indicator(path,'Fingerprint_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=Graph()
        for node in data_dict[i[0]]:
            g.name=i[0]
            g.add_vertex(node)
            g.add_one_attribute(node,node_dic[node])
            for node2 in adjency[node]:
                g.add_edge((node,node2))
        data.append((g,i[1]))

    return data
