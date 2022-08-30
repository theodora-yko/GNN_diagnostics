from models import *
from train_utils import *
from uncertainty_metrics import *

import networkx as nx, torch_geometric.utils as F


def repeat_pipelines(model, dataset, data, epoch_size=100, repeat_loo = 3, find_common_nodes=True, indicate=True):
    """
    arguments:
        - dataset: must be already trained in order to call check_pipeline
        - data
        - epoch_size: 
        - repeat_loo:
        - indicate: if TRUE, indicates whenever an influential node is found
    returns: 
        - result: a dictionary of list of influential nodes & output, accuracy level at each epoch level
        - node: a list of influential nodes that appeared in all loo_pipelines 
    """
    result = {}

    for test_num in range(1, repeat_loo+1):
        print(f"---- test number {test_num} ----")
        new_outputs, _, _, new_accuracies = check_pipeline(model, dataset, data, data.train_mask,
                                                           data.test_mask,
                                                           n_epochs=epoch_size,
                                                           original_output=None,
                                                           indicate=False, \
                                                           return_prediction=True,
                                                           compute_y_differences=True,
                                                           dimension=32,
                                                           task='classification',
                                                           loss_function=torch.nn.CrossEntropyLoss(),
                                                           lr=0.001)
        influential_nodes = {}
        output_per_node = {}
        for node, output in new_outputs.items():
            output_per_node[node.item()] = output
        for node, accuracy in new_accuracies.items():
            influential_nodes[node.item()] = accuracy
        result[test_num] = {'influential_nodes': influential_nodes, 'output': output_per_node}
        num_nodes = len(result[test_num])
        
        if indicate:
            print(f"Influential nodes found for test number {test_num}: {result[test_num]['influential_nodes']}")

    return result

def find_common_nodes(result): 
    first_key = list(result.keys())[0]
    node_list = []
    node_list += list(result[first_key]['influential_nodes'].keys())
    node_list = set(node_list)
    for test_num in result.keys():
        node_list = node_list.intersection(result[test_num]['influential_nodes'])

    if len(node_list) != 0:
        print()
        print('=================================')
        print(f"common nodes: {node_list}")
        return node_list
  
    else:
        print()
        print('=================================')
        print("no common influential nodes found")
        return None 
    

# def display_influential_nodes(result): 
#     epoch_levels = list(result.keys())
#     for epoch_level in epoch_levels: 
#         num_nodes = len(result[epoch_level]['influential_nodes'])
#         print(f"For epoch size of {epoch_level}, {num_nodes} influential nodes found")
#         print(f"  - influential nodes: {list(result[epoch_level]['influential_nodes'].keys())}")
#         print(f"  - final accuracy: {list(result[epoch_level]['influential_nodes'].values())[-1]}")

def compute_centrality(dataset, influential_nodes, task='degree_centrality'):
    """
    arguments:
        - dataset
        - influential_nodes: list of influential nodes
        - task: instruction for which type of centrality to compute
    returns: a dictionary of the centrality of given influential nodes
    """
    graph = F.to_networkx(dataset[0])  # process the graph to networkX type
    if task == 'degree_centrality':
        dictionary = nx.degree_centrality(G=graph)
    elif task == 'betweenness_centrality':
        dictionary = nx.betweenness_centrality(G=graph)  # calculate betweenness, returns a dictionary
    
    # if the nodes in the training set chosen based on some sort of centrality? 
    # try to plot all the influences in the node 
    result = {}
    for i in influential_nodes:
        result[i] = dictionary[i]

    print(f"search result: \n{result}")
    return result

def node_similarity(dataset, influential_nodes):
    """
    arguments:
        - dataset
        - influential_nodes: list of influential nodes
    returns: a dictionary of node centralities of given influential nodes
    """
    graph = F.to_networkx(dataset[0])  # process the graph to networkX type
    sim = nx.simrank_similarity(graph)
    similarities_map = np.array([[sim[u][v] for v in influential_nodes] for u in influential_nodes])
    similarities_PD = pd.DataFrame(data = similarities_map)
    print(similarities_PD)
    return similarities_map

def find_common_neighbours(dataset, influential_nodes): 
    graph = pygUtil.to_networkx(dataset[0])
    nx.k_nearest_neighbors(G=graph, source='in+out', target='in+out', nodes=None, weight=None)


def accessibility(dataset, influential_nodes):
    """
    input arguments:
        - dataset
        - influential_nodes: list of influential nodoes

    output: returns a dictionary of the betweeness centrality of the influential nodes
    """
    pass


import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_distribution(influential_class, heavily_affected_nodes):
    """
    function: plot the distribution
    """

    fig, ax_improvements = plt.subplots()
    ax_affected_nodes = ax_improvements.twinx()

    # labels
    ax_improvements.set_xlabel('node index')
    ax_improvements.set_ylabel('accuracy improvements', color='g')
    ax_affected_nodes.set_ylabel('affected nodes', color='b')

    # plot the accuracy improvements
    improvements = list(influential_class.accuracy_improvements.values())
    nodes = list(influential_class.accuracy_improvements.keys())
    ax_improvements.scatter(nodes, improvements, color='g')

    # plot the affected nodes
    node_number = len(list(heavily_affected_nodes.keys()))
    colors = cm.rainbow(np.linspace(0, 1, node_number))
    count = 0
    for node, info in heavily_affected_nodes.items():
        node_padded = [node] * len(info['affected nodes'])
        ax_affected_nodes.scatter(node_padded, info['affected nodes'], color=colors[count])
        count += 1

    plt.show


