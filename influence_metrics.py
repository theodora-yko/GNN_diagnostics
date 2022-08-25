from models import *
from train_utils import *
from uncertainty_metrics import *

import networkx as nx, torch_geometric.utils as F


def diff_epoch_pipelines(model, dataset, data, epoch_sizes=[10, 50, 100], find_common_nodes=True, display=True,
                         indicate=True):
    """
    arguments:
        - dataset
        - data
        - epoch sizes: an array of different epoch sizes
    returns: a dictionary of list of influential nodes & output, accuracy level at each epoch level
    """
    result = {}

    for size in epoch_sizes:
        print(f"---- Current test size: {size} ----")
        new_outputs, _, _, new_accuracies = check_pipeline(model, dataset, data, data.train_mask,
                                                           data.test_mask,
                                                           n_epochs=size,
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
        result[size] = {'influential_nodes': influential_nodes, 'output': output_per_node}

    if indicate:
        for size, item in result.items():
            print(f"Influential nodes found: {result[size]['influential_nodes']}")

    node_list = []
    if find_common_nodes:
        node_list += list(result[1]['influential_nodes'].keys())
        node_list = set(node_list)
        for size in result.keys():
            node_list = node_list.intersection(result[size]['influential_nodes'])

        if len(node_list) != 0:
            print()
            print('=================================')
            print(f"common nodes: {node_list}")
        else:
            print()
            print('=================================')
            print("no common influential nodes found")

    return result, node_list


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
    sim = nx.simrank_similarity(G)
    similarities_map = np.array([[sim[u][v] for v in influential_nodes] for u in influential_nodes])
    print(similarities_map)
    return similarities_map


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


