import torch
import copy
from scipy.special import rel_entr
from train_utils import *


def loo_pipeline(model, dataset, data, train_mask,
                 test_mask, which_node,
                 indicate=False,
                 n_epochs=200,
                 original_output=None,
                 compute_y_differences=False,
                 task='classfication',
                 loss_function=torch.nn.CrossEntropyLoss(),
                 lr=0.001):
    """
    #loo - Leave One Out
    model should be the trained model/
    original_output = output using the original data, optional
    indicate = Boolean
    compute_y_differences = Boolean

    returns: prediction by the given data,
            y_differences (y - y_hat) if compu
            te_y_differences set True,
            accuracy of the model trained using the given data
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if original_output is None:
        model.eval()
        original_output = model(data.x, data.edge_index)
    _, original_predictions = torch.max(original_output.detach(), 1)
    length = len(data.y[train_mask])
    original_accuracy = (original_predictions[test_mask] == data.y[test_mask].detach()).sum().item() / length
    original_misclassified = (original_predictions[test_mask] != data.y[test_mask]).numpy()

    #### Mask a node, and retrain on the data for each node
    new_model = copy.deepcopy(model)
    optimizer = torch.optim.Adam(new_model.parameters(), lr=lr, weight_decay=5e-4)
    new_mask = copy.deepcopy(train_mask)
    new_mask[which_node] = False
    if indicate: print(f"--Masking node {which_node}")
    train_acc_list, test_acc_list, loss_list, misclassified, predictions = train(n_epochs, new_model, loss_function,
                                                                                 optimizer,
                                                                                 indicate=indicate, x=data.x,
                                                                                 edge_index=data.edge_index, y=data.y,
                                                                                 m=mask(new_mask, test_mask),
                                                                                 scatter_size=30, plotting=False)
    loo_output = new_model(data.x, data.edge_index)
    _, loo_predictions = torch.max(loo_output.detach(), 1)
    length = len(data.y[test_mask])
    loo_accuracy = (loo_predictions[test_mask] == data.y[test_mask].detach()).sum().item() / length
    loo_misclassified = (loo_predictions[test_mask] != data.y[test_mask]).numpy()
    if compute_y_differences:
        if len(loo_output) != len(original_output):
            y_differences = None
            print(HERE)
        else:
            original_scores = torch.nn.functional.softmax(original_output, dim=0).detach().numpy()
            loo_scores = torch.nn.functional.softmax(loo_output, dim=0).detach().numpy()
            kl = np.mean(rel_entr(original_scores, loo_scores), 1)
            y_differences = np.mean(np.linalg.norm(loo_scores - original_scores))
            # softmax(original_output) - softmax(new_output) -> difference of the score
        # compare two differences / nonnegative version
        # KL divergence, take MSE loss

        # deeplearning & nonlinear model - embedding solution is unique
        # score is more identifiable, as sums up to 1, comparable
        # way the model train is end up in diff local minimum
        # for x,y,z, to be unique, needs another constraint for ex alpha = 0
    else:
        y_differences = None
        kl = None

    return loo_output, y_differences, kl, loo_accuracy


def check_pipeline(model, dataset, data, train_mask,
                   test_mask,
                   n_epochs=200,
                   original_output=None,
                   indicate=False, \
                   return_prediction=False,
                   compute_y_differences=False,
                   dimension=32,
                   task='classification',
                   loss_function=torch.nn.CrossEntropyLoss(),
                   lr=0.001):
    """
    model should be the trained model/
    original_output = output using the original data, optional
    indicate = Boolean
    compute_y_differences = Boolean

    returns: prediction by the given data,
            y_differences (y - y_hat) if compute_y_differences set True,
            accuracy of the model trained using the given data
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    new_output = {}
    if original_output is None:
        if indicate: print(f"calculating the initial accuracy")
        model.eval()
        original_output = model(data.x, data.edge_index)
    _, original_predictions = torch.max(original_output.detach(), 1)
    length = len(data.y[test_mask])
    original_accuracy = (original_predictions[test_mask] == data.y[test_mask].detach()).sum().item() / length
    if indicate: print(f"original accuracy: {original_accuracy}")

    original_misclassified = (original_predictions[test_mask] != data.y[test_mask]).numpy()
    y_differences = {}
    improved_accuracy = {}
    kl = {}
    #### Mask a node, and retrain on the data for each node
    for i in torch.where(data.train_mask)[0]:
        loo_output, y_prime, kl_prime, loo_accuracy_prime = loo_pipeline(model, dataset, data, train_mask,
                                                                         test_mask, indicate=indicate, which_node=i,
                                                                         n_epochs=n_epochs,
                                                                         original_output=original_output,
                                                                         compute_y_differences=compute_y_differences,
                                                                         task=task,
                                                                         loss_function=loss_function,
                                                                         lr=lr)
        if loo_accuracy_prime >= original_accuracy:
            original_accuracy = loo_accuracy_prime
            new_output[i] = loo_output
            y_differences[i] = y_prime
            kl[i] = kl_prime
            improved_accuracy[i] = loo_accuracy_prime
            if indicate: print(f"masking node {i} improves accuracy to {loo_accuracy_prime}, kl score: {kl_prime}")

    print()
    return new_output, y_differences, kl, improved_accuracy
