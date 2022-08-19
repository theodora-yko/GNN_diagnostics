import torch

from models import *
from train_utils import *

def loo_pipeline(model, dataset, data, train_mask,
                 test_mask, which_node,
                 n_epochs,
                 criterion, optimizer,
                 original_output=None,
                 indicate=False, \
                 return_prediction=False,
                 compute_y_differences=False,
                 task=task,
                 loss_function=torch.nn.functional.cross_entropy(),
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
    if original_output is None:
        model.eval()
        original_output  = model(data.x, data.edge_index)
    _, original_predictions = torch.max(original_output.detach(),1)
    length = len(data.y[train_mask])
    original_accuracy = (original_predictions[train_mask] == data.y[train_mask].detach()).sum().item()/length
    original_misclassified = (original_predictions[train_mask] != data.y[train_mask]).numpy()

    #### Mask a node, and retrain on the data for each node
    new_model = copy.deepcopy(model)
    new_mask = copy.deepcopy(train_mask)
    new_mask[which_node] = False
    train_acc_list, test_acc_list, _, _, new_predictions = train(n_epochs, new_model,
                                                       criterion=loss_function,
                                                       optimizer=torch.optim.Adam(new_model.parameters(), lr=lr),
                                                       data.x, data.y, data.edge_index,
                                                       m = mask(new_mask, test_mask),
                                                       plotting = False,
                                                       task=task)
    loo_output  = new_model(data.x, data.edge_index)
    _, loo_predictions = torch.max(loo_output.detach(),1)
    length = len(data.y[test_mask])
    loo_accuracy = (loo_predictions[test_mask] == data.y[test_mask].detach()).sum().item()/length
    loo_misclassified = (loo_predictions[test_mask] != data.y[train_mask]).numpy()
    if compute_y_differences:
        if len(loo_output) != len(original_output):
            y_differences = None
            print(HERE)
        else:
            original_scores = torch.nn.functional.softmax(original_output, dim=0).numpy()
            loo_scores = torch.nn.functional.softmax(loo_output, dim=0).numpy()
            kl = sum(rel_entr(original_scores, loo_scores))
            #print(f"kl divergence: {kl}")
            y_differences = F.mse_loss(loo_scores, original_scores)
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
                   n_epochs,
                   criterion, optimizer,
                   original_output=None,
                   indicate=False, \
                   return_prediction=False,
                   compute_y_differences=False,
                   dimension=32,
                   task=task,
                   loss_function=torch.nn.functional.cross_entropy(),
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
    if original_output is None:
        model.eval()
        original_output  = model(data.x, data.edge_index)
    _, original_predictions = torch.max(original_output.detach(),1)
    length = len(data.y[train_mask])
    original_accuracy = (original_predictions[train_mask] == data.y[train_mask].detach()).sum().item()/length
    original_misclassified = (original_predictions[train_mask] != data.y[train_mask]).numpy()
    y_differences = []
    kl = []
    #### Mask a node, and retrain on the data for each node
    for i in np.which(train_mask)[0]:
        loo_output, y_prime, kl_prime, loo_accuracy_prime = loo_pipeline(model, dataset,
                                                                   data, train_mask,
                                                                   test_mask, which_node,
                                                                   n_epochs,
                                                                   criterion, optimizer,
                                                                   original_output=None,
                                                                   compute_y_differences=compute_y_difference,
                                                                   task=task,
                                                                   loss_function=loss_function,
                                                                   lr=lr)
        y_differences += [y_prime]
        kl += [kl_prime]


    return new_output, y_differences, kl, new_accuracy
