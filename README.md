# GNN_diagnostics

## questions
* what is tsne, umap, 
* for check_pipeline, when n_epochs is too low, returns original_accuracy higher than 1?
maybe i should be careful not to run check_pipeline more than once as then the accuracy wil lget higher than 1? 
* can you explain original_output  = model(data.x, data.edge_index) in check_pipeline? 

* KL divergence - Although the KL divergence measures the “distance” between two distributions, 
it is not a distance measure. This is because that the KL divergence is not a metric measure

* am i focusing on node / edges? 
  - print the edges of the influential nodes 
  - print the neighbours of the influential nodes to k degree connection
  
## Background & Documentation
GraphSage - https://theodorayko.blogspot.com/2022/07/implementation-of-graphsage-using.html

## Summary
**note**: For more detailed explanations & conditions of each function/class, check the specific file.
1. **GNN**
- contains the definition of neural network models in use   
- GNN algorithms in use: GraphSAGE
- accuracy: class, used for training & evaluating a ML model
2. **data**:
downloads datasets in need
3. **train_mask**
manually defines the train mask of a given dataset 
- random_train_mask: creates a random training set using PyG's inbuilt function
- create_train_mask: if default_train set False, returns data with train_mask, val_mask, test_mask attributes manually set
- mask_a_node: Given a dataset, mask a node of given index into False (i.e. mask one datapoint in the training set)
4. **preliminary analysis**
- return_accuracy: returns predictions, gradients if comput_gradient set True, accuracy of the model trained using the given data 
- find_the_influential_class: class, self.influential_accuracy returns a list of influential points and gradients
- multiple_testing: repeats finding the influential points multiple times  

## Roadmap
- [x] preliminary GNN model
- [x] manual_masks
- [x] preliminary analysis (1) - identifying the influential points 
- [x] preliminary analysis (2) - compute & compare the gradient of each training set 
- [ ] preliminary analysis (3) - plot the distribution
- [ ] preliminary analysis (4) - spatial relationships between the points 
- [ ] implement influence function from https://arxiv.org/pdf/1703.04730.pdf
- [ ] link manual_masks with influence function and improve accuracy

## Challenges ToT
- reaching the accuracy level of the example I was imitating smh took SO long
- switching between different types of data (ESPECIALLY torch tensors) 
- understanding the structure of a PyG dataset, was very confused initially
- each training takes time to compute, and with a bigger epoch size, it was often the case that I had to wait for several minutes to figure out there is an error in my code. 

