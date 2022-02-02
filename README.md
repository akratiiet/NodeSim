#NodeSim
This code is to generate network embedding using NodeSim Method and predict links in the given network. 

main.py code parses the arguments, reads the input graph and generates the NodeSim embedding. The input network, each node should have its community label.
Input Network: Input/sample.gpickle 
Output Embedding: Output/sample.emb

link_prediction.py method trains a logistic regression model using NodeSim link prediction method and compute the accuracy. This method reads training and testing files provided in Input folder.

If you use this method, please cite:

@article{saxena2021nodesim,
  title={NodeSim: Node Similarity based Network Embedding for Diverse Link Prediction},
  author={Saxena, Akrati and Fletcher, George and Pechenizkiy, Mykola},
  journal={arXiv preprint arXiv:2102.00785},
  year={2021}
}
