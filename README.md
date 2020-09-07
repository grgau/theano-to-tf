# theano-to-tf
Converting an LSTM from Theano framework to Tensorflow

### Execution example (from project main dir):

#### Training (for 846 input codes data)
`sh train-tensorflow.sh`

#### Testing (for 846 input codes data)
`sh test-tensorflow.sh compiled_models/tf-model-846.X`

#### Expected results
Results for Recall@10, Recall@20, and Recall@30
0.454597871699988
0.6272884912766815
0.7070440422617419