import torch

train_num_epoch = 1

device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
embedding_dim = 300
hidden_size = 100
num_classes = 4
dropout_rate = 0.3
learning_rate = 0.001
weight_decay = 1e-4
pad_size = 200

word_embedding_filename = 'sgns.context.word-character.char1-1.bz2'
train_filepath = ''
test_filepath = ''