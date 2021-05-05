from model import *
from utils import *
from config import *

def train_func():
    train_loss = 0
    train_F1 = 0
    train_precision = 0
    train_recall = 0
    times = 0
    for item in tqdm.tqdm(train_loader, desc='train', total=len(train_loader) - 1):
        times += 1
        if len(item[0]) < batch_size:
            break
        sentence, tags, length_list = item[0].to(device), item[1].to(device), item[2]
        optimizer.zero_grad()
        tag_seq_list, loss = model(sentence, tags, length_list)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        evaluation = evaluate(tag_seq_list, tags, length_list)
        train_F1 += evaluation[0]
        train_precision += evaluation[1]
        train_recall += evaluation[2]
    
    return train_loss / (times * batch_size), train_F1 / times, train_precision / times, train_recall / times

print('Loading word embeddings...')
vocab_model = KeyedVectors.load_word2vec_format(word_embedding_filename, binary = False, encoding = "utf-8", unicode_errors = "ignore", limit = None)

idx = 2
i = 0
for key in vocab_model.vocab:
	if len(key) == 1:
		words2idx[key] = idx
		idx2words[idx] = key
		words_list.append(key)
		idx += 1
		word_vectors.append(vocab_model.vectors[i])
	i += 1

words2idx['<pad>'] = PAD_IDX
words2idx['<unk>'] = UNK_IDX
idx2words[PAD_IDX] = '<pad>'
idx2words[UNK_IDX] = '<unk>'
word_embeddings = np.concatenate((pad_vec, unk_vec, np.array(word_vectors)), axis = 0)
vocab_size = len(word_embeddings)

print('Word embeddings loaded succesfully!')
print('-----------------------------------------')
print('Loading data...')
train= pd.read_csv(train_filepath+'msr_training.utf8', header = None)
train.columns = ['original']
train = data_cleaning(train)

batch_size = 32
train_dataset = MSRSEG(train)
train_loader = Data.DataLoader(dataset=train_dataset,
							   batch_size=batch_size,
							   collate_fn=collate_fn,
							   shuffle=False)

print('Data loaded suceesfully!')
print('-----------------------------------------')
print('Training started...')

model = biLSTM_CNN_CRF(vocab_size, embedding_dim, hidden_size, tag2idx, dropout_rate, word_embeddings).to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_result = []
file_path = 'result'
file_name = 'train_result.txt'
model_name = 'model.pkl'
f = open(file_path+'/'+file_name, 'w')
f.close()

for i in range(train_num_epoch):
	train_result.append(train_func())
	print("epochs: ", i)
	print("train: -loss: %.4f  -F1: %.4f  -precision: %.4f  -recall: %.4f" %train_result[-1])
	write_result(i, train_result, file_path+'/'+file_name)
	torch.save(model, file_path+'/'+model_name)
	
print('Training result saved in ', file_path+'/'+file_name)
print('Model saved in ', file_path+'/'+model_name)
