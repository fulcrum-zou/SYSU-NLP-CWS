from model import *
from utils import *
from config import *

def test_func():
    test_loss = 0
    test_F1 = 0
    test_precision = 0
    test_recall = 0
    times = 0
    for item in tqdm.tqdm(test_loader, desc='test', total=len(test_loader)):
        times += 1
        sentence, tags, length_list = item[0].to(device), item[1].to(device), item[2]
        with torch.no_grad():
            tag_seq_list, loss = model(sentence, tags, length_list)
            test_loss += loss.item()
            evaluation = evaluate(tag_seq_list, tags, length_list)
            test_F1 += evaluation[0]
            test_precision += evaluation[1]
            test_recall += evaluation[2]
            test_predict_gold.extend(evaluation[3])
    
    return test_loss / (times * test_batch_size), test_F1 / times, test_precision / times, test_recall / times

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
test = pd.read_csv(test_filepath+'msr_test.utf8', header = None)
test.columns = ['sentence']
test_gold = pd.read_csv(test_filepath+'msr_test_gold.utf8', header = None)
test.insert(0, 'original', test_gold)
test = data_cleaning(test, is_test = True)

test_batch_size = 1
test_dataset = MSRSEG(test)
test_loader = Data.DataLoader(dataset=test_dataset,
							  batch_size=test_batch_size,
							  collate_fn=collate_fn,
							  shuffle=False)

print('Data loaded suceesfully!')
print('-----------------------------------------')
print('Testing started...')


test_result = []
test_predict_gold = []

file_path = 'result'
file_name = 'test_result.txt'
model_name = 'model.pkl'
f = open(file_path+'/'+file_name, 'w')
f.close()

model = torch.load(file_path+'/'+model_name)

test_result.append(test_func())
print("test: -loss: %.4f  -F1: %.4f  -precision: %.4f  -recall: %.4f" %test_result[-1])
write_result(i, test_result, file_path+'/'+file_name)
torch.save(model, file_path+'/'+model_name)
	
print('Test result saved in ', file_path+'/'+file_name)

test['predict'] = [tag_to_words(test['sentence'].iloc[i], test_predict_gold[i]) for i in range(len(test))]
test['predict'].to_csv(file_path+'/'+'output.utf8', encoding='utf-8', header = None, index = False)
