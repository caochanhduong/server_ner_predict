# -*- coding: utf-8 -*-
#import fastai
#from fastai.text import *
#from fastai.lm_rnn import *
#from sebastian.eval import eval_ner
# from app.fastai_seq_utilities import *
from fastai_seq_utilities import *
import torch
# import app.utilities_tgdd as utilities
import utilities_tgdd as utilities
from fastai.text import *
import torch
import pickle
from pathlib import Path
import torch.optim as optim
from torch.autograd import Variable
import pickle
import collections
from functools import partial
# from app.util import PreprocessClass
from util import PreprocessClass
import rest_api_pb2

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

print(torch.__version__)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
# # id2token = pickle.load(open('itos.pkl', 'rb'))
# # id2label = ['_pad_', '_xbos_', '_xfld_', '_1_', 'O', 'B-feature', 'I-feature', 'B-product', 'I-product', 'B-app', 'I-app', '<START>', '<STOP>']
# # Define constant
# START_TAG = "<START>"
# STOP_TAG = "<STOP>"
# LM_PATH = "models/lm"
# PRINT_FLAG = False

# dir_path = Path(LM_PATH)
# id2token = pickle.load(open('itos.pkl', 'rb'))
# token2id = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(id2token)})

# id2label = ['_pad_', '_xbos_', '_xfld_', '_1_', 'O', 'B-feature', 'I-feature', 'B-product', 'I-product', 'B-app', 'I-app', '<START>', '<STOP>']
# tag_to_idx = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(id2label)})
# tagset_size = len(id2label)
# class NERTAG(metaclass=Singleton):
        
#     # Define constant
#     START_TAG = "<START>"
#     STOP_TAG = "<STOP>"
#     LM_PATH = "models/lm"
#     PRINT_FLAG = False
    
#     dir_path = Path(LM_PATH)
#     id2token = pickle.load(open('itos.pkl', 'rb'))
#     token2id = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(id2token)})
    
#     id2label = ['_pad_', '_xbos_', '_xfld_', '_1_', 'O', 'B-feature', 'I-feature', 'B-product', 'I-product', 'B-app', 'I-app', '<START>', '<STOP>']
#     tag_to_idx = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(id2label)})
#     tagset_size = len(id2label)
#     def __init__(self):
        
#         self.model = load_model_seq(self.LM_PATH, \
#       lm_id='', train_file_id='', tag_to_idx=self.tag_to_idx, tagset_size=self.tagset_size, clas_id=None, bs=64, cl=1, backwards=False, bidir=False, \
#       startat=0, unfreeze=True, lr=0.01, dropmult=1, pretrain=True, bpe=False, \
#       use_clr=False, use_regular_schedule=False, use_discriminative=True, last=False, \
#       chain_thaw=True, from_scratch=False, freeze_word2vec=False, \
#       n_cycle=4, cycle_len=1, cycle_mult=2, linear_decoder_dp=0.2, \
#       id2token=self.id2token, id2label=self.id2label, classifier_filename="clas_seqforward.h5")
        
#     def predict_raw(self, raw):
#         return self._predict_single(raw)
        
#     def predict(self, mentions):
#         if not isinstance(mentions, list):
#             mentions = [mentions]
#         input_text = []
#         results = []
#         for doc in mentions:
#             input_text.append(doc.content)

#         result_predicted = list(map(lambda x: self._predict_single(x), input_text))

#         for index, result_value in enumerate(result_predicted): 
#             result = rest_api_pb2.PredictResult()
#             result.id = mentions[index].id
#             result.tags.extend(result_value)
#             results.append(result)
          
#         return results
#     def _predict_single(self, mention):
        
#         result = []
#         scores, processed_text = predict_text(self.token2id, self.model, mention)

#         predicted = scores
        
#         # Print result
#         for word, tag in zip(processed_text, predicted):
#             result.append(self.id2label[tag])
#         return result
    
# def load_model_seq(dir_path="", lm_id='', train_file_id='', tag_to_idx='', tagset_size='', clas_id=None, bs=64, cl=1, \
#               backwards=False, bidir=False, startat=0, unfreeze=True, lr=0.01, dropmult=1.0, \
#               pretrain=True, bpe=False, use_clr=True, use_regular_schedule=False, \
#               use_discriminative=True, last=False, chain_thaw=False, from_scratch=False, \
#               freeze_word2vec=False, n_cycle=3, cycle_len=1, cycle_mult=2, linear_decoder_dp=0.1, \
#               id2token={}, id2label={}, classifier_filename="clas_seqforward.h5"):
     
#     PRE = 'bwd_' if backwards else ''
#     PRE = 'bpe_' + PRE if bpe else PRE

#     train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'
#     dir_path = Path(dir_path)
#     # lm_id = lm_id if lm_id == '' else f'{lm_id}_'
#     # clas_id = lm_id if clas_id is None else clas_id
#     # clas_id = clas_id if clas_id == '' else f'{clas_id}_'

#     # lm_file = f'{PRE}{lm_id}fwd_lm_tgdd_lm_enc'

#     # lm_path = dir_path / f'{lm_file}.h5'
#     # assert lm_path.exists(), f'Error: {lm_path} does not exist.'
    
#     """hyperaparameter settings"""
#     bptt,em_sz,nh,nl = 70,400,1150,3
#     dps = np.array([0.4,0.5,0.05,0.3,0.4])*dropmult
#     opt_fn = partial(optim.Adam, betas=(0.8, 0.99))
    
#     """load datasets"""
#     c = len(id2label)

#     if bpe:
#         vs=30002
#     else:
#         vs = len(id2token)
    
#     m = get_rnn_seq_tagger(bptt, 70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
#                   layers=[em_sz, 50, c], drops=[dps[4], 0.1],
#                   dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3], \
#                   linear_decoder_dp=linear_decoder_dp, tag_to_idx=tag_to_idx, tagset_size=tagset_size)
#     m.load_state_dict(torch.load('clas_seqforward.h5', map_location=lambda storage, loc: storage))
#     m.reset()
#     m.eval()
    
#     return m

# def predict_text(stoi, model, text):
#     prefix_str = ['\nxbos', 'xfld', '1']

#     texts = [text]

# #    tok = list(map(lambda x: x.split(), texts))
#     tok = [PreprocessClass.convert(s).split() for s in texts]

#     preprocessed = tok[0]
#     preprocessed = [utilities.map_number_and_punct(p.lower()) for p in tok[0]]
#     preprocessed = prefix_str + preprocessed
    
    
#     # turn into integers for each word
#     encoded = [stoi[p] for p in preprocessed]

#     ary = np.reshape(np.array(encoded),(-1,1))

#     # turn this array into a tensor
#     tensor = torch.from_numpy(ary)

#     # wrap in a torch Variable
#     variable = Variable(tensor)

#     # do the predictions
#     predictions = model((variable, []))

#     return predictions[1], preprocessed

# def predict_processed(model, encoded):
#     ary = np.reshape(np.array(encoded),(-1,1))

#     tensor = torch.from_numpy(ary)

#     variable = Variable(tensor)

#     # do the predictions
#     predictions = model((variable, []))

#     return predictions[1]

# model = load_model_seq(\
#      lm_id='', train_file_id='', clas_id=None, bs=64, cl=1, backwards=False, bidir=False, \
#      startat=0, unfreeze=True, lr=0.01, dropmult=1, pretrain=True, bpe=False, \
#      use_clr=False, use_regular_schedule=False, use_discriminative=True, last=False, \
#      chain_thaw=True, from_scratch=False, freeze_word2vec=False, \
#      n_cycle=4, cycle_len=1, cycle_mult=2, linear_decoder_dp=0.2, \
#      id2token=id2token, id2label=id2label,tag_to_idx=tag_to_idx, classifier_filename="clas_seqforward.h5",tagset_size = len(id2label))

# text = "ip6 với samsung galaxy s7 cái nào nhanh hơn"

# ner_model = NERTAG()
# print(ner_model.predict_raw(text))


#@title
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class Prep:
    
    def __init__(self, tweet, label, vocab):
        self.tweet = tweet
        self.label = label
        self.vocab = vocab
    
    # cleaning function
    def cleaning(x):
        re1 = re.compile(r'  +')
        x = x.replace('#','').replace('&amp;', '&')
        #return re1.sub(' ', html.unescape(x))
        return re1.sub(' ', re.sub('https?://[A-Za-z0-9./]+', '',html.unescape(x)))
    
    # tokenizer function
    def get_texts(df, n_lbls=1):
        BOS = 'xbos'
        labels = np.unique(df.iloc[:,range(n_lbls)].values, return_inverse=True)[n_lbls]
        texts = f'\n{BOS} ' + df[n_lbls].astype(str)
        for i in range(n_lbls+1, len(df.columns)): 
            texts += df[i].astype(str)
        texts = texts.apply(Prep.cleaning).values.astype(str)
        tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
        return tok, list(labels)
        
    # iterator function
    def get_all(df, n_lbls):
        tok, labels = [], []
        for i, r in enumerate(df):
            print(i)
            tok_, labels_ = Prep.get_texts(r, n_lbls)
            tok += tok_;
            labels += labels_
        return tok, labels
    
    # function to automatically tokenize single tweet
    def tokenize(tweet, vocab, label = '0', chunksize = 1,
                 folder_name = 'Test_lm', file_name = 'text'): 
        
        text = np.array(pd.Series(tweet))
        labels = np.array(pd.Series(label))

        colNames = ['labels','text']
        textdf = pd.DataFrame({'text':text, 'labels':labels}, columns = colNames)
    
        textdf.to_csv(folder_name+'/'+file_name+'.csv', header=False, index=False)
    
        BOS = 'xbos'
        textdf1 = pd.read_csv(folder_name+'/'+file_name+'.csv', header=None, chunksize=chunksize)
        TextLm = Prep.get_all(textdf1, 1)
        TextLm = (TextLm[0])
        
        for i in enumerate(TextLm[0]):
            if TextLm[0][i[0]] not in vocab.keys():
                TextLm[0][i[0]] = '_unk_'
        
        tok = [[vocab[o] for o in p] for p in TextLm]
        tok = tok[0]
    
        output = {
            "Tokens": TextLm[0],
            "Encoded_Tokens": tok
        }
        
        return output
    
    # 
    def OneHot(sequences, dimension):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results


class LSTM:
    
    def __init__(self, input, wgts, nh):
        
        self.input = input
        self.wgts = wgts
        self.nh = nh
        self.em_sz = []
        self.hidden_state_l0 = []
        self.hidden_state_l1 = []
        self.hidden_state_l2 = []
        self.cell_state_l0 = []
        self.cell_state_l1 = [] 
        self.cell_state_l2 = [] 
        
    def setInput(self, input):
      self.input = input
    # single lstm layer
    def single(input, wgts, nh, stage = '0'):
        
        # input weights and bias from the loaded torch model, converted into numpy variables
        wii = wgts['rnns.'+stage+'.module.weight_ih_l0'][:nh]
        
        wif =wgts['rnns.'+stage+'.module.weight_ih_l0'][nh:2*nh]
        wig =wgts['rnns.'+stage+'.module.weight_ih_l0'][2*nh:3*nh]
        wio =wgts['rnns.'+stage+'.module.weight_ih_l0'][3*nh:4*nh]
    
        bii =wgts['rnns.'+stage+'.module.bias_ih_l0'][:nh]
        bif =wgts['rnns.'+stage+'.module.bias_ih_l0'][nh:2*nh]
        big =wgts['rnns.'+stage+'.module.bias_ih_l0'][2*nh:3*nh]
        bio =wgts['rnns.'+stage+'.module.bias_ih_l0'][3*nh:4*nh]
        
        # output weights and bias from the loaded torch model, converted into numpy variables
        whi =wgts['rnns.'+stage+'.module.weight_hh_l0'][:nh]
        whf =wgts['rnns.'+stage+'.module.weight_hh_l0'][nh:2*nh]
        whg =wgts['rnns.'+stage+'.module.weight_hh_l0'][2*nh:3*nh]
        who =wgts['rnns.'+stage+'.module.weight_hh_l0'][3*nh:4*nh]
    
        bhi =wgts['rnns.'+stage+'.module.bias_hh_l0'][:nh]
        bhf =wgts['rnns.'+stage+'.module.bias_hh_l0'][nh:2*nh]
        bhg =wgts['rnns.'+stage+'.module.bias_hh_l0'][2*nh:3*nh]
        bho =wgts['rnns.'+stage+'.module.bias_hh_l0'][3*nh:4*nh]
    
        hs, cs = torch.from_numpy(np.zeros(nh)).float(), torch.from_numpy(np.zeros(nh)).float()
        hidden_matrix = np.empty((0,nh))
        cell_matrix = np.empty((0,nh))
        
        ## LSTM Process:
        # vectors for ignore gate, forget gate, cell gate, output gate, cell state and hidden state
        # are calculated and updated per loop for every word vector. Cell states and hidden states 
        # are all stored
        # for t,v in enumerate(input):
        #     # print(type(np.matmul(hs,whi)))
        #     ig = sigmoid(np.matmul(wii,input[t]) + bii + np.matmul(hs,whi) + bhi)
        #     fg = sigmoid(np.matmul(wif,input[t]) + bif + np.matmul(hs,whf) + bhf)
            
        #     cg = np.tanh(np.matmul(wig,input[t]) + big + np.matmul(hs,whg) + bhg)
        #     og = sigmoid(np.matmul(wio,input[t]) + bio + np.matmul(hs,who) + bho)
        #     cs = np.multiply(fg,cs) + np.multiply(ig,cg)
        #     hs = np.multiply(og,np.tanh(cs))
        #     # print(hidden_matrix.shape)
        #     # print(hs.numpy().shape)
        #     hidden_matrix = np.append(hidden_matrix, hs.numpy().reshape(1,-1), axis=0)
        #     cell_matrix = np.append(cell_matrix, cs.numpy().reshape(1,-1), axis=0)
        
        for t,v in enumerate(input):
            # print(type(np.matmul(hs,whi)))
            # igd = torch.matmul(wii,input[t])+ bii + torch.matmul(hs,whi)
            ig = sigmoid(torch.matmul(wii,input[t]) + bii + torch.matmul(hs,whi) + bhi)
            fg = sigmoid(torch.matmul(wif,input[t]) + bif + torch.matmul(hs,whf) + bhf)
            cg = np.tanh(torch.matmul(wig,input[t]) + big + torch.matmul(hs,whg) + bhg)
            og = sigmoid(torch.matmul(wio,input[t]) + bio + torch.matmul(hs,who) + bho)
            cs = (fg * cs) + (ig * cg)
            hs = (og * np.tanh(cs))
            # print(hidden_matrix.shape)
            # print(hs.numpy().shape)
            hidden_matrix = np.append(hidden_matrix, hs.cpu().numpy().reshape(1,-1), axis=0)
            
            cell_matrix = np.append(cell_matrix, cs.cpu().numpy().reshape(1,-1), axis=0)

        # hidden_state = np.array(hidden_matrix)
        hidden_state = torch.from_numpy(hidden_matrix).float()

        cell_state = np.array(cell_matrix)
        return hidden_state, cell_state
    
    # stacked consisting of three layers
    def stacked(self):
        self.em_sz = len(self.input[0])
        # First LSTM Layer, nh = 1150
        hidden_0 = LSTM.single(input = self.input, wgts = self.wgts, nh = self.nh, stage = '0')
        # store hidden states and cell states into class object
        self.hidden_state_l0, self.cell_state_l0 = hidden_0[0], hidden_0[1]
        
        # Second LSTM Layer, nh = 1150
        hidden_1 = LSTM.single(input = hidden_0[0], wgts = self.wgts, nh = self.nh, stage = '1')
        #store hidden states and cell states into class object
        self.hidden_state_l1, self.cell_state_l1 = hidden_1[0], hidden_1[1]
        
        # Third LSTM Layer, nh = 400 = Embedding Size as LSTM Output
        hidden_2 = LSTM.single(input = hidden_1[0], wgts = self.wgts, nh = self.em_sz, stage = '2')
        # store hidden states and cell states into class object
        self.hidden_state_l2, self.cell_state_l2 = hidden_2[0], hidden_2[1]
        
        return self
# activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    relu = np.maximum(0,x)
    return relu
    
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

vocab = torch.load("/home/lap11305/LVTN/code_model_anh_Dang/ner_tagging/app/new_vocab.h5")
itos_wiki=vocab.itos
stoi_wiki=vocab.stoi
print(len(vocab.itos))
list(stoi_wiki.items())[0:10]

def sentence_to_index_vector(input_sentence):
  # input_sentence = re.sub('[\,\_=\+\-\#\@\$\%\$\\.\?\:\(\)\~\!\@\;\'\|\<\>\]\[\"\–“”…*]',' ',input_sentence)
  list_token=input_sentence.split(' ')
  # print(list_token)
  while '' in list_token:
    list_token.remove('')
  # print(list_token)
  return vocab.numericalize(list_token)

lm_wgts = torch.load('/home/lap11305/LVTN/code_model_anh_Dang/ner_tagging/app/model_cpu_add_corpus_vocab_enc.pth', map_location='cpu')
# model = torch.load('/content/drive/Thesis/datav2/intentdb/models/wiki_ulmfit/model_cpu_add_corpus_vocab_enc.pth',map_location='cpu')

enc_weight = lm_wgts['encoder.weight']
# embedding = np.matmul(onehot, lm_wgts['encoder.weight'])
# print(embedding)
# embedding.shape

lstm = LSTM(None, lm_wgts, 1152)
def forward_07_ml(sentence_index_vector):
  # print("begin forward_07_ml")
  onehot = Prep.OneHot(sentence_index_vector, dimension = len(itos_wiki))
#   embedding = torch.matmul(torch.from_numpy(onehot).float(), enc_weight)
  embedding = torch.matmul(torch.from_numpy(onehot).float(), enc_weight)
  lstm.setInput(embedding)
  # st_lstm = LSTM.stacked(lstm)
  result = lstm.stacked().hidden_state_l2
  # print("end forward_07_ml")

  # print(type(st_lstm.hidden_state_l2))
  return result


forward_07_ml(sentence_to_index_vector("đi mùa hè xanh vui không"))


class AWD_CRF(nn.Module):


    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(AWD_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        
        # self.lstm = get_language_model(AWD_LSTM, 27498)
        # self.lstm.eval()
        # # lm.reset()
        # self.lstm.load_state_dict(torch.load(PRE_PATH/"model_cpu_add_new_vocab.pth"))

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        # self.hidden = self.init_hidden()
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        # lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = forward_07_ml(sentence)
        # print(lstm_out.shape)
        # print(len(sentence))
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        # print("----------------------------type lstm_out")
        # print(lstm_out)
        lstm_out=lstm_out.float().cpu()
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


id_to_label = ['pad', 'xbos', 'xfld', '1', 'O', 'B-name_activity', 'I-name_activity', 'B-type_activity', 'I-type_activity', 'B-holder', 'I-holder', 'B-time', 'I-time', 'B-city', 'I-city', 'B-district', 'I-district', 'B-ward', 'I-ward', 'B-name_place', 'I-name_place', 'B-street', 'I-street', 'B-reward', 'I-reward', 'B-contact', 'I-contact', 'B-register', 'I-register', 'B-works', 'I-works', 'B-joiner', 'I-joiner', '<START>', '<STOP>']
tag_to_ix = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(id_to_label)})
ix_to_tag = collections.defaultdict(lambda:0, {k:v for k,v in enumerate(id_to_label)})

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 400
HIDDEN_DIM = 400

word_to_ix=stoi_wiki
class NERTAG(metaclass=Singleton):

    def __init__(self):
        self.model = AWD_CRF(27498, tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
        self.model.load_state_dict(torch.load('/home/lap11305/LVTN/code_model_anh_Dang/ner_tagging/app/ner_137_GPU_400e.pth'))

    def predict(self, mentions):
        self.model.eval()
        if not isinstance(mentions, list):
            mentions = [mentions]
        input_text = []
        results = []
        for doc in mentions:
            input_text.append(doc.content)

        result_predicted=[]
        for x in input_text:
            with torch.no_grad():
                precheck_sent = prepare_sequence(x.split(" "), word_to_ix)
                result_predicted.append([ ix_to_tag[x] for x in self.model(precheck_sent)[1]])
        # result_predicted = list(map(lambda x: self._predict_single(x), input_text))

        for index, result_value in enumerate(result_predicted): 
            result = rest_api_pb2.PredictResult()
            result.id = mentions[index].id
            result.tags.extend(result_value)
            results.append(result)
          
        return results

# # model.eval() # call before predicting
# def predict_ner_tgdd_lm(input_sentence):
#   # input_sentence = re.sub('[\,\_=\+\-\#\@\$\%\$\\.\?\:\(\)\~\!\@\;\'\|\<\>\]\[\"\–“”…*]',' ',input_sentence)
#   with torch.no_grad():
#     precheck_sent = prepare_sequence(input_sentence.split(" "), word_to_ix)
#     print([ ix_to_tag[x] for x in model(precheck_sent)[1]])


# predict_ner_tgdd_lm("đi mùa hè xanh là làm đường , dạy học phải không")
# ner_model = NERTAG()
# print(ner_model.predict([{"content":"đi mùa hè xanh",]))