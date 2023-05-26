import torch 
import math
import pdb
from tqdm import tqdm
from torch.nn import Linear, Module
from torch.nn import ModuleList, ReLU, Softmax, Parameter, LSTM, Embedding
import numpy as np 
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
class Attention(Module):

    def __init__(self, args):
        super(Attention, self).__init__()
        self.hidden_dim = 128

        self.input_dim = 768
        self.out_dim = 50

        self.k = Linear(self.input_dim, self.hidden_dim, bias = False)
        self.v = Linear(self.input_dim, self.out_dim, bias = False)
        self.q = Linear(self.out_dim, self.hidden_dim, bias = False)
        
        self.softmax = Softmax(dim = -1)
    
    def forward(self, key, query, value, mask):
        '''
        Args:
            key: B x ISL x INPUT_DIM
            query: B x 1 x OUT_DIM
            value: B x ISL x INPUT_DIM
        '''
        #key: B x ISL x HIDDEN_DIM
        #query: B x 1 x HIDDEN_DIM
        #value: B x ISL x OUT_DIM
        key = self.k(key)
        query = self.q(query)
        value = self.v(value)

        #attn_mask: B x ISL
        attn_mask = (key*query).sum(-1)/self.hidden_dim**0.5
        attn_mask = attn_mask + mask

        #attention weight
        attn_mask = self.softmax(attn_mask).unsqueeze(-1)

        #final output
        #output: B x 1 x OUT_DIM
        output = (value*attn_mask).sum(1, keepdim = True)
        return output

class DecoderEmbedding(Module):

    def __init__(self, args):
        super(DecoderEmbedding, self).__init__()
        self.args = args
        self.vocab_dict = dict()
        
        #get the pretrained embeddings
        self.embeddings = self.load_embeddings()
        self.embeddings = Embedding.from_pretrained(self.embeddings, padding_idx=0, freeze = False)


    def load_embeddings(self):
        file = os.path.join(self.args.data_root, 'decoder_vocab.txt')
        file = open(file, 'r')
        
        #pad token
        embeddings = [[0.0 for i in range(50)]]
        
        #start of sentence token
        embeddings.append([1 - 2*np.random.rand() for x in range(50)])
        self.vocab_dict['<SOS>'] = 1
        
        index = 2
        for line in tqdm(file):
            line = line.strip()
            self.vocab_dict[line] = index
            index += 1
            embeddings.append([1 - 2*np.random.rand() for x in range(50)])
        
        #end of sent token
        embeddings.append([1 - 2*np.random.rand() for x in range(50)])
        self.vocab_dict['<EOS>'] = index
        self.end_token = index

        #unk token
        index += 1
        self.vocab_dict['<UNK>'] = index
        embeddings.append([1 - 2*np.random.rand() for x in range(50)])
        
        #the vocab inv list
        self.vocab_inv = ['' for i in range(len(self.vocab_dict) + 1)]
        for w in self.vocab_dict:
            self.vocab_inv[self.vocab_dict[w]] = w
        
        return torch.tensor(embeddings)
    
    def vocab(self,word):
        if(word in self.vocab_dict):
            return self.vocab_dict[word]
        return len(self.vocab_dict)-1

    def init_with_glove(self, glove):
        for i in tqdm(self.vocab_dict):
            if(i in glove.vocab_dict):
                self.embeddings[self.vocab_dict[i]] = glove.embeddings.weight[glove.vocab_dict[i]]
        self.embeddings = Embedding.from_pretrained(self.embeddings, padding_idx=0, freeze = False)

    def forward(self, indices):
        '''
        indices: B x SEQ_LEN
        embeddings: B  x SEQ_LEN x 50
        '''
        output = self.embeddings(indices)
        return output

class Encoder(Module):

    def __init__(self, args):
        super(Encoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained("bert-base-cased")

    def forward(self, input):
        output = self.tokenizer(input, return_tensors = 'pt', padding = True, truncation=True)
        for e in output:
            output[e] = output[e].cuda()
        
        #generate masks
        mask = (output['input_ids'] != 0).float()
        mask = 1-mask
        mask[mask == 1] = float('-inf')

        #get the output
        output = self.model(**output)

        return mask, output['last_hidden_state']
        
class Decoder(Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        
        #meta data
        self.args = args
        self.hidden_dim = self.args.decoder.hidden_dim
        self.layers = self.args.decoder.layers
        self.k = self.args.decoder.k
        
        #the embdding layer
        self.embeddings = DecoderEmbedding(args)
        
        #the lstm cell
        self.lstm = LSTM(100, self.hidden_dim, self.layers, batch_first = True)
        
        #the fc with vocab size
        self.fc = Linear(self.hidden_dim, len(self.embeddings.vocab_dict) + 1)
        
        #context to every token
        # input_dim = 50+self.args.encoder.hidden_dim*self.args.encoder.layers
        # self.fc1 = Linear(input_dim, 50)
        self.attention = Attention(args)
        
        #softmax layer
        self.softmax = Softmax(dim = -1)

        #generate datset specific masks
        self.generate_masks()

    '''
    decoder only produce output probablity for 
    in database keywords
    '''
    def generate_masks(self):
        
        #dbid_dict[<db_id>] = index
        self.dbid_dict = dict()
        
        #open file
        file = os.path.join(self.args.data_root, 'masks.txt')
        file = open(file, 'r')
        
        #db_mask: [NO_DATABASES x DECODER_VOCAB]
        self.db_mask = []
        ind = 0
        for line in file:
            line = line.strip().split(' ')
            self.dbid_dict[line[0]] = ind

            #add padding + sos + eos + unknown
            self.db_mask.append([0 , 0] + [float(x) for x in line[1:]] + [1,1])
            ind += 1

        #db_mask: [DATABASES x 1 x DEOCDER_VOCAB_SIZE]
        self.db_mask = torch.tensor(self.db_mask).unsqueeze(1).cuda()
        self.db_mask = 1 - self.db_mask
        self.db_mask[self.db_mask == 1] = -1e10


    def forward(self, db_id, prev_output, context, prev_hidden, mask):
        '''
        Args:
            db_id: [B]
            prev_output: B x 1
            prev_hidden: (hidden,cell) [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM], [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM]
            hidden_all: [B, ISL, ENCOCDER_HIDDEN_DIM*ENCODER_LAYERS]
            mask: [B x ISL]
        Returns:
            predicted_labels: B x 1 x DECODER_VOCAB_SIZE
            next_hidden: (hidden,cell) [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM], [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM]
        '''
        #prev_output: B x 1 x DECODER_INPUT_DIM
        prev_output_embedding = self.embeddings(prev_output)
        
        # #context: [B x 1 x 50]
        # context = context.gather(1, lens - 1)
        context = self.attention(context, prev_output_embedding, context, mask)
        final_input = torch.cat([context, prev_output_embedding], dim = -1)

        #output:B x 1 x DECODER_HIDDEN_DIM
        #next_hidden: (hidden,cell) [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM], [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM]
        output, next_hidden = self.lstm(final_input, prev_hidden)

        #predicted_labels: B x 1 x DECODER_VOCAB_SIZE
        predicted_labels = self.fc(output)

        #need to add the mask of databse
        #mask: [B x 1 x DEOCDER_VOCAB_SIZE]
        mask = self.db_mask[db_id,:]
        predicted_labels = predicted_labels + mask
        predicted_labels = self.softmax(predicted_labels)
        return predicted_labels, next_hidden
    
    #do beam search with only one batch size
    def beam_search(self, db_id, hidden_all, prev_hidden, mask):
        '''
        Args:
                prev_hidden: (hidden, cell):
                    [ENCODER_LAYER, 1, ENCOCDER_HIDDEN_DIM], 
                    [ENCODER_LAYER, 1, ENCOCDER_HIDDEN_DIM]
        Return:
            DECODER_SENTENCE: [N|MAX_LEN]
        '''
        #prev_token: [1]: <SOS>
        best = []
        
        #add prev_token to beam
        beam = [{'prev': [1], 'score':1.0, 'hidden':prev_hidden}]

        #while beam is not empty
        for i in range(self.args.decoder.max_len):
            
            #children beam
            next_beam = []

            #for each node explore
            for node in beam:
                
                #check if end of token
                if(node['prev'][-1] == self.embeddings.end_token):
                    next_beam.append(node)
                
                #search for children
                else:
                    prev_output = torch.tensor([[node['prev'][-1]]]).long().cuda()
                    prev_hidden = node['hidden']
                    
                    #predicted_labels: DECODER_VOCAB
                    predicted_labels, next_hidden = self.forward(db_id, prev_output, hidden_all, prev_hidden, mask)
                    predicted_labels = predicted_labels.squeeze(0).squeeze(0)

                    #topk_nodes: K
                    topk_nodes = torch.topk(predicted_labels, self.k, dim = -1)

                    #add them to new nodes
                    for node_id in range(self.k):
                        prev_output = node['prev'] + [topk_nodes[1][node_id].item()]
                        score = node['score'] + math.log(topk_nodes[0][node_id])
                        new_node = {'score':score, 'prev':prev_output,'hidden':next_hidden}
                        next_beam.append(new_node)

            #sort and keep top k
            next_beam.sort(key = lambda x: -1*x['score'])
            next_beam = next_beam[:self.k]
            beam = next_beam
        return beam[0]['prev']

    # def forward(self, db_id, output_sequence):
    #     pass


class TextToSql(Module):

    def __init__(self,args):
        super(TextToSql, self).__init__()
        self.args = args
        
        #get the encoder
        self.encoder = Encoder(args)
        
        #decoder
        self.decoder = Decoder(args)
        
        #self.decoder.embeddings.init_with_glove(self.encoder.embeddings)
        self.encoder_vocab = None
        self.decoder_vocab = self.decoder.embeddings.vocab

        #the softmax
        self.softmax = Softmax(dim = -1)

        for param in self.encoder.model.parameters():
            param.requires_grad = False

    def forward(self, db_id, input_sequence, output_sequence = None):
        '''
        args:
            db_id: N
            input_sequence: list of input sentences
            output_sequence: N x OSL
        '''
        
        #context: [B, ISL, 768]
        #mask: [B,ISL]
        mask , context = self.encoder(input_sequence)

        #hidden: [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM]
        #cell: [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM
        hidden = torch.zeros(1,context.shape[0], 128).cuda()
        cell = torch.zeros(1,context.shape[0], 128).cuda()

        #decode the output sequence
        if(output_sequence != None):
            
            #at decoding 0
            #prev_output: [B x 1]
            output_sequence_len = output_sequence.shape[1] - 1
            prev_output = output_sequence[:,:1]

            #final_output: B x OSL x DECODER_VOCAB_SIZE
            final_output = []
            
            #now do the rollouts
            for i in range(0,output_sequence_len):

                #next_output: [B, 1, DECODER_VOCAB_LEN]
                #hidden: [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM]
                #cell: [ENCODER_LAYER, B, ENCOCDER_HIDDEN_DIM]
                next_output, (hidden, cell) = self.decoder(db_id, prev_output, context, (hidden, cell), mask)
                
                #teacher forcing?
                prob = np.random.rand()
                if(prob <= self.args.teacher_forcing_prob):
                    prev_output = output_sequence[:,(i+1):(i+2)]
                else:
                    prev_output = torch.argmax(next_output, dim = -1)
                
                #store the output
                final_output.append(next_output)
            final_output = torch.cat(final_output, dim = 1)
        else:
            final_output = self.decoder.beam_search(db_id, context, (hidden, cell), mask)

        return final_output
