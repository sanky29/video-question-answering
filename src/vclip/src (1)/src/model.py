import torch 
# import torchvision ## Commented to avoid error
import math
import pdb
from tqdm import tqdm
from torch.nn import Linear, Module
from torch.nn import ModuleList, ReLU, Softmax, Parameter, LSTM, Embedding
# from torchvision import transforms ## commented to avoid error
import numpy as np 
import os

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from mmpt.models import MMPTModel


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


'''
lstm for the frame encoding
'''
class LSTMEncoder(Module):

    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
        self.args = args
        
        #the architecture
        arch = self.args.video_models.lstm
        self.model = LSTM(arch.input_dim, arch.hidden_dim, arch.layers, batch_first = True, bidirectional= False)

    def forward(self, x):
        '''
        Args:
            x: [B x SEQ_LEN x ARCH.INPUT_DIM]
        Returns:
            output: [B x SEQ_LEN x ]
        '''
        #output: 
        # ( [B x T x (BIDIRECTIONAL x HIDDEN_DIM) ], 
        #    ( [B x 1 x (BIDIRECTIONAL x HIDDEN_DIM)], [B x 1 x (BIDIRECTIONAL x HIDDEN_DIM)] ) )
        output = self.model(x)
        return output[0]


'''position encoder for video model'''
class PositionEmbedding(Module):

    def __init__(self, args):
        super(PositionEmbedding, self).__init__()
        
        #the position dimension
        self.dim = args.video_models.transformer.input_dim

        #the embeddings
        #pe: MAX_LEN x DIM
        self.pe = torch.zeros(args.video_models.transformer.max_len,self.dim)
        
        #seq: DIM
        #pos: MAX_LEN
        seq = torch.exp(-1*torch.arange(0,self.dim,2)/float(self.dim)*math.log(10000))
        pos = torch.arange(0,args.video_models.transformer.max_len)
        
        #pe: 1 x MAX_LEN x DIM
        self.pe[:,0:self.dim//2] = torch.sin(pos.unsqueeze(-1)*seq.unsqueeze(0))
        self.pe[:,self.dim//2:] = torch.cos(pos.unsqueeze(-1)*seq.unsqueeze(0))
        self.pe = self.pe.unsqueeze(0)
        self.pe = self.pe.to(args.device)
    
    def forward(self, x):
        '''
        Args:
            x: [B x SEQ_LEN x DIM]
        Returns:
            x: [B x SEQ_LEN x DIM]
        '''
        x = x + self.pe[:,:x.shape[1],:]
        return x

'''
transformer with attention to previous states.
'''
class TransformerVideoEncoder(Module):

    def __init__(self,args):
        super(TransformerVideoEncoder,self).__init__()
        
        #meta data
        self.args = args
        arch = self.args.video_models.transformer

        #the position embedder
        self.position_embedings = PositionEmbedding(args)
        
        #encdoder layers
        encoder_layers = TransformerEncoderLayer(arch.input_dim, arch.heads, arch.hidden_dim, dropout = arch.dropout, batch_first = True) 
        self.transformer = TransformerEncoder(encoder_layers, arch.layers)
        
    def forward(self, x):
        '''
        Args:
            x: [B x SEQ_LEN x ENCODER_IN_DIM] 
        Return:
            update: [B x SEQ_LEN x ENCODER_OUT_DIM] 
        '''

        #return the ans
        #x: [B x SEQ_LEN x ENCODER_OUT_DIM]
        x = self.position_embedings(x)
        x = self.transformer(x)
        return x


class FrameEncoder(Module):

    def __init__(self, args):
        super(FrameEncoder, self).__init__()
        resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))

    def forward(self, x):
        '''
        Args:
            x: [B x 3 x H x W]
        Returns:
            output: [B x FRAME_ENCODING_DIM]
        '''
        x = transforms.functional.resize(x, (256,256))
        x = transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True).float()
        
        #x: [B x 512]
        x = output = self.model(x).squeeze(-1).squeeze(-1)
        return output

class VideoEncoder(Module):

    def __init__(self, args):
        super(VideoEncoder, self).__init__()
        
        #get the frame encoder
        self.args = args
        self.frame_encoder = FrameEncoder(args)

        #linear layer to transform to temporal
        self.proj = Linear(512, self.args.video_models[args.video_models.arch].input_dim)
        
        #get the temporal encoder
        self.get_temporal_encoder()

    def get_temporal_encoder(self):
        if(self.args.video_models.arch == 'lstm'):
            self.temporal_encoder = LSTMEncoder(self.args)
        elif(self.args.video_models.arch == 'transformer'):
            self.temporal_encoder = TransformerVideoEncoder(self.args)
        
        
    def forward(self,x):
        '''
        Args:
            x: [B x T x 3 x H x W]
        Returns:
            output: [B x T x ENCODING_DIM]
        '''
        T = x.shape[1]
        B = x.shape[0]
        
        #x: [B*T x 3 x H x W]
        x = x.flatten(0,1)
        
        #output: [B x T x 512]
        output = self.frame_encoder(x)
        output = output.unflatten(0, (B,T))
        
        #output: [B x T x VIDEO_MODEL_INPUT_DIM]
        output = self.proj(output)

        #output: [B x T x VIDEO_ENCODING_DIM]
        output = self.temporal_encoder(output)
        return output
        

class TextEncoder(Module):

    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        self.model = BertModel.from_pretrained(args.bert_model)

    def forward(self, input):
        '''
        Args:
            input: [B x QPV x SEQ_LEN] list of question text
        Returns:
            output: [B x QPV x QUES_ENC_DIM]
        '''
        QVP = len(input['attention_mask'][0])
        B = len(input['attention_mask'])
        # input = [i for j in input for i in j]
        
        # #get tokens
        # output = self.tokenizer(input, return_tensors = 'pt', padding = True, truncation=True)
        # for e in output:
        #     output[e] = output[e].cuda()

        #attn_mask: [B, QVP, SEQ_LEN]
        input['attention_mask'] = input['attention_mask'].flatten(0,1)
        input['input_ids'] = input['input_ids'].flatten(0,1)
        
        #output: [B, QVP, SEQ_LEN, BERT_OUT_DIM]
        output = self.model(**input)
        output = output.last_hidden_state.unflatten(0, (B, QVP))
        attn_mask = input['attention_mask'].view(B, QVP, -1)
        
        return output, attn_mask

'''
this is final classifies
'''
class Classifier(Module):

    def __init__(self, args):
        super(Classifier, self).__init__()

        self.args = args
        self.video_proj = Linear(self.args.video_models.out_dim, self.args.classifier.hidden_dim)
        self.ques_proj = Linear(self.args.ques_models.out_dim, self.args.classifier.hidden_dim)

        self.lin1 = Linear(self.args.classifier.hidden_dim*2, self.args.classifier.hidden_dim)
        self.lin2 = Linear(self.args.classifier.hidden_dim, self.args.classifier.out_dim)
        self.relu = ReLU()
        self.softmax = Softmax(dim = -1)

    def forward(self, video_enc, ques_enc, ques_mask, output_mask):
        '''
        Args:
            video_enc: [B x QPV x SEQ_LEN_VIDEO x VIDEO_ENCODER_DIM]
            ques_enc: [B x QPV x SEQ_LEN_QUES x QUES_ENCODER_DIM]
            ques_mask: [B x QPV x SEQ_LEN_QUES]
            output_mask: [B x QPV x OUT_DIM]
        Returns:
            output: [B x QPV x OUT_DIM]
        '''

        #video_enc: [B x QPV x HIDDEN_DIM]
        # todo: take the first embedding (add Special token for it)
        video_enc = self.video_proj(video_enc[:,:,-1,:])

        #ques_enc: [B x QPV x HIDDEN_DIM]
        ques_enc = self.ques_proj(ques_enc[:,:,0,:])

        #output: [B x QPV x OUTPUT_DIM]
        output = torch.cat([video_enc, ques_enc], dim = -1)
        output = self.lin1(output)
        output = self.relu(output)
        output = self.lin2(output)
        output = output + output_mask
        output = self.softmax(output)
        return output

class VQA(Module):

    def __init__(self, args, mode = 'train'):
        super(VQA, self).__init__()
        self.args = args
        
        #get the encoder
        self.text_encoder = TextEncoder(args)
        
        #decoder
        self.video_encoder = VideoEncoder(args)

        #the classifier
        self.classifier = Classifier(args)

        #the tokenizer
        self.tokenizer = lambda x: self.text_encoder.tokenizer(
                        x,                      # Sentence to encode.
                        max_length = 45, 
                        padding = 'max_length')  # Return pytorch tensors.

    def forward(self, data, mode = 'train'):
        '''
        Args:
            data:   dictionary with following keys
                video: [B x T x C x H x W]
                ques: [B x QPV] list of list of text
                mask: [B x QPV x OUTPUT_DIM]
        Returns:
            output: [B x QPV x OUTPUT_DIM] probabilites
        '''
        QPV = len(data['ques']['attention_mask'][0])
        
        # #get video encoding
        # #video_enc: [B x SEQ_LEN x VIDEO_ENC_DIM] 
        video_enc = self.video_encoder(data['video'])
        
        #get the final output
        #video_enc: [B x QPV x SEQ_LEN x VIDEO_ENC_DIM] 
        video_enc = video_enc.unsqueeze(1).expand(-1, QPV, -1, -1)
        
        #get question encoding
        #ques_enc: [B x QPV x SEQ_LEN x BERT_OUT_DIM] 
        ques_enc, attn_mask = self.text_encoder(data['ques'])
        
        #output: [B x QPV x OUTPUT_DIM] 
        #(Pdb) ques_enc.shape
        # torch.Size([4, 8, 34, 768])
        # (Pdb) video_enc.shape
        # torch.Size([4, 8, 128, 256])
        output = self.classifier(video_enc, ques_enc, attn_mask, data['mask'])
        return output

class Vclip(Module):
    def __init__(self, args):
        super(Vclip, self).__init__()
        self.args = args
        
        #the architecture
        # arch = self.args.video_models.lstm
        # self.model = LSTM(arch.input_dim, arch.hidden_dim, arch.layers, batch_first = True, bidirectional= False)
        arch = self.args.vclip.base 

        # todo: IMP: uncomment line, comment the 2 lines after that
        self.model, tokenizer, _ = MMPTModel.from_pretrained(arch.yaml_path)
        # self.model = None
        # tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
        
        self.tokenizer = lambda x: tokenizer(
                        x,                      # Sentence to encode.
                        max_length = 45, 
                        padding = 'max_length')

        ## For decoder/ classifier
        self.lin1 = Linear(self.args.classifier.hidden_dim*2, self.args.classifier.hidden_dim)
        self.lin2 = Linear(self.args.classifier.hidden_dim, self.args.classifier.out_dim)
        self.relu = ReLU()
        self.softmax = Softmax(dim = -1)

    def forward(self, data):
        '''
        Args:
            data:   dictionary with following keys
                video: [B x T x C x H x W]
                ques: {
                    'input_ids': [B x QPV x Seq_len], 
                    'attention_mask': [B x QPV x Seq_len]
                    }
                mask: [B x QPV x OUTPUT_DIM]
        
        Returns:
            output: [B x 1 x 23]
        '''
        
        B, T, C, H, W = data['video'].shape

        # todo: IMP: remove this
        # pdb.set_trace()

        QPV = data['ques']['input_ids'].shape[1]

        # cant process when multiple questions are given, don't know what to do
        if QPV > 1:
            raise Exception('[ERROR]: Code is Not working for QPV>1')

        # vclip input: [B, QPV, T, H, W, C] -- correct
        # vclip input: [B, T, QPV, H, W, C] -- incorrect
        output = self.model(
            data['video'].expand(B, QPV,-1,-1,-1, -1).view(B, QPV, T, H, W, C).float().cuda(), 
            data['ques']['input_ids'].view(1,-1), 
            data['ques']['attention_mask'].view(1,-1)
        )
        
        # Classifier
        output = torch.cat([output['pooled_video'], output['pooled_text']], dim = -1)
        output = self.lin1(output)
        output = self.relu(output)
        output = self.lin2(output)
        output = output + data['mask']
        output = self.softmax(output).view(1,1,-1)
        # fixed output shape for loss computation
        # todo: incorporate QPV 

        return output
