import torch 
import torchvision
import math
import pdb
from tqdm import tqdm
from torch.nn import Linear, Module
from torch.nn import ModuleList, ReLU, Softmax, Parameter, LSTM, Embedding
from torchvision import transforms
import numpy as np 
import os

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class Attention(Module):

    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim

        self.input_dim = input_dim
        self.out_dim = out_dim

        self.k = Linear(self.input_dim, self.hidden_dim, bias = False)
        self.v = Linear(self.input_dim, self.out_dim, bias = False)
        self.q = Linear(self.out_dim, self.hidden_dim, bias = False)
        
        self.softmax = Softmax(dim = -1)
    
    def forward(self, key, query, value):
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
        # attn_mask = attn_mask + mask

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
        x = x + self.pe[:,:x.shape[1],:].to(x.device)
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
        
        #a special token 
        self.cls_token = Parameter(torch.randn(1,1,arch.input_dim), requires_grad = True)

    def forward(self, x):
        '''
        Args:
            x: [B x SEQ_LEN x ENCODER_IN_DIM] 
        Return:
            update: [B x SEQ_LEN x ENCODER_OUT_DIM] 
        '''
        B = x.shape[0]

        #x: [B x SEQ_LEN x ENCODER_OUT_DIM]
        x = self.position_embedings(x)
        x = torch.cat([self.cls_token.expand(B,-1,-1), x], dim = 1)
        x = self.transformer(x)

        return x


class FrameEncoder(Module):

    def __init__(self, args):
        super(FrameEncoder, self).__init__()
        resnet_model = torchvision.models.resnet18()
        #resnet_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))

    def forward(self, x):
        '''
        Args:
            x: [B x 3 x H x W]
        Returns:
            output: [B x FRAME_ENCODING_DIM]
        '''
        # x = transforms.functional.resize(x, (256,256))
        # x = transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True).float()
        
        #x: [B x 512]
        x = x.float()
        output = self.model(x).squeeze(-1).squeeze(-1)
        return output

class Resnet3D(Module):

    def __init__(self, args):
        super(Resnet3D, self).__init__()
        #self.model =  torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        #self.model.load_state_dict(torch.load('~/resnet3d.pt'))
        self.model = torch.load('/home/textile/btech/tt1170896/resnet3d_model.pt')
        self.mean = [0.45, 0.45, 0.45]
        self.std = [0.225, 0.225, 0.225]
    
    def forward(self, x):
        T = x.shape[1]
        x = x.flatten(0,1)
        x = transforms.functional.resize(x, (256,256))
        x = transforms.functional.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True).float()
        x = x.unflatten(0,(-1,T))
        x = x.permute(0,2,1,3,4)
        output = self.model(x.float())
        return output.unsqueeze(1)

class VideoEncoder(Module):

    def __init__(self, args):
        super(VideoEncoder, self).__init__()
        
        #get the frame encoder
        self.args = args

        #get the temporal encoder
        self.get_temporal_encoder()

        if(self.args.video_models.arch == 'resnet3d'):
            self.forward = self.forward_2
        else:
            self.frame_encoder = FrameEncoder(args)

            #linear layer to transform to temporal
            self.proj = Linear(512, self.args.video_models[args.video_models.arch].input_dim)
        
            self.forward = self.forward_1


    def get_temporal_encoder(self):
        if(self.args.video_models.arch == 'lstm'):
            self.temporal_encoder = LSTMEncoder(self.args)
        elif(self.args.video_models.arch == 'transformer'):
            self.temporal_encoder = TransformerVideoEncoder(self.args)
        elif(self.args.video_models.arch == 'resnet3d'):
            self.temporal_encoder = Resnet3D(self.args)

    def forward_2(self,x):
        '''
        Args:
            x: [B x T x 3 x H x W]
        Returns:
            output: [B x T x ENCODING_DIM]
        '''
        T = x.shape[1]
        B = x.shape[0]
        
        #output: [B x T x VIDEO_ENCODING_DIM]
        x = self.temporal_encoder(x)
        return x
        
    def forward_1(self,x):
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
        x = self.frame_encoder(x)
        x = output.unflatten(0, (B,T))
        
        #output: [B x T x VIDEO_MODEL_INPUT_DIM]
        x = self.proj(x)

        #output: [B x T x VIDEO_ENCODING_DIM]
        x = self.temporal_encoder(x)
        return x
        

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
        video_enc = self.video_proj(video_enc[:,:,0,:])

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

'''
this is final classifies
'''
class ClassifierAttn(Module):

    def __init__(self, args):
        super(ClassifierAttn, self).__init__()

        self.args = args
        self.video_proj = Linear(self.args.video_models.out_dim, self.args.classifier.hidden_dim)
        self.ques_proj = Linear(self.args.ques_models.out_dim, self.args.classifier.hidden_dim)

        self.attention = Attention(self.args.classifier.hidden_dim, self.args.classifier.hidden_dim*2, self.args.classifier.hidden_dim)

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
        QPV = video_enc.shape[1]
        #video_enc: [B x QPV x SEQ_LEN_QUES x HIDDEN_DIM]
        # todo: take the first embedding (add Special token for it)
        video_enc = self.video_proj(video_enc)

        #ques_enc: [B x QPV x HIDDEN_DIM]
        ques_enc = self.ques_proj(ques_enc[:,:,0,:])

        #get the attention
        video_enc = video_enc.flatten(0,1)
        ques_enc = ques_enc.flatten(0,1).unsqueeze(1)
        
        #video_enc: [(B*QPV) x HIDDEN_DIM]
        video_enc = self.attention(video_enc, ques_enc, video_enc)
        
        #reshape to original shape
        video_enc = video_enc.unflatten(0,(-1,QPV)).squeeze(2)
        ques_enc = ques_enc.unflatten(0,(-1,QPV)).squeeze(2)
        
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
        if(self.args.video_models.arch == 'resnet3d'):
            self.classifier = Classifier(args)
        else:
            self.classifier = ClassifierAttn(args)

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
                ques: [input_id: [B x QPV x SEQ_LEN], attn: [B x QPV x SEQ_LEN]] list of list of text
                    ''attn: [1 1 1 1 0 0 0]''
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
