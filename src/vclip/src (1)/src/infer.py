#python3 infer.py --model_file models/lstm_lstm_attn.pth --model_type lstm_lstm_attn --test_data_file sql_query.csv --output_file output.csv

from preprocess import Preprocessor
from config import get_config
from argparse import ArgumentParser
import torch
import csv 
import pdb
from tqdm import tqdm
import subprocess
def get_parser():
    
    #the parser for the arguments
    parser = ArgumentParser(
                        prog = 'python main.py',
                        description = 'This is main file for COL775 A1. You can train model from scratch or resume training, test checkpoint from this file',
                        epilog = 'thank you!!')

    #there are two tasks ['train', 'test']
    parser.add_argument('--model_file', default = '', required=True)

    #there are two tasks ['train', 'test']
    parser.add_argument('--model_type',  choices=[ 'lstm_lstm', 'lstm_lstm_attn', 'bert_lstm_attn_frozen', 'bert_lstm_attn_tuned'], required=True)
    
    #files
    parser.add_argument('--test_data_file', default = '', required=True)
    parser.add_argument('--output_file', default = '', required=True)
    
    return parser

def solve_for_one_lstm(model, text_tokens, value_mapping, db):
    
    tokens = [model.encoder_vocab(i) for i in text_tokens]
    tokens = torch.tensor(tokens).unsqueeze(0).cuda()
    
    db_id = torch.tensor([model.decoder.dbid_dict[db]]).cuda()
    
    output = model(db_id, tokens)[1:]
    for i in range(len(output)):
        output[i] = model.decoder.embeddings.vocab_inv[output[i]]
    output = (' '.join(output))
    output = output.replace(' . ', '.')
    output = output.replace('. ', '.')
    output = output.replace(' .', '.')
    output = output.replace('<EOS>', '')
    for key in value_mapping:
        output = output.replace(value_mapping[key], key)
    return output



def solve_for_one_bert(model, text_tokens, value_mapping, db):
    
    db_id = torch.tensor([model.decoder.dbid_dict[db]]).cuda()
    text_tokens = ' '.join(text_tokens)
    output = model(db_id,[text_tokens])[1:]
    for i in range(len(output)):
        output[i] = model.decoder.embeddings.vocab_inv[output[i]]
    output = (' '.join(output))
    output = output.replace(' . ', '.')
    output = output.replace('. ', '.')
    output = output.replace(' .', '.')
    output = output.replace('<EOS>', '')
    for key in value_mapping:
        output = output.replace(value_mapping[key], key)
    return output
# class CustomUnpickler(pickle.Unpickler):

#     def find_class(self, module, name):
#         if name == 'Manager':
#             from settings import Manager
#             return Manager
#         return super().find_class(module, name)


if __name__ == '__main__':

    #parse the arguments
    parser = get_parser()
    args = parser.parse_args()
    
    #get arguments
    config = get_config([])

    if(args.model_type == 'lstm_lstm'):
        cmd_str =  f'cp lstm_lstm.py model.py'
        subprocess.run(cmd_str, shell=True)
        solve_for_one = solve_for_one_lstm

    if(args.model_type == 'lstm_lstm_attn'):
        cmd_str =  f'cp lstm_lstm_attn.py model.py'
        subprocess.run(cmd_str, shell=True)
        solve_for_one = solve_for_one_lstm

    if('bert' in args.model_type):
        cmd_str =  f'cp bert_lstm_freeze.py model.py'
        subprocess.run(cmd_str, shell=True)
        solve_for_one = solve_for_one_bert

    model = torch.load(args.model_file)
    model.eval().cuda()
    #get the parser
    preprocess = Preprocessor(config)

    #read file one by one
    data_file = open(args.test_data_file, 'r')
    data_file = csv.reader(data_file)
    next(data_file, None)

    #output file
    # outfile = csv.writer(open(args.output_file, 'w'))
    # goldfile = csv.writer(open('gold.csv', 'w'))

    outfile = open(args.output_file, 'w')
    goldfile = open('gold.csv', 'w')

    for line in tqdm(data_file):
        text_tokens, value_mapping = preprocess.get_text_tokens(line[2])
        db = line[0]
        sql_query = solve_for_one(model, text_tokens, value_mapping, db)
        # outfile.writerow([sql_query])
        # goldfile.writerow([line[1]])
        outfile.write(sql_query + '\n')
        line[1] = line[1].replace('\t', ' ')
        goldfile.write(line[1] +"\t" + f'{line[0]}\n')
