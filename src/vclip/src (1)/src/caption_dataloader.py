import torch
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pdb
'''
question types:
    1. descriptive -> sub_types:
                        1. count
                        2. query_material
                        3. query_shape
                        3. query_color
                        4. exist
                        
    2. explanatory -> choices, ans per choise {'correct', 'wrong'} (multi or no correct)
    3. counterfactual -> choices, ans per choise {'correct', 'wrong'} (multi or no correct)
    4. predictive > choices,  ans per choise {'correct', 'wrong'} (Maybe only one correct option)


'''

class QuestionAns:
    '''Claass to store question ans and other details'''

    def __init__(self, q_id, q, q_type, q_sub_type, ans, video_id, scene_index, program, choice_id = None, choice = None ):

        self.q = q
        self.q_id = q_id
        self.q_type = q_type
        self.q_sub_type = q_sub_type
        self.ans = ans
        self.video_id = int(video_id.replace('.', '_').split('_')[1])
        self.scene_index = scene_index
        self.program = program
        self.choice_id = choice_id
        self.choice = choice

        ''' Note that there are 2  one for question and other for choice, I am not parsing the program for choice'''


        # Minor verification of structure of file
        if q_type == 'explanatory' and choice_id is None:
            raise Exception('ERROR parsing json: explanatory question found with no choice_id') 

        if q_type == 'counterfactual' and choice_id is None:
            raise Exception('ERROR parsing json: counterfactual question found with no choice_id') 

        if q_type == 'descriptive' and q_sub_type is None:
            raise Exception('ERROR parsing json: descriptive question found with no question sub type')

    def get_qa(self, add_more_content = []):
        '''
        Parameters:
        -----
        Input
        -----
        add_more_content:   takes a list of strings which you want to append to question
                            valid values {'program', 'scene_index'}
                            NOTE: I have not clue of both these, so prefer not to add them
        ------
        Output
        ------
        question    : string of question
        ans         : string of ans
        '''

        question = [self.q]
        
        # if self.q_sub_type is not None:
        #     question += [self.q_sub_type]

        if self.choice is not None:
            question += [self.choice]
        
        # if 'program' in add_more_content:
        #     question += self.program
        
        # if 'scene_index' in add_more_content:
        #     question += [str(self.scene_index)]

        question = ' , '.join(question)
        ans = self.ans

        return question, ans
    
    # Not in use
    def get_qa_enc(self, token, add_more_content = []):

        question, ans = self.get_qa(add_more_content)
        
        question = token.tokenize(question)
        question = token.convert_tokens_to_ids(question)
        ans = token.tokenize(ans)
        ans = token.convert_tokens_to_ids(ans)

        return question, ans



def get_qa_list(json_file):

    print('For json file:', json_file)

    with open(json_file, 'rb') as f:
        json_data = json.load(f)
    
    qa_list = []

    for video_json in json_data:
        video_id = video_json['video_filename']
        scene_index = video_json['scene_index']

        # Iterating through question for this video
        for q_json in video_json['questions']:
            q_id = q_json['question_id']
            q = q_json['question']
            q_type = q_json['question_type']
            q_sub_type = None
            
            if 'question_subtype' in q_json:
                q_sub_type = q_json['question_subtype']
            program = q_json['program']

            if q_type in ['explanatory','counterfactual', 'predictive']:
                # Iterating through choice
                for choice_json in q_json['choices']:
                    choice = choice_json['choice']
                    choice_id = choice_json['choice_id']

                    ans = choice_json['answer']
                    qa_obj = QuestionAns(
                        q_id, q, q_type, q_sub_type,
                        ans, video_id, scene_index, 
                        program, choice_id, choice)
                    
                    qa_list.append(qa_obj)
                
            else:
                ans = q_json['answer']
                choice = None
                choice_id = None
                qa_obj = QuestionAns(
                        q_id, q, q_type, q_sub_type,
                        ans, video_id, scene_index, 
                        program, choice_id, choice)
                
                qa_list.append(qa_obj)

    return qa_list

## Main dataloader

class AnsVocab:
    def __init__(self):
        self.token2idx = {}
        self.token2count = {}
        self.idx2token = {}
        self.num_tokens = 0
    
    def add_token(self, word):
        if word in self.token2idx:
            self.token2count[word] += 1
        else:
            self.token2idx[word] = self.num_tokens
            self.token2count[word] = 1
            self.idx2token[self.num_tokens] = word
            self.num_tokens += 1	
    
    def get_id(self, word):
        if(word not in self.token2idx):
            raise ValueError(f"{word} not in answer vocab")
        return self.token2idx[word]


# Always use this function to get vocab of answer
def get_op_vocab():
    ans_token_ordered = [
        '0','1','2','3','4','5',
        'metal','rubber',
        'cube','cylinder','sphere',
        'blue','brown', 'cyan','gray','green','purple','red','yellow',
        'yes','no',
        'correct','wrong']
    ans_vocab = AnsVocab()
    for token in ans_token_ordered:
        ans_vocab.add_token(token)
    return ans_vocab

class CaptionLoader(Dataset):

    def __init__(self, json_file, tokenizer, questions_per_video, transform=None):
        """
        Args:
            json_file (string): Path to the json file with captions for CLEVERER.
            questions_per_video (int): questions need to sample for each video
            tokenizer : bert_tokenizer to tokenize the questions
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.json_file = json_file
        self.tokenizer = tokenizer
        self.questions_per_video = questions_per_video
        self.transform = transform

        #get the answer vocab
        self.output_vocab = get_op_vocab()
        
        #read and tokenize questions
        self.question_ans_list = get_qa_list(json_file)
        self.question_text = self._question_text()
        questions = self._tokenize()

        self.questions = torch.tensor(np.array([question['input_ids'] for question in questions]))
        self.attn_mask = torch.tensor(np.array([question['attention_mask'] for question in questions]))
        
        #self.padded_tokens = self._pad(tokenized_question_indices)
        self.mask = self._make_mask()

        #get the labels for the questions
        self.labels = []
        for question in self.question_ans_list:
            self.labels.append(self.output_vocab.get_id(question.ans))
        self.labels = torch.tensor(np.array(self.labels))
        
        #make video: question_list mapping
        self.video_question_mapping = dict()
        for question_id, question in enumerate(self.question_ans_list):
            if(question.video_id not in self.video_question_mapping):
                self.video_question_mapping[question.video_id] = []
            self.video_question_mapping[question.video_id].append(question_id)

        #raise exception if questions_per_video > total_question for any video
        for video in self.video_question_mapping:
            if(len(self.video_question_mapping[video]) < self.questions_per_video):
                raise ValueError(f"Video {video} have {len(self.video_question_mapping[video])} questions") 
	
    def __len__(self):
        return len(self.video_question_mapping)

    def __getitem__(self, idx):
        
        #sample from question list
        questions = self.video_question_mapping[idx]
        samples = np.random.choice(questions, self.questions_per_video, replace = False)

        questions = self.questions[samples]
        attn_mask = self.attn_mask[samples]
        mask = self.mask[samples]
        labels = self.labels[samples]
        # if self.transform:
        #     question = self.transform(question)

        return questions, attn_mask, mask, labels
    
    def _tokenize(self):

        tokenized_question_indices = [self.tokenizer(question_ans_obj.get_qa()[0]) for question_ans_obj in self.question_ans_list]
        #tokenized_question_indices = [torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_question)) for tokenized_question in tokenized_questions]
        return tokenized_question_indices

    def _question_text(self):
        question_text = np.array([question_ans_obj.get_qa()[0] for question_ans_obj in self.question_ans_list])
        #tokenized_question_indices = [torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_question)) for tokenized_question in tokenized_questions]
        return question_text

    def _pad(self, tokenized_question_indices):
        '''Returns a tensor of padded tokens'''
        # self.max_length = max(len(row) for row in self.tokenized_question_indices)
        '''Using pad_sequence of pytorch rnn so it automatically converts the input to a tensor of given dim'''
        return pad_sequence(tokenized_question_indices, batch_first=True)
    
    def _make_mask(self):
        '''
        All answers:
        ['0','1','2','3','4','5',
        'metal','rubber',
        'cube','cylinder','sphere',
        'blue','brown', 'cyan','gray','green','purple','red','yellow',
        'yes','no',
        'correct','wrong']
        '''
        ans_vocab = get_op_vocab()
        non_masked_idx = {
            'descriptive': {
                'count': [0,1,2,3,4,5],
                'query_material': [6,7],
                'query_shape': [8,9,10],
                'query_color': [11,12,13,14,15,16,17,18],
                'exist': [19,20]
            },
            'explanatory': [21,22],
            'counterfactual': [21,22],
            'predictive': [21,22]
        }
        
        mask = torch.zeros((len(self.question_ans_list), ans_vocab.num_tokens))
        mask = mask - float('Inf')
        

        for idx, qa in enumerate(self.question_ans_list):
            if qa.q_type == 'descriptive':
                temp_index = non_masked_idx[qa.q_type][qa.q_sub_type]
            else:
                temp_index = non_masked_idx[qa.q_type]
            mask[idx][temp_index] = 0.0
        
        return mask


'''
To use:
from transformers import BertTokenizer

# The below line requires internet when running for first time, for downloading bert tokenizer
token = BertTokenizer.from_pretrained('bert-base-uncased')


dataset = CaptionLoader(json_file='temp_train.json', tokenizer=token)
BATCH_SIZE = 32
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
'''
