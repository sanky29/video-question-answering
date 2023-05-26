from yacs.config import CfgNode

default_config = {

    'train':{
        'video_dir': '../a2_sanket/data/train/video',
        'frame_dir': '../a2_sanket/data/train/frames',
        'question_file':'../data/train/train.json',
        'questions_per_video':1,
        'batch_size':1
    },

    'val':{
        'batch_size':1
    },
    
    'test':{
        'video_dir': '../a2_sanket/data/test/video',
        'frame_dir': '../a2_sanket/data/test/frames'
    },

    'video_models':{
        'arch':'lstm',
        'lstm':{
            'input_dim':128,
            'hidden_dim':256,
            'layers':1
        },
        'transformer':{
            'input_dim':256,
            'hidden_dim':512,
            'layers':2,
            'heads':8,
            'dropout':0.1,
            'max_len':1000
        }
        
    },

    'ques_models':{
        'out_dim': 768
    },

    'classifier':{
        'hidden_dim':768,
        'out_dim':23
    },
    'vclip':{
        'base':{
            'yaml_path': './vclip.yaml'
        }
    },

    'bert_model':'../data/bert_model',
    'bert_tokenizer':'../data/bert_tokenizer' ,

    'epochs':1000,
    'lr': 0.001,
    'device':'cuda',
    'checkpoint_every':1,
    'validate_every': 1,
    'val_split': 0.01,

    'root': '/home/cse/msr/csy217545/scratch/COL775/A2/experiments',
    'experiment_name': 'demo',
    'workers': 0,
    'resume_dir': None,
    'checkpoint': None

}
default_config = CfgNode(default_config)

#get the config from the extras
def get_config(extra_args):
    default_config.set_new_allowed(True)
    default_config.merge_from_list(extra_args)
    default_config.extras = extra_args
    return default_config


#load from a file
def load_config(file_name, new_config):
    
    #the file of args
    new_config.set_new_allowed(True)
    extras = new_config.extras
    new_config.merge_from_file(file_name)
    new_config.merge_from_list(extras)
    new_config.extras = []
    return new_config

#dump to file
def dump_to_file(file_name, config):
    config_str = config.dump()
    with open(file_name, "w") as f:
        f.write(config_str)


# python main.py --task train --args epochs 100 experiment_name Tra
