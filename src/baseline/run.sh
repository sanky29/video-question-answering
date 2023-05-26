#python main.py --task train --args train.questions_per_video 8 experiment_name resnet18_lstm1_qpv_8
#python main.py --task train --args train.questions_per_video 8 experiment_name resnet18_trans1_qpv_8 video_models.arch transformer 
python main.py --task train --args video_models.arch transformer experiment_name Transformer_1_lr_001_qpv_8_bs_4 lr 0.001