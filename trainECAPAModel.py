'''
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
'''

import os
import numpy as np
import argparse, glob, os, torch, warnings, time
import matplotlib.pyplot as plt
import soundfile as sf
import random

from tools import *
from dataLoader import train_loader
from ECAPAModel import ECAPAModel
from count_label import count_label
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import LeaveOneOut


##tensorboard setup 
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description = "ECAPA_trainer")
## Training Settings
parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments, eg: 200 for 2 second')
parser.add_argument('--max_epoch',  type=int,   default=80,      help='Maximum number of epochs')
parser.add_argument('--batch_size', type=int,   default=64,     help='Batch size')
parser.add_argument('--n_cpu',      type=int,   default=4,       help='Number of loader threads')
parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay every [test_step] epochs')

## Training and evaluation path/lists, save path
parser.add_argument('--path', type=str,   default="/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
#parser.add_argument('--path', type=str,   default="../dataset/SAVEE_RESAMPLED",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
#parser.add_argument('--path', type=str,   default="../dataset/RAVDESS_RESAMPLED",     help='The path of the training list, https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/train_list.txt')
parser.add_argument('--musan_path', type=str,   default="../../ECAPA-TDNN/dataset/musan_split",                    help='The path to the MUSAN set, eg:"/data08/Others/musan_split" in my case')
parser.add_argument('--rir_path',   type=str,   default="../../ECAPA-TDNN/dataset/rirs_noises/RIRS_NOISES/simulated_rirs",     help='The path to the RIR set, eg:"/data08/Others/RIRS_NOISES/simulated_rirs" in my case');
parser.add_argument('--speech_type', type = str, default="Neutral_speech", help='The speech type to train [Neutral_speech vs Emotional_speech]')

parser.add_argument('--save_path',  type=str,   default="exps/exp3/model",                                     help='Path to save the score.txt and models')
#parser.add_argument('--initial_model',  type=str,   default="../V1_0105/exps/exp1/model_0105.model",           help='Path of the initial_model')
#parser.add_argument('--initial_model',  type=str,   default="../V1_0125/exps/exp1/model_0125.model",           help='Path of the initial_model')
#parser.add_argument('--initial_model',  type=str,   default="../V1_0130/exps/exp1/model_0130.model",           help='Path of the initial_model')
#parser.add_argument('--initial_model',  type=str,   default="../V2/v21/exps/exp1/model_0120.model",           help='Path of the initial_model')
#parser.add_argument('--initial_model',  type=str,   default="../V2/v22/exps/exp1/model_0122.model",           help='Path of the initial_model')
parser.add_argument('--initial_model',  type=str,   default="../V3/exps/exp1/model_0129.model",           help='Path of the initial_model')

## Model and Loss settings
#parser.add_argument('--C',       type=int,   default=1024,   help='Channel size for the speaker encoder')
parser.add_argument('--C',       type=int,   default=1008,   help='Channel size for the speaker encoder')  #Channel size = 1008 only when using model V3 
parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
#parser.add_argument('--n_class', type=int,   default=7,   help='Number of unique emotions')
parser.add_argument('--n_class', type=int,   default=2,   help='Absence or presence of a characteristic')

## Command
parser.add_argument('--eval',    dest='eval', action='store_true', help='Only do evaluation')

## Initialization
warnings.simplefilter("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
args = parser.parse_args()
args = init_args(args)

#dataset_list_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices", "competence_train_list_one_hot_sentencev3_aug_edited.txt")
dataset_list_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices", "train_list_one_hot_sentencev3_aug_edited.txt")
#dataset_list_path = os.path.join(args.path, "savee_data_list.txt") # for savee dataset
#dataset_list_path = os.path.join(args.path, "ravdess_data_list.txt") # for ravdess dataset

#dataset_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices/dataset_1/wav/mixed_chunk_sentence_test")
dataset_path = os.path.join(args.path, args.speech_type, "Amazon+Google_voices/dataset_1/wav/mixed_chunk_sentence_16kHz")
#dataset_path = os.path.join(args.path)

musan_path = args.musan_path
rir_path = args.rir_path

##Define SummaryWriter from Tensorboard to visualize the acc and loss of the model
writer = SummaryWriter()

## Search for the exist models
modelfiles = glob.glob('%s/model_0*.model'%args.model_save_path)
modelfiles.sort()

## Only do evaluation, the initial_model is necessary
if args.eval == True:
	s = ECAPAModel(**vars(args))
	print("Model %s loaded from previous state!"%args.initial_model)
	s.load_parameters(args.initial_model)
	
	eer, _, FPRs, TPRs, AUC, eval_acc, true_label, pred_label, files = s.eval_network(eval_list = val_file, eval_path = dataset_path)

	with open('result_evaluation´.csv', 'w') as output:
		output.writelines('true_label, pred_label, file_name\n')
		for idx in range(len(pred_label)):
			output.writelines(str(true_label[idx]) + ',' + str(pred_label[idx]) + ',' + str(file_name[idx]) + '\n')  

	print("evaluation accuracy: %2.2f%%"%(eval_acc))
	
	quit()

## If initial_model is exist, system will train from the initial_model
if args.initial_model != "":
	print("Model %s loaded from previous state!"%args.initial_model)
	s = ECAPAModel(**vars(args))
	s.load_parameters(args.initial_model)
	epoch = 1

## Otherwise, system will try to start from the saved model&epoch
elif len(modelfiles) >= 1:
	print("Model %s loaded from previous state!"%modelfiles[-1])
	epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
	s = ECAPAModel(**vars(args))
	s.load_parameters(modelfiles[-1])
## Otherwise, system will train from scratch
else:
	epoch = 1
	s = ECAPAModel(**vars(args))

EERs, FPRs, TPRs = [], [], []
eer = 0.0
AUC = 0.0

label_0_count = 0
label_1_count = 0

score_file = open(args.score_save_path, "a+")
		
# Read the lines from the file into a list
with open('../dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/train_list_one_hot_sentencev3_aug_edited.txt', 'r') as file:
#with open('../dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/competence_train_list_one_hot_sentencev3_aug_edited.txt', 'r') as file:
#with open('../dataset/SAVEE/AudioData/savee_data_list.txt', 'r') as file:
#with open('../dataset/SAVEE/AudioData/ravdess_data_list.txt', 'r') as file:
    lines = file.readlines()

# Shuffle the list
random.shuffle(lines)

# Write the shuffled list back into the file
with open('../dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/train_list_one_hot_sentencev3_aug_edited.txt', 'w') as file:
#with open('../dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/competence_train_list_one_hot_sentencev3_aug_edited.txt', 'w') as file:
#with open('../dataset/SAVEE/AudioData/savee_data_list.txt', 'w') as file:
#with open('../dataset/SAVEE/AudioData/ravdess_data_list.txt', 'w') as file:
    file.writelines(lines)

"""
#with open('../dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/train_list_one_hot_sentencev3_aug_edited.txt', "r") as file:
#with open('../dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/competence_train_list_one_hot_sentencev3_aug_edited.txt', "r") as file:
    for line in file:
	# Extract the label from the line
        label = int(line.split()[0])

        # Increment the respective counter based on the label
        if label == 0:
            label_0_count += 1
        elif label == 1:
            label_1_count += 1

    # Print the counts
    print("Number of data with label '0':", label_0_count)
    print("Number of data with label '1':", label_1_count)
"""

data = open(dataset_list_path).read().splitlines()
print (data)
print (len(data))

#TODO:Try leave one speaker out validattion
#loo = LeaveOneOut()
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Initialize variables to track the best model
best_fold = None  # To track which fold has the best model
best_epoch = None # To track epoch with the highest validation accuracy 
best_val_acc = float('-inf')  # To track the highest validation accuracy observed between fold
best_model_state = None  # To store the best model state dictionary across all folds
train_accuracies = [] # To store the best training accuracy for each fold
fold_accuracies = []  # To store the best validation accuracy for each fold
val_file = []

#set the initial model
init_model = None
if ("V1" in args.initial_model.split("/")[1]) or ("V3" in args.initial_model.split("/")[1]):
	init_model = args.initial_model.split("/")[1]
else:
	init_model = args.init_model.split("/")[0] + "/" + args.initial_model.split("/")[1]

for fold, (train_idx, val_idx) in enumerate(kf.split(data)):

	print(f"For FOLD {fold}")

	train_file = [data[i] for i in train_idx]
	val_file = [data[i] for i in val_idx] 

	## 10% our of train file as eval file in case split to test file again
	#train_file, eval_file = train_test_split(train_file, test_size=0.1111, random_state=42)

	with open('train_file.txt', 'w') as f:
		# Iterate over each element in the train_file list
		for item in train_file:
			# Write the element to the file followed by a newline character
			f.write(f"{item}\n")

	with open('test_file.txt', 'w') as f:
		# Iterate over each element in the train_file list
		for item in val_file:
			# Write the element to the file followed by a newline character
			f.write(f"{item}\n")

	print("TRAIN:", train_file)
	print("EVAL:", val_file)
	#print("TEST:",val_file)

	trainloader = train_loader(train_list='train_file.txt', train_path=dataset_path, musan_path=musan_path, rir_path=rir_path, num_frames=args.num_frames)
	trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True, collate_fn = trainloader.custom_collate_fn)
    #trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)

	best_fold_val_acc = float('-inf')  # To track the highest validation accuracy in this fold

	while(1):
	
		## Training for one epoch
		loss, lr, acc = s.train_network(epoch = epoch, loader = trainLoader)
		writer.add_scalars('Loss/train', {'loss': loss}, epoch)
		writer.add_scalars('Accuracy/train', {'accuracy': acc}, epoch)

		## Evaluation every [test_step] epochs
		if epoch % args.test_step == 0:
			eer, _, FPRs, TPRs, AUC, eval_acc, true_label, pred_label, files = s.eval_network(eval_list = val_file, eval_path = dataset_path)
			EERs.append(eer)
			
			# Keep track of the best validation accuracy for this fold
			if eval_acc > best_fold_val_acc:
				with open('metrics/' + init_model + '/' + 'fold%d'%fold + '/' + 'output_fold_%d.txt'%fold, 'w') as f:
					f.writelines('Best model: model_fold_%d_epoch_%d\n'% (fold, epoch))
					f.writelines('true_label, pred_label, file_name\n')
					for idx in range(len(pred_label)):
						f.writelines(str(true_label[idx]) + ',' + str(pred_label[idx]) + ',' + str(files[idx]) + '\n')
				with open('metrics/' + init_model + '/' + 'fold%d'%fold + '/' + 'tpr_fpr_output_fold_%d.txt'%fold, 'w') as f:
					f.writelines('Best model: model_fold_%d_epoch_%d\n'% (fold, epoch))
					for fp, tp in zip(FPRs, TPRs):
						f.writelines(f'{fp}, {tp}\n')
				best_fold_auc = AUC 
				best_fold_val_acc = eval_acc
				best_fold_train_acc = acc
				best_epoch = epoch	
				best_model = s.state_dict()

			print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, EERs[-1], min(EERs)))
			score_file.write("%d epoch, LR %f, LOSS %f, TRAIN_ACC %2.2f%%, EVAL_ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, eval_acc, EERs[-1], min(EERs)))
			score_file.flush()


		if epoch >= args.max_epoch:
			writer.close()
			epoch = 1

			##load model from initial state
			s = ECAPAModel(**vars(args))
			s.load_parameters(args.initial_model)

			break

		epoch += 1

	#plot confusion matrix after getting best epoch from current fold
	file = f"/home/ray/Abschlussarbeit/Raymond/ECAPA/metrics/" + init_model + '/' + 'fold%d'%fold + '/' + "output_fold_%d.txt"%fold
	tpr_fpr_file = f"/home/ray/Abschlussarbeit/Raymond/ECAPA/metrics/" + init_model + '/' + 'fold%d'%fold + '/' + "tpr_fpr_output_fold_%d.txt"%fold
	plot_confusion_matrix(file)
	plot_roc_curve(tpr_fpr_file, best_fold_auc)

	# Save the best model for this fold
	torch.save(best_model, args.model_save_path + f'/best_model_epoch{best_epoch}_fold{fold}.model')
	train_accuracies.append(best_fold_train_acc)
	fold_accuracies.append(best_fold_val_acc)

	# Update the best fold and validation accuracy if the current fold is better
	if best_fold_val_acc > best_val_acc:
		best_val_acc = best_fold_val_acc
		best_fold = fold
		#best_model_state = best_model 

# Give all best validation accuracy on every folds
print(f"Best validation accuracies in every folds: {fold_accuracies}")

# At this point, best_fold holds the index of the fold with the highest validation accuracy
print(f"Best fold: {best_fold} with validation accuracy: {best_val_acc}")

#calculate the average of the training and validation accuracies
avg_train_acc = np.mean([t_acc.cpu().numpy() for t_acc in train_accuracies])

avg_val_acc = np.mean([v_acc.cpu().numpy() for v_acc in fold_accuracies])

print("average train accuracy:", avg_train_acc)
print("average validation accuracy:", avg_val_acc)


	

