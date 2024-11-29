'''
This part is used to train the speaker model and evaluate the performances
'''

import random, numpy
import torch, sys, os, tqdm, numpy, soundfile, time, pickle, tempfile
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tools import *
from loss import AAMsoftmax
from loss_fcn import FCNClassifier
from model_v105 import ECAPA_TDNN # for model V105
#from model_v125 import ECAPA_TDNN # for model V125
#from model_v130 import ECAPA_TDNN # for model V130
#from model_v21 import ECAPA_TDNN # for model V2.1
#from model_v22 import ECAPA_TDNN # for model V2.2
#from model_v3 import ECAPA_TDNN # for model V3

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class ECAPAModel(nn.Module):
	def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
		super(ECAPAModel, self).__init__()
		## ECAPA-TDNN
		self.speaker_encoder = ECAPA_TDNN(C = C).cuda()

		## HERE FREEZING THE PARAMETER FOR OTHER LAYERS, EXCEPT FC LAYER
		#print("Start freezing layers...")
		#for name, param in self.speaker_encoder.named_parameters():
			#if "fc6" not in name:
		#	param.requires_grad = False
		#print("Complete freezing...")

		## Classifier
		self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()
		self.fcn_loss = FCNClassifier(n_class = n_class).cuda()

		#self.criterion = nn.CrossEntropyLoss()
		self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
		print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1024 / 1024))

	def train_network(self, epoch, loader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']

		print("entering training loop, checking dataloader...")
		if len(loader) == 0:
			print("Loader is empty")
		else:
			for num, (data, labels, sr) in enumerate(loader, start = 1):
				print("Execute training")
							
				self.zero_grad()
				labels            = torch.LongTensor(labels).cuda()

				print("This is the labels in ECAPAModel:", labels)
				#print("This is the sr in ECAPAModel:", sr)
				#print("This is the data in ECAPAModel:", data)

				speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)
				print("Speaker embedding size:", speaker_embedding.size())

				## checking speaker_embedding
				#print("This is the speaker embedding:", speaker_embedding)

				#nloss, prec, logits     = self.speaker_loss.forward(speaker_embedding, labels)
				nloss, prec, logits     = self.fcn_loss.forward(speaker_embedding, labels)			
				nloss.backward()
				self.optim.step()

				#print("Scores in training:", logits)
				_, predicted_labels = torch.max(logits, dim=1)
				print("predicted labels:", predicted_labels)

				index += len(labels)
				top1 += prec
				loss += nloss.detach().cpu().numpy()
				sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
				" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
				" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
				sys.stderr.flush()
				#print("exiting training loop")
			sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list, eval_path):
		self.eval()
		files = []
		embeddings = []

		#predicted_emotions ={}
		#true_label = []
		pred_label = []

		scores, labels  = [], []
		#FPRs, TPRs = [], []
		index, top1_eval = 0, 0
	
		lines = eval_list
		#print("This is lines in eval_network", lines)

		for line in lines:
			files.append(line.split()[1])
			labels.append(int(line.split()[0]))
		
		#print(files)
		#print(labels)

		# Create a dictionary mapping each filename to its index in the original list
		index_dict = {filename: index for index, filename in enumerate(files)}

		setfiles = list(set(files))
		setfiles.sort(key=lambda x: index_dict[x])

		for idx, file in tqdm.tqdm(enumerate(setfiles), total = len(setfiles)):
			audio, _  = soundfile.read(os.path.join(eval_path, file))
			print(file)
			print("True label in evluation:", labels[idx])
			# Full utterance
			data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

			# Splitt0ed utterance matrix
			max_audio = 300 * 160 + 240
			if audio.shape[0] <= max_audio:
				shortage = max_audio - audio.shape[0]
				audio = numpy.pad(audio, (0, shortage), 'wrap')
			feats = []
			startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=1)
			#startframe = numpy.int64(random.random()*(audio.shape[0]-max_audio))
			#print("startframe in evaluation:", startframe)
			for asf in startframe:
				feats.append(audio[int(asf):int(asf)+max_audio])
			#audio = audio[startframe:startframe + max_audio]
			feats = numpy.stack(feats, axis = 0).astype(numpy.float)
			#feat = numpy.stack([audio],axis=0)
			data_2 = torch.FloatTensor(feats).cuda()
			#new_data_2 = torch.unsqueeze(data_2, 0)
			
			#print(new_data_2.size())

			# Speaker embeddings
			with torch.no_grad():
				embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1)
				#print(embedding_2)
				#print("true_label", labels)
				#_, prec_eval, logits = self.speaker_loss.forward(embedding_1, labels[idx])
				#_, prec_eval, logits = self.speaker_loss.forward(embedding_2, labels[idx])
				_, prec_eval, logits = self.fcn_loss.forward(embedding_1, labels[idx])
				#_, prec_eval, logits = self.fcn_loss.forward(embedding_2, labels[idx])
				#print("Scores in evaluation:", logits)
				_, predicted_emotion_eval = torch.max(logits, dim=1)
				#max_prob = F.softmax(logits, dim=1)
				pred_label.append(predicted_emotion_eval.item())

			index += len(labels)
			top1_eval += prec_eval

			print("calculated scores in evaluation:", logits)
			print("predicted label in evaluation:", predicted_emotion_eval)	
			#scores.append(max_prob[:, 1].cpu())
			scores.append(logits[:, 1].cpu())

		print("calculated scores:", scores)

		# Coumpute EER and minDCF
		#EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		tunedThreshold, EER, fpr, fnr, tpr, roc_auc = tuneThresholdfromScore(scores, labels, [1, 0.1])
		#FPRs.append(fpr)
		#TPRs.append(tpr)
		
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		#print("fnrs:", fnrs)
		#print("fprs:", fprs)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		#return EER, minDCF, FPRs, TPRs, roc_auc, top1_eval/index*len(labels), labels, pred_label, files
		return EER, minDCF, fpr, tpr, roc_auc, top1_eval/index*len(labels), labels, pred_label, files

	def save_parameters(self, path):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			#print("This is name %s and origname %s in for loop:"%(name, origname))
			#print("This is the size comparison of name %s and origname %s"%(self_state[name].size(), loaded_state[origname].size()))
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
				
			#print("This is name %s and origname %s:"%(name, origname))


			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)