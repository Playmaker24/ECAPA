from sklearn.manifold import TSNE
from model_v105 import ECAPA_TDNN
from model_v125 import ECAPA_TDNN
from model_v130 import ECAPA_TDNN
from model_v21 import ECAPA_TDNN
from model_v22 import ECAPA_TDNN
from model_v3 import ECAPA_TDNN
from dataLoader import train_loader

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

class TSNEVisualizer:
	def __init__(self, loader, model):
		self.loader = loader
		self.model = model

	def extract_features(self):
		embeddings = []
		labels_list = []

		for num, (data, labels, sr) in enumerate(self.loader, start=1):
			print("Inside loop of extract_feature")
            
			# Forward pass to get the speaker embeddings
			speaker_embedding = self.model.forward(data.cuda(), aug=True)
			print("Speaker embedding size:", speaker_embedding.size())
            
			# Store embeddings and labels
			embeddings.append(speaker_embedding.cpu().detach().numpy())
			labels_list.append(labels.cpu().detach().numpy())

		# Stack embeddings and labels to create single arrays
		embeddings = np.vstack(embeddings)
		labels_list = np.hstack(labels_list)
        
		print("finish extracting feature.....")
		return embeddings, labels_list

	"""
    def_extract_feature(self):
        for num, (data, labels, sr) in enumerate(self.loader, start = 1):
			print("inside loop of extract_feature")
									
			self.zero_grad()
			labels            = torch.LongTensor(labels).cuda()

			print("This is the labels in extract features:", labels)
			#print("This is the sr in ECAPAModel:", sr)
			#print("This is the data in ECAPAModel:", data)
			#print("This is data shape:", data.shape)

			speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug = True)
			print("Speaker embedding size:", speaker_embedding.size())
	"""

	def viz(self):
		embeddings, labels = self.extract_features()
        
		# Apply t-SNE
		tsne = TSNE(n_components=2, random_state=42)
		reduced_embeddings = tsne.fit_transform(embeddings)
        
		# Plot the reduced embeddings
		plt.figure(figsize=(10, 8))
		scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='coolwarm')
		
		"""
		plt.legend(*scatter.legend_elements(), title="Classes")
		plt.title('t-SNE Visualization of Speaker Embeddings Social Speaker')
		plt.show()
		"""
		# Manually create legend
		unique_labels = np.unique(labels)
		handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.coolwarm(i/len(unique_labels)), markersize=10) for i in range(len(unique_labels))]
		plt.legend(handles, unique_labels, title="Classes")

		plt.title('t-SNE Visualization of Speaker Embeddings')
		plt.show()

if __name__ == "__main__":

	musan_path = "/home/ray/Abschlussarbeit/ECAPA-TDNN/dataset/musan_split"
	rir_path = "/home/ray/Abschlussarbeit/ECAPA-TDNN/dataset/rirs_noises/RIRS_NOISES/simulated_rirs"
	dataset_path = "/home/ray/Abschlussarbeit/Raymond/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/dataset_1/wav/mixed_chunk_sentence_test"

	loader = train_loader(train_list='train_file.txt', train_path=dataset_path, musan_path=musan_path, rir_path=rir_path, num_frames=200)
	loader = torch.utils.data.DataLoader(loader, batch_size = 16, shuffle = True, num_workers = 4, drop_last = True, collate_fn = loader.custom_collate_fn)

	visualizer = TSNEVisualizer(loader, ECAPA_TDNN(C = 1024).cuda())
	visualizer.viz()
