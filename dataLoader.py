
'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal
from augmentation import augment_0, augment_1, augment_2
from audiomentations import AddGaussianSNR

class train_loader(object):
	def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
		self.train_path = train_path
		self.num_frames = num_frames
		# Load and configure augmentation files
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
		# Load data & labels
		self.data_list  = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)
			#print(f"DATA: %s, LABEL: %s \n"%(file_name, speaker_label))
	"""
	another implementation of getitem, where an augmentation from audiomentations by applying Gaussian SNR with min_db = 15 and max_db = 30
	and additionally one augmentation from MUSAN and RIR randomly picked, adjust range to get how many random augmentations should be applied

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = soundfile.read(self.data_list[index])	
		#print("DATA IN GETITEM: ", self.data_list[index])	
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)

		audio_mod = None
		augment_files = []

		augment_files.append((torch.FloatTensor(audio[0]), self.data_label[index], sr))
		# Data Augmentation

		##Audiomentations
		audio_mod = audio.astype(numpy.float32)
		augment = AddGaussianSNR(min_snr_in_db=15.0, max_snr_in_db=30.0, p=1.0)
		audio_mod = augment(samples=audio_mod, sample_rate=sr)
		augment_files.append((torch.FloatTensor(audio_mod[0]), self.data_label[index], sr))

		augtype = None
		reminder = None
		##MUSAN and RIR
		for i in range(1):
			while augtype == reminder:
				augtype = random.randint(1,5)
			reminder = augtype
			if augtype == 1: # Reverberation:
				augment_files.append((torch.FloatTensor(self.add_rev(audio)[0]), self.data_label[index], sr))
			elif augtype == 2: # Babble
				augment_files.append((torch.FloatTensor(self.add_noise(audio, 'speech')[0]), self.data_label[index], sr))
			elif augtype == 3: # Music
				augment_files.append((torch.FloatTensor(self.add_noise(audio, 'music')[0]), self.data_label[index], sr))
			elif augtype == 4: # Noise
				augment_files.append((torch.FloatTensor(self.add_noise(audio, 'noise')[0]), self.data_label[index], sr))
			elif augtype == 5: # Television noise
				audio_mod = self.add_noise(audio, 'speech')
				audio_mod = self.add_noise(audio_mod, 'music')
				augment_files.append((torch.FloatTensor(audio_mod[0]), self.data_label[index], sr))
	
		return augment_files
	"""
	
	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = soundfile.read(self.data_list[index])	
		#print("DATA IN GETITEM: ", self.data_list[index])	
		length = self.num_frames * 160 + 240
		if audio.shape[0] <= length:
			shortage = length - audio.shape[0]
			audio = numpy.pad(audio, (0, shortage), 'wrap')
		start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
		audio = audio[start_frame:start_frame + length]
		audio = numpy.stack([audio],axis=0)

		augment_files = []
		# Data Augmentation

		##MUSAN and RIR
		#augtype = random.randint(0,5)
		#if augtype == 0:   # Original
		#	audio = audio
		augment_files.append((torch.FloatTensor(audio[0]), self.data_label[index], sr))
		#elif augtype == 1: # Reverberation
		#	audio = self.add_rev(audio)
		augment_files.append((torch.FloatTensor(self.add_rev(audio)[0]), self.data_label[index], sr))
		#elif augtype == 2: # Babble
		#	audio = self.add_noise(audio, 'speech')
		augment_files.append((torch.FloatTensor(self.add_noise(audio, 'speech')[0]), self.data_label[index], sr))
		#elif augtype == 3: # Music
		#	audio = self.add_noise(audio, 'music')
		augment_files.append((torch.FloatTensor(self.add_noise(audio, 'music')[0]), self.data_label[index], sr))
		#elif augtype == 4: # Noise
		#	audio = self.add_noise(audio, 'noise')
		augment_files.append((torch.FloatTensor(self.add_noise(audio, 'noise')[0]), self.data_label[index], sr))
		#elif augtype == 5: # Television noise
		#	audio = self.add_noise(audio, 'speech')
		#	audio = self.add_noise(audio, 'music')
		audio = self.add_noise(audio, 'speech')
		audio = self.add_noise(audio, 'music')
		augment_files.append((torch.FloatTensor(audio[0]), self.data_label[index], sr))

	
		return augment_files
	
	@staticmethod
	def custom_collate_fn(batch):
		flattened_batch = [item for sublist in batch for item in sublist]
		random.shuffle(flattened_batch)
		data, labels, srs = zip(*flattened_batch)
		data = torch.stack(data, axis=0)
		labels = torch.LongTensor(labels)
		
		#print("LABELS: ", labels)
		#print("SRS: ", srs)

		return data, labels, srs

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio
