import os
import torch
import torchvision
import pandas as pd
import numpy as np
##import matplotlib.pyplot as plt
from sklearn import preprocessing

class preprocess_data:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.result = {}

    def group_norm_means(self, type, gender):

        self.data = pd.read_csv(self.data_path + type + '_ratings.csv')
        ## grouping data
        self.data = self.data[self.data['gender_speaker'] == gender] 
        grouped_data = self.data.groupby(['speaker_' + type + '_sentence'])
        """
        for group_name,group in grouped_data:
            print(f"Group:{group_name}")
            print(group)
        """
        ## normalize grouped data
        for speaker, speaker_data in grouped_data:
            speaker_ratings = speaker_data[['sympathetic_' + type, 'kind_' + type, 'responsible_' + type, 'skillful_' + type]]
            print(speaker_ratings)

            #normalizer = preprocessing.MinMaxScaler(feature_range=(-3,3))
            normalizer = preprocessing.MinMaxScaler(feature_range=(0,1))
            norm_data = normalizer.fit_transform(speaker_ratings)
            #print(norm_data)
            #print(len(norm_data))
            #print(speaker)

            ##calculate means
            mean_data = norm_data.mean(axis=0).tolist()
            #print(mean_data)
            #print(len(mean_data))
            self.result[speaker] = mean_data

            #print(self.result)
        return self.result

if __name__ == "__main__":
    data_path = "/home/ray/Abschlussarbeit/ECAPA-TDNN/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Subjectivedata/"

    preprocess = preprocess_data(data_path)
    res = preprocess.group_norm_means('neutral', 'F')
    print(res)
    print(len(res))
