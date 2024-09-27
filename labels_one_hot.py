import pandas as pd
from preprocess_data import preprocess_data
import matplotlib.pyplot as plt

## Assuming your CSV file is named 'your_data.csv'
#df = pd.read_csv('/home/ray/Abschlussarbeit/ECAPA-TDNN/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Subjectivedata/neutral_ratings.csv')

## data grouping by speaker's gender and normalization



## labeling with one hot encoded 

#df['warmth'] = ((df['sympathetic_neutral'] >= 50) & (df['kind_neutral'] >= 50)).astype(int)
#print((df['sympathetic_neutral'] >= 50) & (df['kind_neutral'] >= 50))
#df['highly competence'] = ((df['responsible_neutral'] >= 50) & (df['skillful_neutral'] >= 50)).astype(int)

## Display the updated DataFrame
#print(df)
#df.to_csv('/home/ray/Abschlussarbeit/ECAPA-TDNN/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Subjectivedata/neutral_ratings.csv', index=False)

class labeling:

    def __init__(self):
        self.pre = preprocess_data("/home/ray/Abschlussarbeit/Raymond/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Subjectivedata/")
        #self.means = self.pre.group_norm_means()
        self.means = self.pre.group_norm_means('neutral', 'M')
        #self.dict = {'00': 0, '01': 1, '10': 2, '11': 3}
        self.dict = {'0': 0, '1': 1}
        self.result = {}

    def create_label(self):
        print(self.means)
        for speaker in self.means:
            print(speaker)
            print(self.means[speaker])
            #label = str(int((self.means[speaker][0] >= 0.5) & (self.means[speaker][1] >= 0.5))) + str(int((self.means[speaker][2] >= 0.5) & (self.means[speaker][3] >= 0.5)))
            print((self.means[speaker][0] + self.means[speaker][1]) / 2)
            #warmth_label = str(int(((self.means[speaker][0] + self.means[speaker][1]) / 2) >= 0.55))
            competence_label = str(int(((self.means[speaker][2] + self.means[speaker][3]) / 2) >= 0.55))
            #self.result[speaker] = warmth_label
            self.result[speaker] = competence_label

            print("result:", self.result)
        return self.result
    
    def create_stat_analysis(self):
        data = {}
        
        for key, value in self.means.items():
            avg_first_two = (value[0] + value[1]) / 2
            avg_last_two = (value[2] + value[3]) / 2
            data[key] = [avg_first_two, avg_last_two]
        print("data:", data)
        
        speakers = []
        for key in data.keys():
            if key.split('_')[0] != 'Wavenet':
                speakers.append(key.split('_')[0])
            else:
                speakers.append('Wavenet_' + key.split('_')[1])
        speakers = set(speakers)
        print(speakers)
        adjectives = ['warmth', 'competence']

        for speaker in speakers:
            speaker_data = {key: value for key, value in data.items() if key.startswith(speaker)}
            #print("speaker data:", speaker_data)
            fig, axs = plt.subplots(1, 2, figsize=(16, 4))
            fig.suptitle(f'{speaker} Speaker')
    
            for i, adj in enumerate(adjectives):
                adj_data = [speaker_data[key][i] for key in speaker_data]
                axs[i].bar(range(len(adj_data)), adj_data, color='skyblue')
                axs[i].set_title(adj.capitalize())
                axs[i].set_xticks(range(len(adj_data)))
                if speaker.split('_')[0] != 'Wavenet':
                    axs[i].set_xticklabels([key.split('_')[1] for key in speaker_data])
                else:
                    axs[i].set_xticklabels([key.split('_')[2] for key in speaker_data])
    
            plt.tight_layout()
            plt.show()

    def create_label_list(self, labels):
        file_path = "/home/ray/Abschlussarbeit/Raymond/dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/competence_male_train_list_one_hot_edited_chunk.txt"

        ## wrtiing to file
        with open(file_path, "w") as file:
            for speaker, label in labels.items():
                if speaker.split('_')[0] == 'Wavenet':
                    #formatted_string = label + ' ' + 'Wavenet_' + speaker.split('_')[1] + '/' + speaker + '\n'
                    #formatted_string = str(self.dict[label]) + ' ' + 'Wavenet_' + speaker.split('_')[1] + '/' + speaker + '\n'
                    formatted_string = str(self.dict[label]) + ' ' + speaker.split('_')[1] + '/' + speaker.split('_')[1] + '_' + speaker.split('_')[2] +'.wav' + '\n'
                else:
                    #formatted_string = label + ' ' + speaker.split('_')[0] + '/' + speaker + '\n'
                    formatted_string = str(self.dict[label]) + ' ' + speaker.split('_')[0] + '/' + speaker + '.wav' +'\n'
                file.write(formatted_string)
        
        print(f"File '{file_path}' created successfully.")

if __name__ == "__main__":
    
    res = labeling()
    res.create_stat_analysis()
    #print(res.create_label())
    labels = res.create_label()
    res.create_label_list(labels)

##print(means)