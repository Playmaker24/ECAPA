import os


class count_label:
    
    def __init__(self, path):
        # Initialize counters
        self.label_0_count = 0
        self.label_1_count = 0
        self.dataset_list = path

    def find_minority(self):
        # Iterate over each line in the dataset
        with open(self.dataset_list, "r") as file:
            for line in file:
            # Extract the label from the line
                label = int(line.split()[0])

            	# Increment the respective counter based on the label
                if label == 0:
                    self.label_0_count += 1
                elif label == 1:
                    self.label_1_count += 1

        # Print the counts
        print("Number of data with label '0':", self.label_0_count)
        print("Number of data with label '1':", self.label_1_count)

        ##-----------------------label filtering-----------------------

        data = open(self.dataset_list).read().splitlines()

        if self.label_0_count > self.label_1_count:
            filtered_list = [d for d in data if d[0] != '0']
        else:
            filtered_list = [d for d in data if d[0] != '1']

        return filtered_list

"""
if __name__ == "__main__":
    #data_path = "dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/train_list_one_hot_sentence_aug_edited.txt"
    data_path = "dataset/Speech_data_forSSC_copy/Speech_data_forSSC_2.0/Speech_data/Neutral_speech/Amazon+Google_voices/train_list_one_hot_chunk_sentence_edited.txt"
    label = count_label(data_path)
    filtered = label.find_minority()

    print(filtered)
"""