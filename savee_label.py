import os, re
import numpy

class savee_label:
    def __init__(self, path):
        super(savee_label, self).__init__()
        self.path = path
        self.speaker = ["DC", "JE", "JK", "KL"]
        self.dict = {"a": 0, "d": 1, "f": 2, "h": 3, "n": 4, "sa": 5, "su": 6} 

    def labeling(self):

        data_list = []
        for s in self.speaker:
            folder = os.path.join(self.path, s)

            if os.path.isdir(folder):
                 for roots, dirs, files in os.walk(folder):
                    for file in files:
                        file_split = list(filter(None, re.split(r'\d+', file)))
                        label = self.dict[file_split[0]]
                        entry = str(label) + " " + s + "/" + file
                        data_list.append(entry)
        
        with open("/home/ray/Abschlussarbeit/Raymond/dataset/SAVEE/AudioData/savee_data_list.txt", 'w') as f:
            for entry in data_list:
                f.write(entry + "\n")


if __name__ == "__main__":

    savee_init = savee_label("/home/ray/Abschlussarbeit/Raymond/dataset/SAVEE/AudioData")

    savee_init.labeling()