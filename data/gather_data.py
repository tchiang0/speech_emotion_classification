import pandas as pd
import re
import os


class DataGatherer():
    def __init__(self):
        pass

    def create_data_df(self, dir_name, emo_map):
        file_emotion = []
        file_path = []

        dir = os.listdir(dir_name)

        for file in dir:
            if dir_name == "CREMA-D":
                file_path.append(dir_name + "/" + file)
                file_tag =  re.split(r'_|\.', file)
                emo = file_tag[2]
                file_emotion.append(emo_map[emo])
            elif dir_name == "RAVDESS":
                if len(file_tag) == 8:
                    emo = file_tag[2]
                    file_path.append(dir_name + "/" + file)
                    file_emotion.append(emo_map[emo])
            elif dir_name == "SAVEE":
                file_tag = re.split(r'_|\.', file)
                emo = re.sub(r'[^a-zA-Z]', '', file_tag[1])
                file_path.append("data/SAVEE/" + file)
                file_emotion.append(emo_map[emo])
            else:
                for folder in dir:
                    if not folder.startswith("."):
                        emo = folder[4:].lower()
                        file_emotion += [emo] * len(os.listdir(dir + "/" + folder))
                        file_path += [dir + "/" + folder + "/" + file for file in os.listdir(dir + "/" + folder)]

        d = {'emotion': file_emotion, 'file_path': file_path}
        df = pd.DataFrame(d)
        return df
    

def main():
    dirs = ["CREMA-D", "RAVDESS", "SAVEE", "TESS"]
    cd_emo = {
        'ANG': "angry",
        'DIS': "disgust",
        'FEA': "fear",
        'HAP': "happy",
        'NEU': "neutral",
        'SAD': "sad"
    }
    r_emo = {
        '01': "neutral",
        '02': "calm",
        '03': "happy",
        '04': "sad",
        '05': "angry", 
        '06': "fear", 
        '07': "disgust", 
        '08': "surprise"
    }
    s_emo = {
        'a': "angry",
        'd': "disgust",
        'f': "fear",
        'h': "happy",
        'sa': "sad",
        'n': "neutral",
        'su': "surprise"
    }
    emo_list = [cd_emo, r_emo, s_emo, -1]
    all_data_path = pd.DataFrame()
    data_gatherer = DataGatherer()
    
    for idx in range(len(dirs)):
        dir_name = dirs[idx]
        emo_map = emo_list[idx]
        df = data_gatherer.create_data_df(dir_name, emo_map)
        all_data_path = pd.concat([all_data_path, df], axis=0).reset_index(drop=True)
    
    return all_data_path
