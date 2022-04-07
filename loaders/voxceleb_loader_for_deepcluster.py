import numpy as np
import torch
from utils import pickle_util, sample_util, worker_util, vec_util
from torch.utils.data import DataLoader


def get_iter(batch_size, full_length, name2face_emb, name2voice_emb, movie2label):
    train_iter = DataLoader(DataSet(name2face_emb, name2voice_emb, full_length, movie2label),
                            batch_size=batch_size, shuffle=False, pin_memory=True, worker_init_fn=worker_util.worker_init_fn)
    return train_iter


class DataSet(torch.utils.data.Dataset):

    def __init__(self, name2face_emb, name2voice_emb, full_length, movie2label):
        self.train_movie_list = list(movie2label.keys())

        self.movie2wav_path = pickle_util.read_pickle("./dataset/voxceleb/cluster/movie2wav_path.pkl")
        self.movie2jpg_path = pickle_util.read_pickle("./dataset/voxceleb/cluster/movie2jpg_path.pkl")

        self.full_length = full_length
        self.name2face_emb = name2face_emb
        self.name2voice_emb = name2voice_emb
        self.movie2label = movie2label

    def __len__(self):
        return self.full_length

    def __getitem__(self, index):
        movie = sample_util.random_element(self.train_movie_list)
        label = self.movie2label[movie]

        img = sample_util.random_element(self.movie2jpg_path[movie])
        wav = sample_util.random_element(self.movie2wav_path[movie])
        wav, img = self.to_tensor([wav, img])

        return wav, img, torch.LongTensor([label])

    def to_tensor(self, path_arr):
        ans = []
        for path in path_arr:
            if ".wav" in path:
                emb = self.name2voice_emb[path]
            else:
                emb = self.name2face_emb[path]
            emb = torch.FloatTensor(emb)
            ans.append(emb)
        return ans


class OredredDataSet(torch.utils.data.Dataset):

    def __init__(self, name2face_emb, name2voice_emb):
        self.train_movie_list = pickle_util.read_pickle("./dataset/voxceleb/cluster/train_movie_list.pkl")

        self.movie2wav_path = pickle_util.read_pickle("./dataset/voxceleb/cluster/movie2wav_path.pkl")
        self.movie2jpg_path = pickle_util.read_pickle("./dataset/voxceleb/cluster/movie2jpg_path.pkl")

        self.name2face_emb = name2face_emb
        self.name2voice_emb = name2voice_emb

    def __len__(self):
        return len(self.train_movie_list)

    def __getitem__(self, index):
        movie = self.train_movie_list[index]
        img = np.mean([self.name2face_emb[i] for i in self.movie2jpg_path[movie]], axis=0)
        wav = np.mean([self.name2voice_emb[i] for i in self.movie2wav_path[movie]], axis=0)

        img = vec_util.to_unit_vector(img)
        wav = vec_util.to_unit_vector(wav)
        return torch.FloatTensor(wav), torch.FloatTensor(img), torch.LongTensor([index])


def get_ordered_iter(batch_size, name2face_emb, name2voice_emb):
    train_iter = DataLoader(OredredDataSet(name2face_emb, name2voice_emb),
                            batch_size=batch_size, shuffle=False,
                            pin_memory=True, worker_init_fn=worker_util.worker_init_fn)
    return train_iter
