import torch
from utils import pickle_util, sample_util, worker_util
from torch.utils.data import DataLoader


def get_iter(batch_size, full_length, name2face_emb, name2voice_emb):
    train_iter = DataLoader(DataSet(name2face_emb, name2voice_emb, full_length),
                            batch_size=batch_size, shuffle=False, pin_memory=True, worker_init_fn=worker_util.worker_init_fn)
    return train_iter


class DataSet(torch.utils.data.Dataset):

    def __init__(self, name2face_emb, name2voice_emb, full_length):
        self.train_movie_list = pickle_util.read_pickle("./dataset/voxceleb/cluster/train_movie_list.pkl")
        self.movie2wav_path = pickle_util.read_pickle("./dataset/voxceleb/cluster/movie2wav_path.pkl")
        self.movie2jpg_path = pickle_util.read_pickle("./dataset/voxceleb/cluster/movie2jpg_path.pkl")

        self.full_length = full_length
        self.name2face_emb = name2face_emb
        self.name2voice_emb = name2voice_emb

    def __len__(self):
        return self.full_length

    def __getitem__(self, index):
        video = sample_util.random_element(self.train_movie_list)
        img = sample_util.random_element(self.movie2jpg_path[video])
        wav = sample_util.random_element(self.movie2wav_path[video])
        wav, img = self.to_tensor([wav, img])
        return wav, img

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
