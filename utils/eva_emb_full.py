import torch
from torch.utils.data import DataLoader
from utils import distance_util, path_util
from utils import map_evaluate
from sklearn.metrics import roc_auc_score
import scipy.spatial
import numpy as np
import collections
from utils import pickle_util


class EmbEva:

    def __init__(self,
                 voice_embedding_dict,
                 face_embedding_dict,
                 batch_size=512,
                 ):
        self.voice_embedding_dict = voice_embedding_dict
        self.face_embedding_dict = face_embedding_dict
        self.batch_size = batch_size

    def do_valid(self, model):
        obj = {"valid/auc": self.do_verification(model, "./dataset/voxceleb/eval/valid_verification.pkl")}
        return obj

    def do_full_test(self, model):
        obj = {}
        # 1.verification
        obj["test/auc"] = self.do_verification(model, "./dataset/voxceleb/eval/test_verification.pkl")
        obj["test/auc_g"] = self.do_verification(model, "./dataset/voxceleb/eval/test_verification_g.pkl")

        # 2.retrieval
        obj["test/map_v2f"], obj["test/map_f2v"] = self.do_retrival(model, "./dataset/voxceleb/eval/test_retrieval.pkl")

        # 3.matching
        obj["test/ms_v2f"], obj["test/ms_f2v"] = self.do_matching(model, "./dataset/voxceleb/eval/test_matching.pkl")
        obj["test/ms_v2f_g"], obj["test/ms_f2v_g"] = self.do_matching(model, "./dataset/voxceleb/eval/test_matching_g.pkl")
        return obj

    def do_1_N_matching(self, model):
        data = pickle_util.read_pickle(path_util.look_up("./dataset/voxceleb/eval/test_matching_10.pkl"))
        v2emb, f2emb = self.to_emb_dict(model, data["jpg_set"], data["wav_set"])
        key2emb = {**v2emb, **f2emb}
        ans = {}
        ans["v2f"] = handle_1_n(data["match_list"], is_v2f=True, key2emb=key2emb)
        ans["f2v"] = handle_1_n(data["match_list"], is_v2f=False, key2emb=key2emb)
        return ans

    def do_matching(self, model, pkl_path):
        data = pickle_util.read_pickle(pkl_path)
        v2emb, f2emb = self.to_emb_dict(model, data["jpg_set"], data["wav_set"])
        ms_vf, ms_fv = calc_ms(data["match_list"], v2emb, f2emb)
        return ms_vf, ms_fv

    def do_verification(self, model, pkl_path):
        data = pickle_util.read_pickle(pkl_path)
        v2emb, f2emb = self.to_emb_dict(model, data["jpg_set"], data["wav_set"])
        return calc_vrification(data["list"], v2emb, f2emb)

    def do_retrival(self, model, pkl_path):
        data = pickle_util.read_pickle(pkl_path)
        v2emb, f2emb = self.to_emb_dict(model, data["jpg_set"], data["wav_set"])
        map_vf, map_fv = calc_map_value(data["retrieval_lists"], v2emb, f2emb)
        return map_vf, map_fv

    def to_emb_dict(self, model, all_jpg_set, all_wav_set):
        model.eval()
        batch_size = self.batch_size
        image_loader = DataLoader(DataSet(list(all_jpg_set), self.face_embedding_dict), batch_size=batch_size, shuffle=False, pin_memory=True)
        voice_loader = DataLoader(DataSet(list(all_wav_set), self.voice_embedding_dict), batch_size=batch_size, shuffle=False, pin_memory=True)
        f2emb = get_path2emb(image_loader.dataset.data, model.face_encoder, image_loader)
        v2emb = get_path2emb(voice_loader.dataset.data, model.voice_encoder, voice_loader)
        model.train()
        return v2emb, f2emb


def calc_ms(all_data, v2emb, f2emb):
    voice1_emb = []
    voice2_emb = []
    face1_emb = []
    face2_emb = []

    for name1, voice1, face1, name2, voice2, face2 in all_data:
        voice1_emb.append(v2emb[voice1])
        voice2_emb.append(v2emb[voice2])
        face1_emb.append(f2emb[face1])
        face2_emb.append(f2emb[face2])

    voice1_emb = np.array(voice1_emb)
    voice2_emb = np.array(voice2_emb)
    face1_emb = np.array(face1_emb)
    face2_emb = np.array(face2_emb)

    dist_vf1 = distance_util.parallel_distance_cosine_based_distance(voice1_emb, face1_emb)
    dist_vf2 = distance_util.parallel_distance_cosine_based_distance(voice1_emb, face2_emb)
    dist_fv1 = distance_util.parallel_distance_cosine_based_distance(face1_emb, voice1_emb)
    dist_fv2 = distance_util.parallel_distance_cosine_based_distance(face1_emb, voice2_emb)

    vf_result = dist_vf1 < dist_vf2
    fv_result = dist_fv1 < dist_fv2
    ms_vf = np.mean(vf_result)
    ms_fv = np.mean(fv_result)

    obj = {
        "dist_vf1": dist_vf1,
        "dist_vf2": dist_vf2,
        "dist_fv1": dist_fv1,
        "dist_fv2": dist_fv2,
        "test_data": all_data,  # name1, voice1, face1, name2, voice2, face2
        "result_fv": fv_result,
        "result_vf": vf_result,
        "score_vf": ms_vf,
        "score_fv": ms_fv,
    }
    return ms_vf, ms_fv


def calc_map_value(retrieval_lists, v2emb, f2emb):
    tmp_dic = collections.defaultdict(list)
    for arr in retrieval_lists:
        map_vf, map_fv = calc_map_recall_at_k(arr, v2emb, f2emb)
        tmp_dic["map_vf"].append(map_vf)
        tmp_dic["map_fv"].append(map_fv)
    map_fv = np.mean(tmp_dic["map_fv"])
    map_vf = np.mean(tmp_dic["map_vf"])
    return map_vf, map_fv


class DataSet(torch.utils.data.Dataset):

    def __init__(self, data, path2emb):
        self.data = data
        self.path2emb = path2emb

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        short_path = self.data[index]
        data = self.path2emb[short_path]
        data = torch.FloatTensor(data)
        return data, index


def handle_1_n(match_list, is_v2f, key2emb):
    tmp_dict = collections.defaultdict(list)
    for voices, faces in match_list:
        if is_v2f:
            prob = voices[0]
            gallery = faces
        else:
            prob = faces[0]
            gallery = voices

        # 1. to vector
        prob_vec = np.array([key2emb[prob]])
        gallery_vec = np.array([key2emb[i] for i in gallery])

        # 2. calc similarity
        distances = scipy.spatial.distance.cdist(prob_vec, gallery_vec, 'cosine')
        distances = distances.squeeze()
        assert len(distances) == len(gallery_vec)

        # 3. get results of 2~N matching
        for index in range(2, len(gallery) + 1):
            arr = distances[:index]
            is_correct = int(np.argmin(arr) == 0)
            tmp_dict[index].append(is_correct)

    for key, arr in tmp_dict.items():
        tmp_dict[key] = np.mean(arr)
    return tmp_dict


#

def get_path2emb(all_path_list, encoder, loader):
    f2emb = {}
    for data, path_indexes in loader:
        emb_batch = encoder(data.cuda()).detach().cpu().numpy()
        path_indexes = path_indexes.detach().cpu().numpy()
        for p_index, emb in zip(path_indexes, emb_batch):
            the_path = all_path_list[p_index]
            f2emb[the_path] = emb

    return f2emb


def cosine_similarity(a, b):
    assert len(a.shape) == 2
    assert a.shape == b.shape

    ab = np.sum(a * b, axis=1)
    # (batch_size,)

    a_norm = np.sqrt(np.sum(a * a, axis=1))
    b_norm = np.sqrt(np.sum(b * b, axis=1))
    cosine = ab / (a_norm * b_norm)
    # [-1,1]
    prob = (cosine + 1) / 2.0
    return prob


def calc_vrification(the_list, v2emb, f2emb):
    voice_emb = np.array([v2emb[tup[0]] for tup in the_list])
    face_emb = np.array([f2emb[tup[1]] for tup in the_list])
    real_label = np.array([tup[2] for tup in the_list])

    # AUC
    prob = cosine_similarity(voice_emb, face_emb)
    auc = roc_auc_score(real_label, prob)
    return auc


def calc_map_recall_at_k(all_data, v2emb, f2emb):
    # 1.get embedding
    labels = []
    v_emb_list = []
    f_emb_list = []
    for v, f, name in all_data:
        labels.append(name)
        v_emb_list.append(v2emb[v])
        f_emb_list.append(f2emb[f])

    v_emb_list = np.array(v_emb_list)
    f_emb_list = np.array(f_emb_list)

    # 2. calculate distance
    vf_dist = scipy.spatial.distance.cdist(v_emb_list, f_emb_list, 'cosine')
    fv_dist = vf_dist.T

    # 3.map value
    map_vf = map_evaluate.fx_calc_map_label_v2(vf_dist, labels)
    map_fv = map_evaluate.fx_calc_map_label_v2(fv_dist, labels)
    return map_vf, map_fv


def calc_ms_f2v(all_data, v2emb, f2emb):
    voice1_emb = []
    voice2_emb = []
    face1_emb = []

    for face1, voice1, voice2 in all_data:
        voice1_emb.append(v2emb[voice1])
        voice2_emb.append(v2emb[voice2])
        face1_emb.append(f2emb[face1])

    voice1_emb = np.array(voice1_emb)
    voice2_emb = np.array(voice2_emb)
    face1_emb = np.array(face1_emb)

    dist_fv1 = distance_util.parallel_distance_cosine_based_distance(face1_emb, voice1_emb)
    dist_fv2 = distance_util.parallel_distance_cosine_based_distance(face1_emb, voice2_emb)

    fv_result = dist_fv1 < dist_fv2
    ms_fv = np.mean(fv_result)
    return ms_fv


def calc_ms_v2f(all_data, v2emb, f2emb):
    voice1_emb = []
    face1_emb = []
    face2_emb = []

    for voice1, face1, face2 in all_data:
        voice1_emb.append(v2emb[voice1])
        face1_emb.append(f2emb[face1])
        face2_emb.append(f2emb[face2])

    voice1_emb = np.array(voice1_emb)
    face1_emb = np.array(face1_emb)
    face2_emb = np.array(face2_emb)

    dist_vf1 = distance_util.parallel_distance_cosine_based_distance(voice1_emb, face1_emb)
    dist_vf2 = distance_util.parallel_distance_cosine_based_distance(voice1_emb, face2_emb)

    vf_result = dist_vf1 < dist_vf2
    ms_vf = np.mean(vf_result)
    return ms_vf
