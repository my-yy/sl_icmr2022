from utils import pickle_util, vec_util, path_util
from utils.eva_emb_full import EmbEva

model_save_folder = "./outputs/"

# 1. data input
face_emb_dict = pickle_util.read_pickle(path_util.look_up("./dataset/voxceleb/face_input.pkl"))
voice_emb_dict = pickle_util.read_pickle(path_util.look_up("./dataset/voxceleb/voice_input.pkl"))
vec_util.dict2unit_dict_inplace(face_emb_dict)
vec_util.dict2unit_dict_inplace(voice_emb_dict)

# 2.eval
emb_eva = EmbEva(voice_emb_dict, face_emb_dict)
