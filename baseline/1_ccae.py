# add python path
import sys
sys.path.append("/".join(sys.path[0].split("/")[:-1]))


from utils import my_parser, seed_util, wb_util, model_util
from utils.eval_shortcut import Cut
from models import my_model
import torch
from loaders import voxceleb_loader_for_cae
from configs.config import face_emb_dict, voice_emb_dict, emb_eva, model_save_folder
import os


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data = data
    loss_emb, loss_dec = model(voice_data, face_data)
    loss = loss_emb + loss_dec
    loss.backward()
    optimizer.step()
    return loss.item(), {}


def train():
    step = 0
    model.train()

    for epo in range(args.epoch):
        wb_util.log({"train/epoch": epo})
        for data in train_iter:
            loss, info = do_step(epo, step, data)
            step += 1
            if step % 50 == 0:
                obj = {
                    "train/step": step,
                    "train/loss": loss,
                }
                obj = {**obj, **info}
                print(obj)
                wb_util.log(obj)

            if step > 0 and step % args.eval_step == 0:
                if eval_cut.eval_short_cut():
                    return


if __name__ == "__main__":
    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder=model_save_folder, early_stop=10)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 250,
        "load_model": "load_model",
    })
    parser.use_wb("sl_project", "CCAE")
    args = parser.parse()
    seed_util.set_seed(args.seed)

    model = my_model.CAE().cuda()

    if args.load_model is not None and os.path.exists(args.load_model):
        tmp_model = my_model.Encoder(shared=True).cuda()
        model_util.load_model(args.load_model, tmp_model, strict=True)
        model.encoder = tmp_model
        model.face_encoder = tmp_model.face_encoder
        model.voice_encoder = tmp_model.voice_encoder

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_iter = voxceleb_loader_for_cae.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size,
                                                  face_emb_dict, voice_emb_dict)
    eval_cut = Cut(emb_eva, model, args)
    train()
