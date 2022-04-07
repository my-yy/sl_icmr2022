# add python path
import sys

sys.path.append("/".join(sys.path[0].split("/")[:-1]))

from configs import config
from utils import my_parser, seed_util, wb_util, model_util
from utils.eval_shortcut import Cut
from models import my_model
import torch
from loaders import voxceleb_loader_for_cae
from utils import barlow_loss
import os


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data = data
    v_emb, f_emb = model(voice_data, face_data)
    loss = fun_barlow(v_emb, f_emb)
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
    parser = my_parser.MyParser(epoch=100, batch_size=256, seed=4, model_save_folder=config.model_save_folder, early_stop=10)
    parser.custom({
        "batch_per_epoch": 500,
        "eval_step": 250,
        "load_model": ""
    })
    parser.use_wb("sl_project", "barlow")
    args = parser.parse()
    seed_util.set_seed(args.seed)

    # we found no-shared setting could have better performance
    model = my_model.Encoder(shared=False).cuda()
    if args.load_model is not None and os.path.exists(args.load_model):
        model_util.load_model(args.load_model, model, strict=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    from configs.config import face_emb_dict, voice_emb_dict, emb_eva

    train_iter = voxceleb_loader_for_cae.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size,
                                                  face_emb_dict, voice_emb_dict)
    eval_cut = Cut(emb_eva, model, args)
    fun_barlow = barlow_loss.BarlowTwinsLoss()
    train()
