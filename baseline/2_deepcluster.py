# add python path
import sys
sys.path.append("/".join(sys.path[0].split("/")[:-1]))
from utils import my_parser, seed_util, wb_util, deepcluster_util, model_util, my_softmax_loss
from utils.eval_shortcut import Cut
from models import my_model
from configs import config
import torch
from loaders import voxceleb_loader_for_deepcluster


def do_step(epoch, step, data):
    optimizer.zero_grad()
    data = [i.cuda() for i in data]
    voice_data, face_data, label = data
    v_emb, f_emb = model(voice_data, face_data)
    emb = torch.cat([v_emb, f_emb], dim=0)
    label2 = torch.cat([label, label], dim=0).squeeze()
    loss = fun_loss_metric(emb, label2)
    loss.backward()
    optimizer.step()
    info = {
    }
    return loss.item(), info


def train():
    step = 0
    model.train()

    for epo in range(args.epoch):
        wb_util.log({"train/epoch": epo})
        movie2label, _ = deepcluster_util.do_cluster(ordered_iter, args.ncentroids, model=model, input_emb_type=args.input_emb_type)
        train_iter = voxceleb_loader_for_deepcluster.get_iter(args.batch_size, args.batch_per_epoch * args.batch_size,
                                                              face_emb_dict, voice_emb_dict, movie2label)

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

    parser = my_parser.MyParser(epoch=100, batch_size=256, model_save_folder=config.model_save_folder, early_stop=10)
    parser.custom({
        "ncentroids": 1000,
        "batch_per_epoch": 250,
        "eval_step": 250,
        "input_emb_type": "all",
        "load_model": "",
        "shared": False,
    })
    parser.use_wb("sl_project", "deepcluster")
    args = parser.parse()
    seed_util.set_seed(args.seed)

    # model:
    model = my_model.Encoder(shared=args.shared).cuda()
    import os

    if args.load_model is not None and os.path.exists(args.load_model):
        model_util.load_model(args.load_model, model, strict=True)

    fun_loss_metric = my_softmax_loss.MySoftmaxLoss(128, num_class=args.ncentroids).cuda()
    model_params = list(model.parameters()) + list(fun_loss_metric.parameters())
    optimizer = torch.optim.Adam(model_params, lr=args.lr)

    # 3. loader
    from configs.config import face_emb_dict, voice_emb_dict, emb_eva

    eval_cut = Cut(emb_eva, model, args)
    ordered_iter = voxceleb_loader_for_deepcluster.get_ordered_iter(args.batch_size, face_emb_dict, voice_emb_dict)

    train()
