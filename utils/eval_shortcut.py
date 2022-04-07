import os
from utils import wb_util, model_util, pickle_util
from utils import model_selector


class Cut():

    def __init__(self, emb_eva, model, args):
        self.modelSelector = model_selector.ModelSelector()
        self.emb_eva = emb_eva
        self.model = model
        self.args = args

    def eval_short_cut(self):
        emb_eva = self.emb_eva
        model = self.model
        modelSelector = self.modelSelector
        args = self.args

        # 1.do test
        valid_obj = emb_eva.do_valid(model)
        test_obj = emb_eva.do_full_test(model)
        obj = {**valid_obj, **test_obj}

        # 2.log
        wb_util.log(obj)
        modelSelector.log(obj)
        print(obj)

        # 3.init wandb
        wb_util.init(args)

        indicator = "valid/auc"
        if modelSelector.is_best_model(indicator):
            model_util.delete_last_saved_model()
            model_save_name = "auc[%.2f,%.2f]_ms[%.2f,%.2f]_map[%.2f,%.2f].pkl" % (
                obj["valid/auc"] * 100,
                obj["test/auc"] * 100,
                obj["test/ms_v2f"] * 100,
                obj["test/ms_f2v"] * 100,
                obj["test/map_v2f"] * 100,
                obj["test/map_f2v"] * 100,
            )
            model_save_path = os.path.join(args.model_save_folder, args.project, args.name, model_save_name)
            model_util.save_model(0, model, None, model_save_path)
            pickle_util.save_json(model_save_path + ".json", test_obj)
        else:
            print("not best model")

        if modelSelector.should_stop(indicator, args.early_stop):
            print("early_stop!")
            print(model_util.history_array[-1])
            return True
        return False
