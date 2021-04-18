from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

from torch.utils.data import DataLoader

from exp_kitti_sync.dataloader_kitti import KittiDataset

import networks


from layers import *
from networks import *

import argparse

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",                  type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                    type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",         type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--split",                      type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",                  type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",                  type=float, default=100.0,              help="maximum depth")


# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=1,                 help="batch size")
parser.add_argument("--load_weights_folder_depth",  type=str,   default=None,               help="name of models to load")
parser.add_argument("--load_weights_folder_norm",   type=str,   default=None,               help="name of models to load")

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def read_splits_mapping():
    evaluation_entries = []
    for m in range(200):
        seqname = "kittistereo15_{}/kittistereo15_{}_sync".format(str(m).zfill(6), str(m).zfill(6))
        evaluation_entries.append("{} {} {}".format(seqname, "10".zfill(10), 'l'))
    return evaluation_entries

class Trainer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = "cuda"

        self.models["encoder_depth"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder_depth"].to(self.device)
        self.models["depth"] = DepthDecoder(self.models["encoder_depth"].num_ch_enc, num_output_channels=1)
        self.models["depth"].to(self.device)

        self.set_dataset()

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.load_model()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.STEREO_SCALE_FACTOR = 5.4
    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        evaluation_entries = read_splits_mapping()

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, evaluation_entries, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False
        )

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False)

        self.val_num = val_dataset.__len__()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        errors = list()

        import tqdm
        with torch.no_grad():
            for batch_idx, inputs in enumerate(tqdm.tqdm(self.val_loader)):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                _, _, gt_height, gt_width = inputs['depthgt'].shape

                outputs_depth = self.models['depth'](self.models['encoder_depth'](inputs['color']))
                _, pred_depth = disp_to_depth(outputs_depth[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_depth = F.interpolate(pred_depth, [gt_height, gt_width], mode='bilinear', align_corners=True)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
                pred_depth_np = pred_depth.squeeze().cpu().numpy()
                pred_depth_np = np.clip(pred_depth_np, a_min=self.MIN_DEPTH, a_max=self.MAX_DEPTH)

                depthgtnp = inputs['depthgt'].cpu().squeeze().numpy()

                mask = (depthgtnp > self.MIN_DEPTH) * (depthgtnp < self.MAX_DEPTH)
                cropmask = np.zeros_like(mask)
                cropmask[int(0.40810811 * self.opt.crph): int(0.99189189 * self.opt.crph), int(0.03594771 * self.opt.crpw): int(0.96405229 * self.opt.crpw)] = 1
                mask[cropmask == 0] = 0
                mask = mask == 1

                gtnp = depthgtnp[mask]
                prednp = pred_depth_np[mask]

                errors.append(compute_errors(gtnp, prednp))

        mean_errors = np.array(errors).mean(0)
        print("\nCurrent Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    def load_model(self):
        """Load model(s) from disk
        """
        load_depth_folder = os.path.expanduser(self.opt.load_weights_folder_depth)
        assert os.path.isdir(load_depth_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_depth)

        models_to_load = ['encoder_depth', 'depth']
        pthfilemapping = {'encoder_depth': 'encoder', 'depth': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder_depth, "{}.pth".format(pthfilemapping[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
