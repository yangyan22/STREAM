import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "6000000"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_LEVEL"] = "PIX"
seed = 42

from pprint import pprint
from lightning.pytorch import seed_everything
import torch
import random
import numpy as np


 
seed_everything(seed, workers=True)
def setup_seed(seed):
	 
    np.random.seed(seed) 
    random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)   
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   
    
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.manual_seed(seed)
    
    torch.use_deterministic_algorithms(True)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.enabled = False   
    torch.backends.cudnn.benchmark = False  
setup_seed(seed)

from configs.config import parser
from dataset.data_module_multiviews import DataModule
from lightning_tools.callbacks import add_callbacks
from models.STREAM import STREAM
import lightning.pytorch as pl

def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=args.strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )
    args.vision_model = "/data/SJM/swin-base-patch4-window7-224"
    args.llama_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    if args.ckpt_file is not None:
        model = STREAM.load_from_checkpoint(args.ckpt_file, strict=False, args=args)
    else:
        model = STREAM(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()

 
    dataset = "mimic"
    version = "his_bs10_0.2_retr80fuse_fuse1_fuse2_cat_2card"
    savepath = f"./save/{dataset}/{version}"

    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print(f"Folder '{savepath}' created.")
    else:
        print(f"Folder '{savepath}' already exists.")

    args.dataset = "mimic"
    args.annotation = "/data/SJM/mimic_views_ranked_pr_temp_retrieval.json"
    args.base_dir = "/home/mil39/yinanwangbei/MedDataset/mimic-cxr-jpg/files"
    args.cxr_bert_path = '/data/SJM/BiomedVLP-CXR-BERT-specialized'
    args.batch_size = 10
    args.val_batch_size = 36
    args.freeze_vm = False
    args.vis_use_lora = False
    args.llm_use_lora = False
    # args.vis_r = 16
    # args.vis_alpha = 16
    args.savedmodel_path = savepath
    args.gradient_clip_val = 1
    args.max_length = 100
    args.min_new_tokens = 80
    args.max_new_tokens = 120
    args.repetition_penalty = 2.0
    args.length_penalty = 2.0
    args.num_workers = 32
    args.devices = 2
    args.max_epochs = 6
    args.limit_val_batches = 1.0
    args.val_check_interval = 0.2
    args.num_sanity_val_steps = 2
    
    # # testing
    # args.ckpt_file = "/root/autodl-tmp/old_save/mimic/multi_views/checkpoints/checkpoint_epoch1_step33848_bleu0.138186_cider0.250430.pth"
    # args.validate = True

    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(66, workers=True)
    train(args)


if __name__ == '__main__':
    main()