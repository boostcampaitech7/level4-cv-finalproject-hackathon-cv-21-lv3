# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import wandb

from utils import *
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from runner import Runner
from kd_teacher_model import TeacherModel

def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, default='/data/audiolm-evaluator/audiolm-trainer/configs/train_stage1_kd.yaml', help='path to configuration file')
    parser.add_argument("--dryrun", action='store_true', help='if True, use dummy model and skip forward/backward')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--local_rank", type=int, default=0, help="local rank passed by DeepSpeed launcher")

    return parser.parse_args()

def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def main():    
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # load config
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # Wandb logger
    global_rank = int(os.environ["RANK"])
    if global_rank == 0:
        wandb.login()
        wandb.init(project="audio_lm", name=run_config.exp_name)

    # print config
    cfg.pretty_print()

    # build datasets
    datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path),
        "test": SALMONNDataset(data_config.prefix, data_config.test_ann_path, data_config.whisper_path),
    }
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    # Load teacher model
    teacher_model = TeacherModel(device=device)
    teacher_model.to(device)
    teacher_model.half()
    teacher_model.eval()

    # Load student model
    if not args.dryrun:
        student_model = load_model(model_config, pc_lora=False)
        student_model.to(device)
        student_model.half()
    else: # load small dummy language model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)

    # Runner setup
    runner = Runner(cfg, student_model, datasets, job_id=now(), dryrun=args.dryrun, teacher_model=teacher_model, kd=True)

    
    # import deepspeed
    # # Teacher 모델 로드 (SALMONN-7B)# Teacher 모델 로드
    # teacher_model = TeacherModel()

    # # ✅ 1. 모델을 Half Precision으로 변환 (fp16 적용)
    # teacher_model.to(torch.device("cuda"))
    # teacher_model.half()  # ✅ 메모리 절약을 위해 half precision 사용
    # teacher_model.eval()  # ✅ Inference 모드

    # # --- Deepspeed 초기화 (student 모델만) ---
    # student_model = load_model(model_config)
    # # --- Runner 객체를 먼저 생성하여 optimizer를 가져오기 ---
    # runner = Runner(cfg, student_model, datasets, job_id, args.dryrun, teacher_model, kd=True)

    # # ✅ 기존 Runner에서 설정한 optimizer 가져오기
    # optimizer = runner.optimizer

    # if run_config.get("use_deepspeed", False):
    #     # ✅ 기존 ds_config 사용
    #     ds_config = {
    #         "train_batch_size": max(4, torch.cuda.device_count() * 2),
    #         "gradient_accumulation_steps": 2,
    #         "fp16": {"enabled": True},
    #         "zero_optimization": {
    #             "stage": 2,  # ✅ Student Model(1B)에는 Stage 2가 적절
    #             "offload_optimizer": {"device": "cpu", "pin_memory": True},
    #             "allgather_partitions": True,
    #             "allgather_bucket_size": 5e8,
    #             "overlap_comm": True,
    #             "reduce_scatter": True,
    #             "reduce_bucket_size": 5e8,
    #         },
    #     }

    #     # ✅ DeepSpeed 초기화 (기존 Runner의 optimizer를 함께 사용)
    #     student_model, optimizer, _, _ = deepspeed.initialize(
    #         model=student_model,
    #         model_parameters=student_model.parameters(),
    #         config=ds_config,
    #         optimizer=optimizer,  # ✅ Runner에서 정의한 Optimizer를 사용
    #         model_parallel_size=2
    #     )

    # # 만약 Deepspeed를 사용하지 않으면 기존 방식 유지
    # else:
    #     student_model.to(run_config.device)
    # train
    print('################## Start Training ##################')
    runner.train()


if __name__ == "__main__":
    main()