# Boostcamp AI Tech 7 CV 21
 
## 오디오 경량화 모델 레시피 탐구 

![Image](https://github.com/user-attachments/assets/b459fb77-22fa-4101-8c07-7e786f409e66)
## Description
 오늘날 언어 모델은 audio adapter와의 결합 및 사전 학습을 통해, 음성, 음악, 환경음 등 여러 소리를 이해하고, 다양한 downstream task를 수행하도록 발전했다. 그러나 위와 같은 ALM (Audio Language Model)은 일반적으로 LLM(Large Language Model) 기반으로 작동하기에 높은 연산량이 발생하고, 이로 인해 VRAM 크기가 작은 전형적인 디바이스 환경에서는 사용하기 어렵다. 이 문제를 해결하기 위해 본 해커톤 프로젝트에서는 SALMONN(Speech Audio Language Music Open Neural Network) 아키텍쳐 기반의 baseline 모델 성능을 유지하되, 더 작고 빠른 속도의 연산이 가능하도록 경량화하는 레서피를 탐색하고자 한다

## 평가 지표

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/c952d775-7bc6-43e9-915c-fc2f2a71764b" alt="Dice Coefficient" style="width: 90%; height: auto;">
</div>

## Contributor
| <img src="https://github.com/user-attachments/assets/a669d334-7820-4e28-8a05-5a9d745ddc42" alt="박동준" style="width:100px; height:100px;"> | <a href="https://github.com/Ahn-latte"><img src="https://avatars.githubusercontent.com/Ahn-latte" alt="안주형" style="width:100px; height:100px;"></a> | <a href="https://github.com/minseokheo"><img src="https://avatars.githubusercontent.com/minseokheo" alt="허민석" style="width:100px; height:100px;"></a> | <a href="https://github.com/leedoyoung6"><img src="https://avatars.githubusercontent.com/leedoyoung6" alt="이도영" style="width:100px; height:100px;"></a> | <a href="https://github.com/MinSeok1204"><img src="https://avatars.githubusercontent.com/MinSeok1204" alt="최민석" style="width:100px; height:100px;"></a> | <a href="https://github.com/airacle100"><img src="https://avatars.githubusercontent.com/airacle100" alt="윤정우" style="width:100px; height:100px;"></a> |
| ---------------------------------------------------- | ------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| [박동준](https://github.com/Poodlee)                                               | [안주형](https://github.com/Ahn-latte)                                                   | [허민석](https://github.com/minseokheo)                                              | [이도영](https://github.com/leedoyoung6)                                                  | [최민석](https://github.com/MinSeok1204)             | [윤정우](https://github.com/airacle100)             |


## Role
![Image](https://github.com/user-attachments/assets/bc06c013-8f19-4232-acc4-c2e6910c7c21)

## Usage

### Data Preparation
데이터셋을 `data/` 디렉토리에 준비한다.


## Install dependencies
```bash
git clone https://github.com/boostcampaitech7/level4-cv-finalproject-hackathon-cv-21-lv3.git
pip install -r audiolm-trainer/requirements.txt
pip install -r requirements.txt
```

## Evaluate
`salmonn_eval_config.yaml` 에서 데이터셋 경로, 모델 경로 등을 적절히 수정한 후 아래 스크립트를 실행합니다.
```python
python evaluate_salmonn.py --task {asr, aac}
```

위 파일을 실행하면 기본적으로 `submission.csv`가 생성됩니다.

자체적인 평가를 진행하고자 한다면 아래 형식으로 자체 평가용 json 파일을 만들고 평가하고자 하는 task를 인자로 주면 됩니다.
```
{
  "annotation": [
    {
      "testset_id": "any_id_for_test",
      "path": "/path/to/audio_file",
      "task": {asr or audiocaption_v2},
      "test": "Ground truth for sample"
    },
    ...
```

## Validate submission file
```python
python submission_validator.py /path/to/submission.csv
```

위 스크립트는 파일의 형식만 확인하며, 샘플의 개수는 validation하지 않습니다.


## Citation

```
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
