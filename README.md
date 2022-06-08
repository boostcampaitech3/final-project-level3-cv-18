## 😊 피부 평가 (XAI) 시스템

&nbsp; 
## 🔥 Member 🔥
<table>
  <tr height="125px">
    <td align="center" width="120px">
      <a href="https://github.com/kimkihoon0515"><img src="https://avatars.githubusercontent.com/kimkihoon0515"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ed-kyu"><img src="https://avatars.githubusercontent.com/ed-kyu"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/GwonPyo"><img src="https://avatars.githubusercontent.com/GwonPyo"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ysw2946"><img src="https://avatars.githubusercontent.com/ysw2946"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/jsh0551"><img src="https://avatars.githubusercontent.com/jsh0551"/></a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/YJ0522771"><img src="https://avatars.githubusercontent.com/YJ0522771"/></a>
    </td>

  </tr>
  <tr height="70px">
    <td align="center" width="120px">
      <a href="https://github.com/kimkihoon0515">김기훈_T3019</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ed-kyu">김승규_T3037</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/GwonPyo">남권표_T3072</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ysw2946">유승우_T3130</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/jsh0551">장수호_T3185</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/YJ0522771">조유진_T3208</a>
    </td>
  </tr>
</table>

&nbsp; 

## 🔍Project Overview

## 👩‍🏫개발 환경
- GPU환경 : V100 서버, Google Cloud Platform(GCP) 서버
- 팀 협력 Tools : Notion, Weights&Biases, Github, Google Drive, MLflow
- 개발 Tools : VScode, Jupyter Lab
- 라이브러리 및 프레임워크 : PyTorch, Streamlit

## 🗂️Dataset

- 피부를 유분, 수분, 주름, 색소, 민감도 5가지 항목으로 나누고, 정도에 따라 0부터 4까지 평가한 피부 데이터<img src=".\readme_file\data.jpg" alt="data" style="zoom: 80%;" />
- 각 항목의 일부 라벨이 매우 부족한 imbalance data
- 보안 규정 상 피부 이미지, 데이터 수, 라벨링 기준 등은 공개 불가

## 👨‍🏫평가 Metric

- Penalty Recall

<img src="C:.\readme_file\precall.jpg" alt="precall" style="zoom: 67%;" />

- Ordinal data이기 때문에 정답만 맞추는 것 뿐 아니라 정답과 얼마나 비슷하게 예측했는지도 중요함
- 정답과 인접한 라벨로 예측하면 일정 가중치를 부여한 정답으로 인정하고, 정답과 먼 라벨로 예측하면 패널티를 주어 recall을 계산

## 🧱Structure

<img src=".\readme_file\structure.jpg" alt="structure" style="zoom:50%;" />

* Frontend
  * User의 ID와 password를 입력하면 개인 피부 사진을 업로드할 수 있음
  * 사진을 업로드하면 피부 평가 점수와 gradcam을 제공
  * 평가 점수가 저장되어 피부 변화 추이를 그래프로 확인 가능
* Backend
  * FastAPI로 모델 서버를 구축하여 평가 점수와 gradcam을 빠르게 계산하여 제공
  * SQLite로 사용자들의 데이터 정보를 저장

## 🧪Experiments

- Model

  |                  | Parameters | Metric           | Training Time |
  | ---------------- | ---------- | ---------------- | ------------- |
  | Swin V2          | 88M        | P-Recall: 0.5331 | 13m 45s       |
  | Swin Transformer | 28.3M      | P-Recall:0.5822  | 16m 01s       |
  | Twin             | 43.8M      | P-Recall: 0.5435 | 13m 37s       |
  | EfficientNet-B4  | 19M        | P-Recall: 0.4084 | 9m 56s        |
  | EfficientNet-B0  | 5.3M       | P-Recall: 0.5002 | 5m 4s         |

  서비스를 하기 위해 모델의 성능 뿐 아니라 속도와 크기도 고려하여 EfficientNet-B0로 최종 결정

- Loss

  |      | Cross Entropy                        | label Smoothing CE                                       | Focal                                                        | Class Balanced Softmax CE                                | Class Balanced Focal                                     |
  | ---- | ------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- | -------------------------------------------------------- |
  |      | 다중 분류를 위한 손실 함수(Baseline) | Soft target으로 바꾸어 정답과 오답간의 score 격차를 줄임 | 어렵거나 쉽게 오분류되는 케이스에 더 큰 가중치를 주어 불균형 문제를 개선 | 새로운 케이스를 학습할 때 가중치를 주어 불균형 문제 개선 | 새로운 케이스를 학습할 때 가중치를 주어 불균형 문제 개선 |

  <img src=".\readme_file\loss3.jpg" alt="loss3" style="zoom:67%;" />

  <img src=".\readme_file\loss4.jpg" alt="loss4" style="zoom:67%;" />

  <img src=".\readme_file\loss5.jpg" alt="loss5" style="zoom:67%;" />

  <img src=".\readme_file\loss1.jpg" alt="loss1" style="zoom:67%;" />

  <img src=".\readme_file\loss2.jpg" alt="loss2" style="zoom:67%;" />

  각 항목 별 가장 높은 P-Recall을 달성한 loss fuction을 사용

- Imbalanced Data Sampler

  <img src=".\readme_file\sampler.jpg" alt="sampler" style="zoom:80%;" />

  비율이 높은 라벨은 undersampling, 비율이 낮은 라벨은 oversampling하여 학습

  |      | Oil                                             | Wrinkle                                         | Sensitive                                       | Pigmentation | Hydration |
  | ---- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ------------ | --------- |
  |      | 0.5879                                          | 0.4113                                          | 0.456                                           | 0.6934       | 0.4902    |
  |      | &#x1F539;<span style="color:blue">0.6207</span> | &#x1F539;<span style="color:blue">0.4177</span> | &#x1F539;<span style="color:blue">0.5551</span> | 0.6856       | 0.4701    |

  Oil,Wrinkle,Sensitive의 경우 P-Recall이 증가

  <span style="color:blue">

  </span>

- Augmentation

  |                              | Oil                                             | Wrinkle                                         | Sensitive                                       | Pigmentation                                    | Hydration                                       |
  | ---------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
  | Resize(512,512)(Baseline)    | 0.5879                                          | 0.4113                                          | 0.2158                                          | 0.6934                                          | 0.4908                                          |
  | Resize(1024,1024)            | 0.7552                                          | 0.3800                                          | <span style="color:blue">&#x1F539;0.6682</span> | &#x1F539;<span style="color:blue">0.8018</span> | 0.2022                                          |
  | CenterCrop(512,512)          | 0.6286                                          | 0.2765                                          | 0.2557                                          | 0.6671                                          | 0.4686                                          |
  | RandomCrop(512,512)          | 0.7516                                          | <span style="color:blue">&#x1F539;0.4480</span> | 0.4257                                          | <span style="color:blue">&#x1F539;0.7460</span> | 0.4745                                          |
  | RandomResizedCrop(512,512)   | 0.6580                                          | 0.3283                                          | 0.2549                                          | 0.7324                                          | <span style="color:blue">&#x1F539;0.5125</span> |
  | RandomResizedCrop(1024,1024) | <span style="color:blue">&#x1F539;0.7631</span> | 0.4213                                          | 0.6149                                          | 0.7803                                          | 0.2181                                          |

  |                   | Oil                                             | Wrinkle | Sensitive                                       | Pigmentation                                    | Hydration                                       |
  | ----------------- | ----------------------------------------------- | ------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
  | RGBShift          | &#x1F539;<span style="color:blue">0.6197</span> | 0.3171  | 0.2142                                          | <span style="color:blue">&#x1F539;0.7078</span> | <span style="color:blue">&#x1F539;0.5035</span> |
  | ISONoise          | <span style="color:blue">&#x1F539;0.6416</span> | 0.3524  | &#x1F539;<span style="color:blue">0.2517</span> | 0.6920                                          | 0.4390                                          |
  | OpticalDistortion | <span style="color:blue">&#x1F539;0.6441</span> | 0.3613  | 0.2081                                          | <span style="color:blue">&#x1F539;0.7151</span> | 0.4653                                          |
  | RandomBrightness  | <span style="color:blue">&#x1F539;0.6115</span> | 0.3696  | 0.2114                                          | 0.6904                                          | 0.4637                                          |
  | HorizontalFlip    | &#x1F539;<span style="color:blue">0.6516</span> | 0.3642  | <span style="color:blue">&#x1F539;0.2199</span> | 0.6768                                          | 0.3491                                          |

  Augmentation 실험은 다음 두 가지 관점에서 진행

  1) 주어진 데이터들의 크기와 종횡비가 다르므로 해상도와 비율에 대한 영향을 실험
  2) 피부를 촬영하는 환경 차이를 고려한 실험

  

  

## 🏆Result

- baseline -> 최종 모델

<img src=".\readme_file\result2.jpg" alt="result2" style="zoom:67%;" />

- 시연 영상

https://www.youtube.com/watch?v=My0J8PAtfeI&t=1s
