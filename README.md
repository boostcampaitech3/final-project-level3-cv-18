## ๐ ํผ๋ถ ํ๊ฐ (XAI) ์์คํ

&nbsp; 
## ๐ฅ Member ๐ฅ
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
      <a href="https://github.com/kimkihoon0515">๊น๊ธฐํ_T3019</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ed-kyu">๊น์น๊ท_T3037</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/GwonPyo">๋จ๊ถํ_T3072</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/ysw2946">์ ์น์ฐ_T3130</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/jsh0551">์ฅ์ํธ_T3185</a>
    </td>
    <td align="center" width="120px">
      <a href="https://github.com/YJ0522771">์กฐ์ ์ง_T3208</a>
    </td>
  </tr>
</table>

&nbsp; 

## ๐Project Overview

- ์๋น์ค ๋ฐฐ๊ฒฝ ๋ฐ ๋ชฉํ

  <img src=".\readme_file\overview.jpg" alt="overview" style="zoom: 50%;" />

์์ง๊น์ง ์ฌ๋๋ค์ ํผ๋ถ๋ฅผ ์ง๋จํ๊ณ  ์ง์์ ์ผ๋ก ๊ด๋ฆฌํ๊ธฐ ์ํด์ ํผ๋ถ๊ณผ๋ฅผ ๋ฐฉ๋ฌธํ๊ณ  ๋น์ผ ๋น์ฉ์ ์ง๋ถํด์ผ ํฉ๋๋ค.

์ด๋ฌํ ๋ฌผ๋ฆฌ์  ์ ์ฝ์ ์์ ๊ณ ์, ๊ฐ๋จํ๊ฒ ํผ๋ถ๋ฅผ ์ง๋จํ  ์ ์๋ ์๋น์ค๋ฅผ ๋ง๋ค์ด ์ฌ์ฉ์๋ค์๊ฒ ํธ์์ฑ์ ์ ๊ณตํ๊ธฐ ์ํด ํ๋ก์ ํธ๋ฅผ ๊ธฐํํ๊ฒ ๋์์ต๋๋ค.

์ด ํผ๋ถ ์ง๋จ ์๋น์ค์ ๋ชฉํ๋ ์ค๋งํธํฐ ํน์ ์น์บ ์ ํตํด ์ผ๊ตด ์ฌ์ง์ ์ดฌ์ํ์ฌ, ๊ฐ ํญ๋ชฉ์ ๋ํ ์ ์์ ์ด์ ๋ํ ์๊ฐ์  ๊ทผ๊ฑฐ๋ฅผ ํจ๊ป ์ ๊ณตํ๋ ๊ฒ์๋๋ค.



<img src=".\readme_file\structure.jpg" alt="structure" style="zoom:50%;" />

* Frontend
  * User์ ID์ password๋ฅผ ์๋ ฅํ๋ฉด ๊ฐ์ธ ํผ๋ถ ์ฌ์ง์ ์๋ก๋ํ  ์ ์์
  * ์ฌ์ง์ ์๋ก๋ํ๋ฉด ํผ๋ถ ํ๊ฐ ์ ์์ gradcam์ ์ ๊ณต
  * ํ๊ฐ ์ ์๊ฐ ์ ์ฅ๋์ด ํผ๋ถ ๋ณํ ์ถ์ด๋ฅผ ๊ทธ๋ํ๋ก ํ์ธ ๊ฐ๋ฅ
* Backend
  * FastAPI๋ก ๋ชจ๋ธ ์๋ฒ๋ฅผ ๊ตฌ์ถํ์ฌ ํ๊ฐ ์ ์์ gradcam์ ๋น ๋ฅด๊ฒ ๊ณ์ฐํ์ฌ ์ ๊ณต
  * SQLite๋ก ์ฌ์ฉ์๋ค์ ๋ฐ์ดํฐ ์ ๋ณด๋ฅผ ์ ์ฅ



## ๐งฑStructure

โโโ backend     
โ  โโโbackend.py      
โ  โโโ fast.py      
โ  โโโ fast1.py     
โ  โโโ grad_cam.py      
โ  โโโ image_crop.py      
โ  โโโ prediction.py      
โ       
โโโ baseline      
โ  โโโ dataset.py     
โ  โโโ evaluation.py      
โ  โโโ loss.py      
โ  โโโ model.py     
โ  โโโ train.py     
โ โโโ utils.py      
โ       
โโโ EDA     
โ  โโโ EDA.ipynb      
โ       
โโโ frontend      
    โโโ logo          
   โโโ prototype.py       



## ๐ฉโ๐ซ๊ฐ๋ฐ ํ๊ฒฝ
- GPUํ๊ฒฝ : V100 ์๋ฒ, Google Cloud Platform(GCP) ์๋ฒ

- ํ ํ๋ ฅ Tools : Notion, Weights&Biases, Github, Google Drive, MLflow

- ๊ฐ๋ฐ Tools : VScode, Jupyter Lab

- ๋ผ์ด๋ธ๋ฌ๋ฆฌ ๋ฐ ํ๋ ์์ํฌ : PyTorch, Streamlit

  

## ๐๏ธDataset

- ํผ๋ถ๋ฅผ ์ ๋ถ, ์๋ถ, ์ฃผ๋ฆ, ์์, ๋ฏผ๊ฐ๋ 5๊ฐ์ง ํญ๋ชฉ์ผ๋ก ๋๋๊ณ , ์ ๋์ ๋ฐ๋ผ 0๋ถํฐ 4๊น์ง ํ๊ฐํ ํผ๋ถ ๋ฐ์ดํฐ<img src=".\readme_file\data.jpg" alt="data" style="zoom: 80%;" />

- ๊ฐ ํญ๋ชฉ์ ์ผ๋ถ ๋ผ๋ฒจ์ด ๋งค์ฐ ๋ถ์กฑํ imbalance data

- ๋ณด์ ๊ท์  ์ ํผ๋ถ ์ด๋ฏธ์ง, ๋ฐ์ดํฐ ์, ๋ผ๋ฒจ๋ง ๊ธฐ์ค ๋ฑ์ ๊ณต๊ฐ ๋ถ๊ฐ

  

## ๐จโ๐ซํ๊ฐ Metric

- Penalty Recall(P-Recall)

<img src=".\readme_file\precall.jpg" alt="precall" style="zoom: 67%;" />

- Ordinal data์ด๊ธฐ ๋๋ฌธ์ ์ ๋ต๋ง ๋ง์ถ๋ ๊ฒ ๋ฟ ์๋๋ผ ์ ๋ต๊ณผ ์ผ๋ง๋ ๋น์ทํ๊ฒ ์์ธกํ๋์ง๋ ์ค์ํจ

- ์ ๋ต๊ณผ ์ธ์ ํ ๋ผ๋ฒจ๋ก ์์ธกํ๋ฉด ์ผ์  ๊ฐ์ค์น๋ฅผ ๋ถ์ฌํ ์ ๋ต์ผ๋ก ์ธ์ ํ๊ณ , ์ ๋ต๊ณผ ๋จผ ๋ผ๋ฒจ๋ก ์์ธกํ๋ฉด ํจ๋ํฐ๋ฅผ ์ฃผ์ด recall์ ๊ณ์ฐ

  

## ๐งชExperiments

- Model(P-Recall)

  |                  | Parameters | Metric           | Training Time |
  | ---------------- | ---------- | ---------------- | ------------- |
  | Swin V2          | 88M        | P-Recall: 0.5331 | 13m 45s       |
  | Swin Transformer | 28.3M      | P-Recall:0.5822  | 16m 01s       |
  | Twin             | 43.8M      | P-Recall: 0.5435 | 13m 37s       |
  | EfficientNet-B4  | 19M        | P-Recall: 0.4084 | 9m 56s        |
  | EfficientNet-B0  | 5.3M       | P-Recall: 0.5002 | 5m 4s         |

  ์๋น์ค๋ฅผ ํ๊ธฐ ์ํด ๋ชจ๋ธ์ ์ฑ๋ฅ ๋ฟ ์๋๋ผ ์๋์ ํฌ๊ธฐ๋ ๊ณ ๋ คํ์ฌ EfficientNet-B0๋ก ์ต์ข ๊ฒฐ์ 

- Loss(P-Recall)

  |      | Cross Entropy                        | label Smoothing CE                                       | Focal                                                        | Class Balanced Softmax CE*                                | Class Balanced Focal*                                     |
  | ---- | ------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------ | -------------------------------------------------------- | -------------------------------------------------------- |
  | ์ค๋ช | ๋ค์ค ๋ถ๋ฅ๋ฅผ ์ํ ์์ค ํจ์(Baseline) | Soft target์ผ๋ก ๋ฐ๊พธ์ด ์ ๋ต๊ณผ ์ค๋ต๊ฐ์ score ๊ฒฉ์ฐจ๋ฅผ ์ค์ | ์ด๋ ต๊ฑฐ๋ ์ฝ๊ฒ ์ค๋ถ๋ฅ๋๋ ์ผ์ด์ค์ ๋ ํฐ ๊ฐ์ค์น๋ฅผ ์ฃผ์ด ๋ถ๊ท ํ ๋ฌธ์ ๋ฅผ ๊ฐ์  | ์๋ก์ด ์ผ์ด์ค๋ฅผ ํ์ตํ  ๋ ๊ฐ์ค์น๋ฅผ ์ฃผ์ด ๋ถ๊ท ํ ๋ฌธ์  ๊ฐ์  | ์๋ก์ด ์ผ์ด์ค๋ฅผ ํ์ตํ  ๋ ๊ฐ์ค์น๋ฅผ ์ฃผ์ด ๋ถ๊ท ํ ๋ฌธ์  ๊ฐ์  |

  | Loss function             | Oil     | Sensitive | Wrinkle | Pigmentation | Hydration |
  | ------------------------- | ------- | --------- | ------- | ------------ | --------- |
  | Baseline (Cross Entropy)  | 0.7501  | 0.456     | 0.3425  | 0.687        | 0.4726    |
  | Label Smoothing CE (0.1)  | 0.755   | 0.3545    | ๐น0.4113 | ๐น0.6934      | ๐น0.4908   |
  | Focal                     | 0.7627  | 0.3894    | 0.3356  | 0.682        | 0.4823    |
  | Class Balanced Softmax CE* | ๐น0.7246 | 0.5706    | 0.3618  | 0.6742       | 0.4598    |
  | Class Balanced Focal*      | 0.746   | ๐น0.5894   | 0.3708  | 0.6522       | 0.4617    |

>  *https://arxiv.org/abs/1901.05555

  ๊ฐ ํญ๋ชฉ ๋ณ ๊ฐ์ฅ ๋์ P-Recall์ ๋ฌ์ฑํ loss fuction์ ์ฌ์ฉ

* Masked Label Smoothing(P-Recall)

>  https://arxiv.org/abs/2203.02889

  * ๋ค๋ฅธ ๋ผ๋ฒจ๊ณผ์ score ๊ฒฉ์ฐจ๋ฅผ ์ค์ฌ์ฃผ๋ Label Smoothing์ ํน์ฑ์ ์ ์ฉํ์ฌ, ์ธ์ ํ ๋ผ๋ฒจ์๋ง ํ๋ฅ ์ ๋ฐฐ๋ถ

  <img src=".\readme_file\mls.jpg" alt="mls" style="zoom:67%;" />

<img src=".\readme_file\msl_hydration.jpg" alt="msl_hydration" style="zoom:50%;" />

<img src=".\readme_file\msl_pigmentation.jpg" alt="msl_pigmentation" style="zoom:50%;" />

<img src=".\readme_file\msl_wrinkle.jpg" alt="msl_wrinkle" style="zoom:50%;" />

- Imbalanced Data Sampler(P-Recall)

>  https://github.com/ufoym/imbalanced-dataset-sampler

  <img src=".\readme_file\sampler.jpg" alt="sampler" style="zoom:80%;" />

  ๋น์จ์ด ๋์ ๋ผ๋ฒจ์ undersampling, ๋น์จ์ด ๋ฎ์ ๋ผ๋ฒจ์ oversamplingํ์ฌ ํ์ต

  |      | Oil                                             | Wrinkle                                         | Sensitive                                       | Pigmentation | Hydration |
  | ---- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ------------ | --------- |
  |      | 0.5879                                          | 0.4113                                          | 0.456                                           | 0.6934       | 0.4902    |
  |      | &#x1F539;<span style="color:blue">0.6207</span> | &#x1F539;<span style="color:blue">0.4177</span> | &#x1F539;<span style="color:blue">0.5551</span> | 0.6856       | 0.4701    |

  Oil,Wrinkle,Sensitive์ ๊ฒฝ์ฐ P-Recall์ด ์ฆ๊ฐ

  <span style="color:blue">

  </span>

- Augmentation(P-Recall)

  |                              | Oil                                             | Wrinkle                                         | Sensitive                                       | Pigmentation                                    | Hydration                                       |
  | ---------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
  | Resize(512,512)(Baseline)    | 0.5879                                          | 0.4113                                          | 0.2702                                          | 0.6934                                          | 0.4908                                          |
  | Resize(1024,1024)            | 0.7552                                          | 0.3800                                          | 0.7547 | &#x1F539;<span style="color:blue">0.8018</span> | 0.2022                                          |
  | CenterCrop(512,512)          | 0.6286                                          | 0.2765                                          | 0.3815                                          | 0.6671                                          | 0.4686                                          |
  | RandomCrop(512,512)          | 0.7516                                          | <span style="color:blue">&#x1F539;0.4480</span> | 0.6782                                          | <span style="color:blue">&#x1F539;0.7460</span> | 0.4745                                          |
  | RandomResizedCrop(512,512)   | 0.6580                                          | 0.3283                                          | 0.4560                                          | 0.7324                                          | <span style="color:blue">&#x1F539;0.5125</span> |
  | RandomResizedCrop(1024,1024) | <span style="color:blue">&#x1F539;0.7631</span> | 0.4213                                          | <span style="color:blue">&#x1F539;0.7568</span>                                         | 0.7803                                          | 0.2181                                          |

  |                   | Oil                                             | Wrinkle | Sensitive                                       | Pigmentation                                    | Hydration                                       |
  | ----------------- | ----------------------------------------------- | ------- | ----------------------------------------------- | ----------------------------------------------- | ----------------------------------------------- |
  | RGBShift          | &#x1F539;<span style="color:blue">0.6197</span> | 0.4037  | <span style="color:blue">&#x1F539;0.2801</span>                                          | <span style="color:blue">&#x1F539;0.7078</span> | <span style="color:blue">&#x1F539;0.5385</span> |
  | ISONoise          | <span style="color:blue">&#x1F539;0.6416</span> | 0.3651  | &#x1F539;<span style="color:blue">0.4452</span> | 0.6920                                          | <span style="color:blue">&#x1F539;0.5035                                          |
  | OpticalDistortion | <span style="color:blue">&#x1F539;0.6441</span> | 0.3613  | <span style="color:blue">&#x1F539;0.3407</span>                                           | <span style="color:blue">&#x1F539;0.7151</span> | 0.4390                                          |
  | RandomBrightness  | <span style="color:blue">&#x1F539;0.6115</span> | <span style="color:blue">&#x1F539;0.4175</span>   | 0.2681                                          | 0.6904                                          | 0.4653                                          |
  | HorizontalFlip    | &#x1F539;<span style="color:blue">0.6516</span> | 0.3642  | <span style="color:blue">&#x1F539;0.2733</span> | 0.6768                                          | 0.4637                                          |

  Augmentation ์คํ์ ๋ค์ ๋ ๊ฐ์ง ๊ด์ ์์ ์งํ

  1) ์ฃผ์ด์ง ๋ฐ์ดํฐ๋ค์ ํฌ๊ธฐ์ ์ขํก๋น๊ฐ ๋ค๋ฅด๋ฏ๋ก ํด์๋์ ๋น์จ์ ๋ํ ์ํฅ์ ์คํ
  2) ํผ๋ถ๋ฅผ ์ดฌ์ํ๋ ํ๊ฒฝ ์ฐจ์ด๋ฅผ ๊ณ ๋ คํ ์คํ

  

  

## ๐Result

- ์ต์ข ๊ฒฐ๊ณผ(P-Recall)

  - ๊ฐ ํญ๋ชฉ๋ณ๋ก P-Recall์ ์ํฅ์ ์ฃผ์๋ ๊ธฐ๋ฒ๋ค์ ๋ชจ๋ ์ ์ฉํ์ฌ ์ต์ข ๋ชจ๋ธ ํ์ต

  |                   | Oil                                                          | Wrinkle                                        | Sensitive                         | Pigmentation                           | Hydration                             |
  | ----------------- | ------------------------------------------------------------ | ---------------------------------------------- | --------------------------------- | -------------------------------------- | ------------------------------------- |
  | Loss fuction      | focal                                                        | Masked Label Smoothing                         | Class Balanced Focal              | Masked Label Smoothing                 | Masked Label Smoothing                |
  | Augmentations     | RandomResizedCrop, RGBShift,ISONoise, OpticalDistortion, RandomBrightness, Horizontal Flip | RandomCrop, RGBShift, RandomBrightnessContrast | ResizedCrop, RGBShift, ISONoise, OpticalDistortion, Horizontal Flip | Resize, RandomCrop, RGBshift, ISONoise | RandomResizedCrop, RGBshift, ISONoise |
  | Baseline P-Recall | 0.5879                                                       | 0.4113                                         | 0.4560                            | 0.6934                                 | 0.4908                                |
  | **P-Recall**      | **0.7674**                                                   | **0.6524**                                     | **0.7852**                        | **0.8278**                             | **0.5859**                            |

  

- baseline -> ์ต์ข ๋ชจ๋ธ

<img src=".\readme_file\result2.jpg" alt="result2" style="zoom:67%;" />

- ์์ฐ ์์

https://www.youtube.com/watch?v=nDWqPPHq6UQ
 
- ๋ฐ๋ชจ ์ฌ์ดํธ
  
https://share.streamlit.io/kimkihoon0515/streamlit_demo/prototype.py
