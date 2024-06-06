# VAEAT: Variational AutoeEncoder with Adversaria Training for Multivariate Time Series Anomaly Detection
https://www.sciencedirect.com/science/article/pii/S0020025524007667
## Requirements
 * PyTorch 1.6.0
 * CUDA 10.1 (to allow use of GPU, not compulsory)

# Dataset

* SMAP and MSL:

```
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

* SMD:

```
https://github.com/NetManAIOps/OmniAnomaly
```

* SWaT:

```
http://itrust.sutd.edu.sg/research/dataset
```


* Run the code

```
python main.py <dataset>
```

```
where `<dataset>` is one of `SMAP`, `MSL`, `SMD`, `SWAT`, `PSM`

@article{HE2024120852,
title = {VAEAT: Variational AutoeEncoder with adversarial training for multivariate time series anomaly detection},
author = {Sheng He and Mingjing Du and Xiang Jiang and Wenbin Zhang and Congyu Wang},
journal = {Information Sciences},
volume = {676},
pages = {120852},
year = {2024}
}
```
