# Noise reduction
## 概要
これは、次のことを行うソフトウェアです。

欲しい信号signal(x)にノイズnoise(x)が乗っている観測データ  
 obs(x) = signal(x) + noise(x)  
から、ノイズを除去するモデルの学習を行う。

## 必要なソフトウェア
このソフトウェアを動作させるにはJupyter notebookが必要です。
pythonコードのみを抜き出せば、Jupyter無しでも動くかもしれませんが、未確認です。
その他に必要なソフトウェアと、実際に動作させたときのバージョンを次に示します。
* matplotlib  2.2.2
* TensorFlow  1.10.0
* numpy       1.15.0 

## ライセンス
Copyright (c) 2018 高エネルギー数値計算  
このソフトウェアはMITライセンスの下で公開されています。  
http://opensource.org/licenses/mit-license.php

また、このソフトウェアの一部は、次のコードが利用されています。

TensorFlowではじめるDeepLearning実践入門　サンプルコード  
Copyright(c) 2018 Takuya Shinmura  
ライセンス：  
https://opensource.org/licenses/mit-license.php  
コードの場所：  
https://github.com/thinkitcojp/TensorFlowDL-samples
