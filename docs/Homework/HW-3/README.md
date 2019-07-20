
# Homework 3


## Purpose: Image Sentiment Classification

本次作業為人臉表情情緒分類，總共有七種可能的表情（0：生氣, 1：厭惡, 2：恐懼, 3：高興, 4：難過, 5：驚訝, 6：中立(難以區分為前六種的表情))。

本次作業利用 training dataset 訓練一個 CNN model，預測出每張圖片的表情 label（同樣地，為 0~6 中的某一個)，並跟參數個數差不多的 DNN model 去進行比較。

## Data 簡介

* training dataset 為兩萬八千張左右 48x48 pixel的圖片，以及每一張圖片的表情 label（注意：每張圖片都會唯一屬於一種表情）。

* Testing dataset 則是七千張左右 48x48 的圖片

* 如下圖，由左至右依序為生氣、厭惡、恐懼、高興、難過、驚訝、中立

![](02-Output/DisplayData.png)

* 各類別數量
<table style="width:50%">
  <tr>
    <td>**0：生氣** </td> 
    <td> 3995 </td> 
  </tr>
  
  <tr>
    <td>**1：厭惡**</td>
    <td> 436 </td> 
  </tr>
  
  <tr>
    <td>**2：恐懼**</td>
    <td> 4097 </td> 
  </tr>

  <tr>
    <td>**3：高興**</td>
    <td> 7215 </td> 
  </tr>

  <tr>
    <td>**4：難過**</td>
    <td> 4830 </td> 
  </tr>

  <tr>
    <td>**5：驚訝**</td>
    <td> 3171 </td> 
  </tr>

  <tr>
    <td>**6：中立**</td>
    <td> 4965 </td> 
  </tr>
  
</table>


## Summary 總結

在資料處理的部分，先將 training data 進行標準化，並取出最後 5000 資料當作 validation data (大約 20% 資料量)。

首先 CNN model 模型架構如下圖所示，其中 C1、C2、C3、C4 為 Convolution block，包含 Convolution2D()、BatchNormalization()、Activation()、MaxPooling2D() 和 Dropout()。

- Convolution2D() : Filter Size 均為 (3,3)，Strides 均為 1
- BatchNormalization() : 讓每批量的數據分布相似並達到加速收斂的效果，另一個目的為讓 training data、validation data 和 testing data 數據分布相似，提高模型的泛化能力和避免 overfitting
- Activation() : 均使用 Relu
- MaxPooling2D() : Pooling size 均為 (2,2)，Strides 均為 2
- Dropout() : C1、C2、C3、C4、FC1、FC2 的 dropout rate 依序為 0.3, 0.3, 0.3, 0.4, 0.5, 0.5

![](02-Output/Cnn.png)

模型總參數數量為 4,183,815，其中有 3968 個是 BatchNormalization() 的 non-trainable 個數。

在 CNN model 訓練過程中，可以觀察到大約在 20 個 epoch 左右，validation loss 來到了低點，validation accuracy 似乎也到了極限 (63.86%)。在 20 個 epoch 之後 validation accuracy 只有稍為的提升，最好的 validation accuracy 為 65.54% 。

![](02-Output/cnnLossAccuracyCurves.png)


DNN model 模型架構如圖下所示，其中 FC1、FC2、FC3、FC4 為 Fully Connection layer，包含 BatchNormalization()、Activation() 和 Dropout()。

- BatchNormalization() : 目的為加速收斂和避免 overfitting
- Activation() : 均使用 Relu
- Dropout() : FC1、FC2、FC3 的 dropout rate 均為 0.5

![](02-Output/Dnn.png)

模型總參數數量為 4,478,983，其中有 6,144 個是 BatchNormalization() 的 non-trainable 個數。

在 DNN model 訓練過程中，我觀察到與 CNN model 差異點有:

- DNN model 的 validation accuracy 完全無法跟 CNN model 相比。
- 訓練速度上 DNN model 速度完勝於 CNN model 。 DNN model 一個 epoch 大約 20sec 而 CNN model 約 200sec。個人猜測主要原因為 Convolution 需要消耗比較多的運算。
- 在 validation loss 部分似乎不是很樂觀，在 40 個 epoch 之後甚至有上升的趨勢。

![](02-Output/dnnLossAccuracyCurves.png)


### Confusion Matrix

這個部分分別針對 DNN 和 CNN model 計算 confusion matrix 並且利用視覺化的方式呈現結果。

DNN model 的部分，根據 confusion matrix (下左圖) 可以知道，

- 唯獨開心 (Hppy) 模型的正確辨識程度高過於 50%，這類剛好也是資料數量最多的類別。
- 模型似乎捕捉不到在各種情緒上的細微差異，導致各類別容易誤判。

 CNN model，根據 confusion matrix (下右圖) 來看，

- 很清楚的知道各類別預測狀況相對於 DNN model 來說進步很多。
- 害怕 (Fear) 很容易會誤判成傷心 (Sad)，誤判機率大約 22%。
- 中立 (Neutral) 和傷心 (Sad) 相對於模型來說不容易分辨，彼此誤判的機率差不多 20%。
- 生氣 (Angry) 會被誤判成傷心 (sad)，誤判機率大約 17%。
- 厭惡 (Disgust) 會被誤判成生氣 (Angry)，誤判機率大約 16%。
- 在開心 (Hppy) 和驚訝 (Surprise) 這種表情鮮明的類別，模型的辨識狀況相對而言就會比較好，而生氣 (Angry)、厭惡 (Disgust)、害怕 (Fear) 這種複雜的情緒對模型來說相對而言也比較難辨識其中差異，甚至連人類都不太容易分辨出差異點。

<div class="half">
    <img src="02-Output/dnnConfusionMatrix.png" height="300px">
    <img src="02-Output/cnnConfusionMatrix.png" height="300px">
</div>

下圖我們將各類別預測錯誤且預測信心最高的圖片找出來，可以發現一些有趣的現象:

- 我個人覺得一、三、六、七張測得都相對比 label 好，猜測是 label 標記錯誤
- 從三、四、六、七隱約可以發現模型判斷 Happy 和 Surprise 之間的差別，Surprise 有點像是情緒放大版的 Happy，其中如果嘴巴形狀又是接近圓形更可能被判定為 Surprise
- 第五張很難辦定是 Neutral 還是 Sad，好像也都可以。

![](02-Output/cnnDisplayData.png)


### Saliency Map

這部份主要是要觀察圖片每個 pixel 對於 CNN model 預測結果的影響力，藉此來了解模型在做分類時主要是 focus 在圖片哪個部分。

做法其實很簡單，只要針對圖片的預測結果對每個 pxiel 進行微分，再取絕對值，就知道哪個 pxiel 對模型預測的影響力比較大。另外先定義符號，<a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{y}_{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{y}_{k}" title="\widehat{y}_{k}" /></a> 為模型預測的 label，<a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a> 為圖片，<a href="https://www.codecogs.com/eqnedit.php?latex=x_{ij}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{ij}" title="x_{ij}" /></a> 為圖片的 pixel ，數學上的表達為 <a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;|\frac{\partial&space;\widehat{y}_{k}}{\partial&space;x_{ij}}&space;\right&space;|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;|\frac{\partial&space;\widehat{y}_{k}}{\partial&space;x_{ij}}&space;\right&space;|" title="\left |\frac{\partial \widehat{y}_{k}}{\partial x_{ij}} \right |" /></a> 簡單說就是該值越大，代表影響力越大。

以下為將上述的內容用視覺化的方式表達出來。下列圖中最左邊的原圖分別從 5000 筆 validation data 中選出的各類別預測正確照片，由上到下 label 分別為 生氣 (Angry)#23、厭惡 (Disgust)#189、害怕 (Fear)#53、開心 (Happy)#2、傷心 (Sad)#6、驚訝 (Surprise)#15、中立 (Neutral)#4。

從第四張類別 Happy 的 heatmap 可以發現在嘴巴部分有較高的值，可見模型在做分類時，是將重點放在偵測嘴巴的部分，而這張圖被判定為 Happy 的主要依據也是因為嘴巴的笑容。第七張類別 Surprise 的 heatmap 可以發現眼睛和嘴巴部位的值相對於其他部位來的高，很清楚的知道模型再對這類別做分類的重點是在眼睛和嘴巴，其他類別雖然沒有特別明顯的部位，但主要都將重點放在臉部。

![](02-Output/cnnSaliencyMapAngry.png)

![](02-Output/cnnSaliencyMapDisgust.png)

![](02-Output/cnnSaliencyMapFear.png)

![](02-Output/cnnSaliencyMapHappy.png)

![](02-Output/cnnSaliencyMapNeutral.png)

![](02-Output/cnnSaliencyMapSad.png)

![](02-Output/cnnSaliencyMapSurprise.png)

### Visualizing Filters

利用 gradient ascent，觀察 filter 被 activate 的情況，以及圖片經過 filter 的結果。

下圖左邊為第一層 convolution 後的 filter，右邊為 validation data 中的第二筆資料 (Saliency Map Happy 類別的原圖) 經過第一層 convolution 的結果。

<div align="center">
    <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;第一層 convolution 後的 filter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 第一層 validation data convolution 後的結果</figcaption>
    <img src="02-Output/cnnFiltersWhiteNoiseconv2d_1.png" height="460px">
    <img src="02-Output/cnnFiltersResultImage2conv2d_1.png" height="460px">
</div>

接下來將經過完整的 convolution + batch normalization + activation 後的 filter 取出來 (下圖左)。發現其實每個 filter 都是由基本的線條所組成且看起來很相似，但可以觀察到有個現象是，filter 中的線條似乎傾斜角度有不同(旋轉)，猜測可能是因為照片臉的角度有正臉、側臉等因素造成的，最後可以將這個 layer 理解成被基本的紋理所 activate。右圖則為圖片經過相對應的 layer 所產生的結果。

<div align="center">
    <figcaption>第一層 convolution + batch normalization + activation 後的 filter&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 第一層 validation data convolution block 後的結果</figcaption>
    <img src="02-Output/cnnFiltersWhiteNoiseactivation_1.png" height="460px">
    <img src="02-Output/cnnFiltersResultImage2activation_1.png" height="460px">
</div>

之後數層 convolution 後的 filter 和 convolution block 後的 filter 以及 validation data 中的第二筆資料相對應該層的輸出結果如下列所示:

<div align="center">
    <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;第二層 convolution 後的 filter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 第二層 validation data convolution 後的結果</figcaption>
    <img src="02-Output/cnnFiltersWhiteNoiseconv2d_2.png" height="400px" width="400px" title="第二層 convolution 後的 filter">
    <img src="02-Output/cnnFiltersResultImage2conv2d_2.png" height="400px" width="400px" title="第二層 convolution 後的結果">
</div>

<div align="center">
    <figcaption>第二層 convolution + batch normalization + activation 後的 filter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 第二層 validation data convolution block 後的結果</figcaption>
    <img src="02-Output/cnnFiltersWhiteNoiseactivation_2.png" height="400px" width="400px" title="第二層 convolution + batch normalization + activation 後的 filter">
    <img src="02-Output/cnnFiltersResultImage2activation_2.png" height="400px" width="400px" title="第二層 convolution + batch normalization + activation 後的結果">
</div>

<div align="center">
    <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;第三層 convolution 後的 filter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 第三層 validation data convolution 後的結果</figcaption>
    <img src="02-Output/cnnFiltersWhiteNoiseconv2d_3.png" height="400px" width="400px" title="第三層 convolution 後的 filter">
    <img src="02-Output/cnnFiltersResultImage2conv2d_3.png" height="400px" width="400px" title="第三層 convolution 後的結果">
</div>

<div align="center">
    <figcaption>第三層 convolution + batch normalization + activation 後的 filter&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 第三層 validation data convolution block 後的結果</figcaption>
    <img src="02-Output/cnnFiltersWhiteNoiseactivation_3.png" height="400px" width="400px" title="第三層 convolution + batch normalization + activation 後的 filter">
    <img src="02-Output/cnnFiltersResultImage2activation_3.png" height="400px" width="400px" title="第三層 convolution + batch normalization + activation 後的結果">
</div>

<div align="center">
    <figcaption>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;第四層 convolution 後的 filter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 第四層 validation data convolution 後的結果</figcaption>
    <img src="02-Output/cnnFiltersWhiteNoiseconv2d_4.png" height="400px" width="400px" title="第四層 convolution 後的 filter">
    <img src="02-Output/cnnFiltersResultImage2conv2d_4.png" height="400px" width="400px" title="第四層 convolution 後的結果">
</div>

<div align="center">
    <figcaption>第四層 convolution + batch normalization + activation 後的 filter &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 第四層 validation data convolution block 後的結果</figcaption>
    <img src="02-Output/cnnFiltersWhiteNoiseactivation_4.png" height="400px" width="400px" title="第四層 convolution + batch normalization + activation 後的結果">
    <img src="02-Output/cnnFiltersResultImage2activation_4.png" height="400px" width="400px" title="第四層 convolution + batch normalization + activation 後的結果">
</div>

由上面左邊一系列的圖可以得知越深層 convolution 的 filter 已經從之前的直線紋理演變成曲線紋理，意味著 filter 捕捉的特徵越來越複雜，convolution block 的 filter 也有相同的趨勢，但卻出現了許多不被激活的 filter (圖片下方 loss 為 0)，目前猜測是因為原本 convolution 的 filter 數值變化可能就不大又經過 BatchNormalization 之後將數值歸一化所導致(待驗證)。

右邊一系列的圖，我們發現在第一層的 layer 中，filter 捕捉較大的範圍，人臉相當清晰，而第二層 layer 則捕捉人臉的輪廓、眼睛、頭髮、嘴巴...等等，更深層的 layer 則捕捉更局部更細微的特徵。在第四層 layer 可以很清楚的知道每個 filter 只針對特定的小方格去做特徵的提取。


### Data Augmentation

這個部份我們針對原來 CNN Model 的部分，對訓練資料多做 Data Augmentation，藉此來說明 Data Augmentation 的效果。Data Augmentation 就是對圖片做平移、翻轉、選轉、縮放、推移...等等，進而獲得更多且更多元的訓練資料，使模型更強健和減少過擬和的狀況。
由於使用 data augmentation 的緣故，我們設定在每一次 epoch 時所訓練的資料數為沒有使用 data augmentation 的三倍，讓模型可以完整的訓練完原來的資料和新生成的資料。

訓練過程如下圖所示，可以觀察到經過 100 個 epoch，validation loss 和 validation accuracy 似乎還在遞減，雖然預測精準度大約只提升到 68%，但持續訓練下去或許就會突破 70%。另外訓練過程的曲線相較於沒有使用 data augmentation 來的完美，也間接說明模型訓練得比較好，很清楚地可以知道是 data augmentation 的功勞。
最後如果希望預測精準度可以再提高，個人建議可以增加每一次 epoch 訓練資料的倍數(3倍提高到5倍之類的)，但這方法所需要付出的代價為花更多時間訓練模型。

![](02-Output/gencnnLossAccuracyCurves.png)

### 心得:

在做這份作業的過程中，如果沒有將pixel除以255，模型訓練效果會非常差，主要原因是因為沒有除以255導致模型訓練速度過慢，在沒有良好的設備和時間的情況下，結果都不會太優。而除以255之後 pixel 數值會分布在0~1之間，這樣可以加速模型的訓練，以至於在同樣的模型相同的訓練次數結果會差很多。###要在驗證

隨著 Convolution 越來越多層，模型在訓練集的預測正確率可以高達90%以上，但在驗證集始終無法突破 55% 的預測正確率，這現象意味著模型過擬和訓練資料。面對這樣的問題我們採用 droupout 來抑制過擬合現象，首先在 fully connection 的部分採用 droupout，在驗證集的表現似乎有提升 3% ~ 5% 左右，但就是過不了 60% 。隨著 droupout 的強度越來越強，甚至對 Convolution 也進行 droupout 的過程中我們也可以發現在驗證集的正確預測率可以達到 65% 。

1. 照片需要除255 效果影響很大 50%以下

2. 3 個 conv + 2-3個 fc vail data ~50%~55%

3. 4 個 conv + 2-3個 fc vail data ~55%~60% basic droupout 0.2 on fc

4. 4 個 conv + 2-3個 fc vail data ~57%~63% basic droupout 0.2 on all layer                     15epoch

5. 4 個 conv + 2-3個 fc vail data ~60%~65% basic droupout 0.5 on fc layer 0.2 0.3 0.3 onconv   20epoch


## Reference

* [原始課程作業說明](https://docs.google.com/presentation/d/1QFK4-inv2QJ9UhuiUtespP4nC5ZqfBjd_jP2O41fpTc/edit?ts=58e452ff#slide=id.p)

* [Keras Image Data Augmentation 個參數詳解](https://zhuanlan.zhihu.com/p/30197320)

* [BatchNormalization](http://blog.csdn.net/hjimce/article/details/50866313)

* [How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)