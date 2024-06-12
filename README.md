# 光學檢查瑕疵分類專案

## 專案介紹
這是工研院電光所在AIdea網站提供的議題,旨在開發自動光學檢查(AOI)系統,用於識別軟性電子顯示器生產過程中的瑕疵。

本專案的目標是訓練一個機器學習模型,根據提供的影像數據,將瑕疵分為6個類別:
正常(normal)、空洞(void)、水平缺陷(horizontal )、垂直缺陷(vertical )、邊緣缺陷(edge )和顆粒(particle)

## 瑕疵分類範例

<img src="https://github.com/NoahWuW/AOI_defection_detection_practice/blob/main/dataset_example.jpg" alt="alt text" width="468" height="390">

## 影像數據來源
工研院電光所提供了兩個數據集:訓練集和測試集,但由於版權原因,這些數據集無法在此提供。
如果您需要訪問這些數據,請前往 [AIdea平台](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4) 自行下載。

下載後的數據包含:

訓練集:
- `train_images.zip`: 2,528張PNG格式的影像
- `train.csv`: 包含影像ID和對應的瑕疵類別標籤(0-5)

測試集:
- `test_images.zip`: 10,142張PNG格式的影像
- `test.csv`: 包含影像ID,瑕疵類別標籤為0-5之一

請確保將解壓縮後的影像檔案放置在適當的目錄中,以供訓練和評估腳本訪問。

## 執行流程
本專案在Google Colab上進行。以下簡易說明執行流程:

1. 下載數據集並解壓縮。
2. 導入所需的庫,如PyTorch、Torchvision等。
3. 切分數據集
3. 定義數據轉換和加載器。
4. 初始化預訓練模型(如ResNet、EfficientNet等),並根據需要修改最後一層。
5. 定義損失函數、優化器和學習率調節器。
6. 訓練模型,在驗證集上評估性能,並保存最佳模型權重。
7. 在測試集上預測並生成提交文件。

## 其它
- 還可嘗試其它預訓練模型和優化器,比較它們的性能。
- 資料預處理的部分使用了RandomResizedCrop，似乎不是很適合這類資料集。
- 可以加入argparse.ArgumentParser，讓訓練有更多的調整彈性
AOI_defection_detection_practice~
