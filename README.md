# trans_MTHv2_to_PaddleOCR

将[MTHv2](https://github.com/HCIILAB/MTHv2_Datasets_Release)数据集转换为适用于PaddleOCR的格式。

如检测det中，PaddleOCR的默认数据集标注格式为 训练集仅一个label文件，其中每行是一个的图片相对路径和文本、边框信息。

识别rec中，数据集图片是一行（列）文字；训练集仅一个label文件，其中每行是图片的相对路径和文本内容。

## 主要方法

* 参数`dataset_root_dir`为MTHv2的路径，如`/home/aistudio/dataset/rec/TKHMTH2200`，其内应有三个数据集MTH1200、MTH100、TKH，脚本会对三个数据集分别操作
* 方法`build_det_mthv2(dataset_root_dir: str)`：对三个数据集逐个生成label文件
* 方法`build_rec_mthv2(dataset_root_dir: str)`：对三个数据集逐个操作，读入原始图片，根据四点坐标切割出单列文本存为图片，并将信息写入新生成的label文件
* 方法`generate_rec_dict(dataset_root_dir: str)`：识别训练时需要一个字典文件，本方法对三个数据集逐个操作，从原始数据集的label_textline目录下逐个读入label文件，将文本信息使用集合逐字去重后，写入dict.txt

## 用法

```python
from trans_dataset import *

dataset_root_dir = '/home/aistudio/dataset/mthv2/rec'
# build_rec_mthv2(dataset_root_dir)
generate_rec_dict(dataset_root_dir)
```

