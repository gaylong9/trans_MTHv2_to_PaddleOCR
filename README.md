# trans_MTHv2_to_PaddleOCR

将[MTHv2](https://github.com/HCIILAB/MTHv2_Datasets_Release)数据集转换为适用于PaddleOCR的格式。

如检测det中，PaddleOCR的默认数据集标注格式为 训练集仅一个label文件，其中每行是一个的图片相对路径和文本、边框信息。

识别rec中，数据集图片是一行（列）文字；训练集仅一个label文件，其中每行是图片的相对路径和文本内容。



* 关键方法为`build_det_mthv2(dataset_root_dir:str)`和`build_rec_mthv2(dataset_root_dir:str)`
* `dataset_root_dir`为MTHv2的路径，如`/home/aistudio/dataset/rec/TKHMTH2200`，其内应有三个数据集MTH1200、MTH100、TKH，脚本会对三个数据集分别操作
* 检测用数据集仅是对原始label的重新整理
* 识别用数据集转换时，会读入原图片，根据边框裁剪出每列文字的最小包围矩形作为数据集图片
