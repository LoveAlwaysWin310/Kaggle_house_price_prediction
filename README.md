# 🏠 Kaggle House Price Prediction

本项目基于 Kaggle 竞赛 **[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)**，使用 **PyTorch** 进行房价预测，并采用 **深度学习 + 特征工程** 来提高模型性能。

## 📌 项目简介
房价预测是一个典型的 **回归问题**，数据集包括波士顿房价信息，如房屋面积、地段、建筑年份等特征。本项目主要使用 **神经网络（MLP）** 进行建模，并结合 **数据清理、特征选择、标准化** 等技术提升预测精度。

## 🏗️ 技术栈
- **深度学习**: `PyTorch`
- **数据处理**: `Pandas`、`NumPy`、`scikit-learn`
- **数据可视化**: `Matplotlib`
- **优化算法**: `Adam`
- **正则化手段**: `Dropout`、`BatchNorm`

---

## 📂 目录结构
```bash
📦 Kaggle_house_price_prediction
│-- 📁 data/             # 数据集存放位置（如 train.csv, test.csv）
│-- 📁 outputs/          # 预测结果及可视化文件
│-- 📄 main.py           # 主训练脚本
│-- 📄 README.md         # 项目介绍（你正在看的这个文件）
