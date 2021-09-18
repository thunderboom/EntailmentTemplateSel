code for Entailment Method Based on Template Selection for Chinese Text Few-shot Learning

### 环境

1. python3
2. pytorch (1.7.0)
3. 预训练模型为macbert_large

### 测试方式

```
1. bash run_classifier_tnews.sh x # tnews进行训练,第x份数据
2. bash run_classifier_tnews.sh x predict # tnews进行预测
3. bash run_base.sh # 对所有任务使用所有train和dev进行训练和预测
```