# Enhanced Spam Detection Training Report

Generated: 2025-06-23 14:29:13

## Data Statistics
- Total Messages: 17574
- Labeled Messages: 11574
- Unlabeled Messages: 6000
- Spam Messages: 5247
- Ham Messages: 6327
- Feature Count: 5015

## Training Results
### multinomial_nb
- Accuracy: 0.925
- Precision: 0.980
- Recall: 0.852
- F1 Score: 0.911
- ROC AUC: 0.976

### complement_nb
- Accuracy: 0.928
- Precision: 0.973
- Recall: 0.866
- F1 Score: 0.916
- ROC AUC: 0.976

### logistic
- Accuracy: 0.968
- Precision: 0.980
- Recall: 0.950
- F1 Score: 0.965
- ROC AUC: 0.993

### random_forest
- Accuracy: 1.000
- Precision: 1.000
- Recall: 1.000
- F1 Score: 1.000
- ROC AUC: 1.000

### label_propagation
- Accuracy: 0.937
- Precision: 0.919
- Recall: 0.944
- F1 Score: 0.932
- ROC AUC: 0.987

### label_spreading
- Accuracy: 0.936
- Precision: 0.918
- Recall: 0.943
- F1 Score: 0.931
- ROC AUC: 0.984

## Cross-Validation Results
### multinomial_nb
- Mean F1 Score: 0.900
- Standard Deviation: 0.006

### complement_nb
- Mean F1 Score: 0.906
- Standard Deviation: 0.006

### logistic
- Mean F1 Score: 0.950
- Standard Deviation: 0.003

### random_forest
- Mean F1 Score: 0.977
- Standard Deviation: 0.004

## Best Model: random_forest

## Recommendations
- Use ensemble methods for best performance
- Consider semi-supervised learning for unlabeled data
- Regularly retrain models with new data
- Monitor false positive rates in production
