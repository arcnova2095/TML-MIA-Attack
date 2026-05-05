This method uses boosted trees to find a good gradient for classification. It is very fast and efficient.
And it can also emulate properties of shadow models by creating n-tree, each of them carrying their
own decisions while hiding some part of data from each tree. Processing thousands of trees was a
matter of seconds unlike Logistic regression.

Basic Working of the code:
1) Public and Private Features from pub.pt and priv.pt are extracted
2) Private features are extracted for prediction and Public features are extracted for training
3) XGBoost classifier trains on the model
4) XGBoost predicts the data membership with private features.
5) These predictions are saved in a CSV and submitted.
