# 10-701 Homework 4 Programming
Author: Elvis Pan

Email: ypan2@andrew.cmu.edu

## Files

| File                         | Description             |
|------------------------------|-------------------------|
|`AdaBoost.py`                 | Code for AdaBoost       |
|`DecisionStump.py`            | Code for decision stump |
|`datasets/train_adaboost.csv` | Training dataset        |
|`datasets/test_adaboost.csv`  | Testing dataset         |

## Run
To reproduce the solution to the questions, run `$ python3 AdaBoost.py` in the directory, and the program will generate the plots in the `result` folder and print the testing loss of the four training configurations.

To try other configurations, import the file `AdaBoost.py` and run it like the following example.
```python
train_x, train_y = read_data("/path/to/data")
AB = AdaBoost()
AB.fit(train_x, train_y, iteration=50)
print(AB.eval_model(train_y, AB.predict(train_x)))
```
