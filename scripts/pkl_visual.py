import pickle
import optuna
import matplotlib.pyplot as plt
from optuna.visualization import plot_param_importances

# 读取保存的study对象
with open('data/optuna_result/study.pkl', 'rb') as f:
    study = pickle.load(f)

# 可视化参数重要性
# 可视化优化过程
fig = optuna.visualization.plot_optimization_history(study)
fig.write_html('data/optuna_result/optimization_history.html')

# 可视化参数分布
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.write_html('data/optuna_result/parallel_coordinate.html')

# 可视化参数重要性
fig = plot_param_importances(study)
fig.write_html('data/optuna_result/param_importances.html')