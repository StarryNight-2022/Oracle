This Project implement Oracle Router and Random Router. The Data will not be published.

step1.evaluate_all.py 负责进行数据预处理，对每个数据进行正确性判断。

step2.get_oracle.py 负责进行oracle route的决策。

运行依赖：
evaluate_all.py -> batch_plot.py (gen_oracle.py -> gen_random.py -> plot_diagram.py)
evaluate_all.py -> plot_bar_chart.py
plot_scatter.py
