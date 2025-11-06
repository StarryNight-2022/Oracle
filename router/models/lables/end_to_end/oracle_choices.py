# In end-to-end cases, the orcacle router consider latency constraint, prompt, and models.
# Just need to give a label for router model's final choice, don't need too much details like two_steps do. 
# 需要配合Oracle Router，读取oracle router生成的决策结果。
# 也可以考虑调用Oracle Router进行决策，我更倾向与这一方式，因为这样更为灵活。
# TODO