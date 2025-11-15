#------------------------ 生成用于训练mf模型的pairwise数据 ---------------------------

# 参考routellm的huggingface数据
# # dataset:
# ['question_id', 'model_a', 'model_b', 'winner', 'judge', 'conversation_a', 'conversation_b', 'turn', 'anony', 'language', 'tstamp', 'openai_moderation', 'toxic_chat_tag']
# dataset[0]:
# {'question_id': '58210e39b3fd4441a2bd4a518bb44c2d', 'model_a': 'chatglm-6b', 'model_b': 'koala-13b', 'winner': 'model_b', 'judge': 'arena_user_973', 
# 'conversation_a': [{'content': 'What is the difference between OpenCL and CUDA?', 'role': 'user'}, {'content': '<省略>', 'role': 'assistant'}], 
# 'conversation_b': [{'content': 'What is the difference between OpenCL and CUDA?', 'role': 'user'}, {'content': '<省略>', 'role': 'assistant'}], 
# 'turn': 1, 'anony': True, 'language': 'English', 'tstamp': 1682351591.1322, 'openai_moderation': {'categories': {'harassment': False, 'harassment/threatening': False, 'hate': False, 'hate/threatening': False, 'self-harm': False, 'self-harm/instructions': False, 'self-harm/intent': False, 'sexual': False, 'sexual/minors': False, 'violence': False, 'violence/graphic': False}, 'category_scores': {'harassment': 2.8765102e-05, 'harassment/threatening': 5.663866e-07, 'hate': 5.574919e-06, 'hate/threatening': 2.3965333e-08, 'self-harm': 5.1901172e-09, 'self-harm/instructions': 1.1517327e-08, 'self-harm/intent': 3.9070875e-09, 'sexual': 4.381485e-06, 'sexual/minors': 5.541973e-08, 'violence': 2.2985896e-06, 'violence/graphic': 4.8212314e-07}, 'flagged': False}, 'toxic_chat_tag': {'roberta-large': {'flagged': False, 'probability': 0.008313022553920746}, 't5-large': {'flagged': False, 'score': 7.943665219245499e-05}}}

# 参考上面信息，最终定义出我们的 pairwise_data.json 文件的格式如下:
# 这是基于回答质量给出的比较
# {"idx":0, "model_a":"", "model_b":"", "winner":""}

