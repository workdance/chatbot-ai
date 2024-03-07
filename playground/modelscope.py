from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
pipe = pipeline(task=Tasks.chat, model='ZhipuAI/ChatGLM-6B-Int4', model_revision='v1.0.1')
inputs = {'text':'你好', 'history': []}
result = pipe(inputs)
inputs = {'text':'介绍下清华大学', 'history': result['history']}
result = pipe(inputs)
print(result)