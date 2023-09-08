import os
# for huggingface models:
os.environ['TRANSFORMERS_CACHE'] = '../hf_models/'

from prody import confProDy
confProDy(verbosity='none') # stop printouts from prody