#融合则修改FusionGenerationModel下的MODEL_LIST,
#单模修改checkpoint地址CHECKPOINT 
#index中:
#from FusionGenerationModel import FusionGenerationModel as GenerationModel 用于融合
#from models import GenerationModel,GenerationModel_Pretrain 用于单模
cd project
python index.py