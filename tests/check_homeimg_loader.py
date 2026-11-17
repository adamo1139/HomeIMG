import json, sys, tempfile
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from homeimg_loader import load_homeimg_pipeline
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

class Tiny(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(width))

source='''import torch
from diffusers import ConfigMixin, ModelMixin, DiffusionPipeline
from diffusers.configuration_utils import register_to_config
class Tiny(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width=2):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(width))
class TinyPipeline(DiffusionPipeline):
    def __init__(self, transformer):
        super().__init__()
        self.register_modules(transformer=transformer)
'''
root=Path(tempfile.mkdtemp(prefix='homeimg_loader_check_'))
sentinel=object(); sys.modules['pipeline']=sentinel
for i in range(4):
    folder=root/str(i); folder.mkdir()
    (folder/'pipeline.py').write_text(source)
    (folder/'model_index.json').write_text(json.dumps({'_class_name':'TinyPipeline','transformer':['pipeline','Tiny']}))
    tiny=Tiny(); tiny.weight.data.fill_(i+1); tiny.save_pretrained(folder/'transformer')
    if i >= 2:
        (folder/'transformer'/'pipeline.py').write_text(source)
    loaded=load_homeimg_pipeline(folder)
    assert torch.equal(loaded.transformer.weight, tiny.weight)
    assert sys.modules['pipeline'] is sentinel
    if i==0: first_class=type(loaded)
    else: assert type(loaded) is not first_class
print('PASS: root and component custom code, weights, export isolation and alias restoration')
