import os
import re
import pytorch_lightning as pl

import lib

class CheckPoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super(CheckPoint, self).__init__(*args, **kwargs)
    
    def _save_model(self, trainer, filepath):
        trainer.dev_debugger.track_checkpointing_history(filepath)

        self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)
        os.makedirs(filepath)

        # epoch = int(re.findall("epoch=(.*?)-", filepath)[0])
        # f1_score = float(re.findall("f1_class=(.*?).ckpt", filepath)[0])
        

        # try:
        #     trainer.model.model.save_pretrained(filepath)
        # except OSError:
        #     print("Warnings: OS Error")