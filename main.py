import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from transformers import MT5Tokenizer
import lib


if __name__ == "__main__":
    parser = ArgumentParser(description="Multilingual Generative Label Sequence Labeling For Event Extraction")
    parser.add_argument("--train_data_path", type = str, default="data/processed_data/Stage_2/English/augmented_train.json")
    parser.add_argument("--val_data_path", type = str, default="data/processed_data/Stage_2/English/dev.json")
    
    parser.add_argument("--en_test_data_path", type = str, default="data/processed_data/Stage_2/English/test.json")
    parser.add_argument("--zh_test_data_path", type = str, default="data/processed_data/Stage_2/Chinese/test.json")
    parser.add_argument("--ar_test_data_path", type = str, default="data/processed_data/Stage_2/Arabic/test.json")
    
    parser.add_argument("--alpha", type = float, default=0.5)
    parser.add_argument("--aug_lang", type = str, default='zh')

    parser.add_argument("--log_dir", type = str, default="logs")
    parser.add_argument("--output_log", type = str, default="en_to_zh_ar.json")
    
    parser.add_argument("--pretrained_path", type = str, default="pretrained-models/en-zh-epoch=08-f1_class=0.7052.ckpt", help= "Pretrained Transformer Model")
    parser.add_argument("--tokenizer_path", type = str, default="pretrained-models/en-zh-epoch=08-f1_class=0.7052.ckpt")

    parser.add_argument("--muse_src_lang_path", type = str, default="muse/wiki.multi.en.vec")
    parser.add_argument("--muse_tgt_lang_path", type = str, default="muse/wiki.multi.en.vec")

    parser.add_argument("--src_max_length", type = int, default=160)
    parser.add_argument("--tgt_max_length", type = int, default=160)
    parser.add_argument("--learning_rate", type = float, default=1e-6)
    parser.add_argument("--weight_decay", type = float, default=0.01)
    parser.add_argument("--max_grad_norm", type = float, default=0.5)
    parser.add_argument("--n_epochs", type = int, default=20)
    parser.add_argument("--num_beams", type = int, default=3)
    
    parser.add_argument("--train_batch_size", type = int, default=2)
    parser.add_argument("--val_batch_size", type = int, default=8)
    parser.add_argument("--accumulate_grad_batches", type = int, default=4)
    parser.add_argument("--n_steps", type = int, default=0)
    parser.add_argument("--gpu", type = int, default=0)
    
    args = parser.parse_args()
    pl.utilities.seed.seed_everything(42)
    # if os.path.exists(os.path.join(args.log_dir, "results.json")):
    #     os.remove(os.path.join(args.log_dir, "results.json"))
    
    tokenizer = MT5Tokenizer.from_pretrained(args.tokenizer_path)
    model = lib.MT5Trainer(tokenizer, args)
    
    checkpoint_callback = lib.CheckPoint(monitor='f1_class', filename='{epoch:02d}-{f1_class:.4f}', save_top_k=-1)
    
    trainer = pl.Trainer(gpus=[args.gpu], 
                         callbacks=[checkpoint_callback],
                         max_epochs=args.n_epochs,
                         gradient_clip_val=args.max_grad_norm,
                         num_sanity_val_steps=0)
    trainer.fit(model)               