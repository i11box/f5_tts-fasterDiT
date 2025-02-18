
```
f5_tts
├─ .coverage
├─ .pytest_cache
│  ├─ CACHEDIR.TAG
│  ├─ README.md
│  └─ v
│     └─ cache
│        ├─ lastfailed
│        ├─ nodeids
│        └─ stepwise
├─ api.py
├─ assets
│  ├─ pic
│  └─ strategy_heatmap
├─ eval
│  ├─ config
│  │  └─ eval_config.toml
│  ├─ ecapa_tdnn.py
│  ├─ eval_fastdit_infer.py
│  ├─ eval_infer_batch.py
│  ├─ eval_infer_batch.sh
│  ├─ eval_librispeech_test_clean.py
│  ├─ eval_seedtts_testset.py
│  ├─ README.md
│  └─ utils_eval.py
├─ infer
│  ├─ examples
│  │  ├─ basic
│  │  │  └─ basic.toml
│  │  ├─ multi
│  │  │  ├─ story.toml
│  │  │  └─ story.txt
│  │  └─ vocab.txt
│  ├─ infer_cli.py
│  ├─ infer_gradio.py
│  ├─ README.md
│  ├─ SHARED.md
│  ├─ speech_edit.py
│  └─ utils_infer.py
├─ model
│  ├─ backbones
│  │  ├─ dit.py
│  │  ├─ mmdit.py
│  │  ├─ README.md
│  │  └─ unett.py
│  ├─ cfm.py
│  ├─ hook.py
│  ├─ logger.py
│  ├─ modules.py
│  ├─ trainer.py
│  ├─ utils.py
│  └─ __init__.py
├─ pyproject.toml
├─ scripts
│  ├─ attn_weights_heatmap.py
│  ├─ count_max_epoch.py
│  ├─ count_params_gflops.py
│  ├─ demo (1).py
│  ├─ draw.py
│  ├─ draw_method_heatmap.py
│  ├─ plot_attention_similarity.py
│  ├─ ref_hook.py
│  └─ 转换_2.10.py
├─ socket_server.py
├─ test
│  ├─ test_compress_manager.py
│  └─ __init__.py
├─ tests
│  └─ block_check
└─ train
   ├─ finetune_cli.py
   ├─ finetune_gradio.py
   ├─ README.md
   └─ train.py

```