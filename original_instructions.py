def repr_module(module):
    # We treat the extra repr like the sub-module, one item per line
    extra_lines = []
    extra_repr = module.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in module._modules.items():
        mod_str = repr_module(module)
        mod_str = indentify(mod_str, '  ')
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = module._get_name() + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    return main_str

import diffusers
if not 'model' in dir():model=diffusers.CogVideoXTransformer3DModel()
print(repr_module(model))

"""
That prints out this:

Make a textual library tree explorer app so I can fold this model interactily and view it 
I want to be able to fold and unfold regions here. And would be nice to have highlighting too! See how I made one part blue for example?

Linear(

  (patch_embed):   Linear(

    (proj):   Conv2d(16, 1920, kernel_size=(2, 2), stride=(2, 2))
    (text_proj):   Linear(in_features=4096, out_features=1920, bias=True)
  )
  (embedding_dropout):   Dropout(p=0.0, inplace=False)
  (time_proj):   Timesteps()
  (time_embedding):   Linear(

    (linear_1):   Linear(in_features=1920, out_features=512, bias=True)
    (act):   SiLU()
    (linear_2):   Linear(in_features=512, out_features=512, bias=True)
  )
  (transformer_blocks):   CogVideoXBlock(

    (0):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (1):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (2):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (3):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(


    ......... 500 lines omitted .............



    (17):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (18):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (19):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (20):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (21):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (22):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (23):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (24):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (25):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (26):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (27):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (28):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
    (29):   FeedForward(

      (norm1):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (attn1):   ModuleList(

        (norm_q):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (norm_k):   LayerNorm((64,), eps=1e-06, elementwise_affine=True)
        (to_q):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_k):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_v):   Linear(in_features=1920, out_features=1920, bias=True)
        (to_out):   Dropout(

          (0):   Linear(in_features=1920, out_features=1920, bias=True)
          (1):   Dropout(p=0.0, inplace=False)
        )
      )
      (norm2):   LayerNorm(

        (silu):   SiLU()
        (linear):   Linear(in_features=512, out_features=11520, bias=True)
        (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
      )
      (ff):   ModuleList(

        (net):   Dropout(

          (0):   Linear(

            (proj):   Linear(in_features=1920, out_features=7680, bias=True)
          )
          (1):   Dropout(p=0.0, inplace=False)
          (2):   Linear(in_features=7680, out_features=1920, bias=True)
          (3):   Dropout(p=0.0, inplace=False)
        )
      )
    )
  )
  (norm_final):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
  (norm_out):   LayerNorm(

    (silu):   SiLU()
    (linear):   Linear(in_features=512, out_features=3840, bias=True)
    (norm):   LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
  )
  (proj_out):   Linear(in_features=1920, out_features=64, bias=True)
)


"""
