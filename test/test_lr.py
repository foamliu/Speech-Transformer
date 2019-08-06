import matplotlib.pyplot as plt
import torch

from config import sos_id, eos_id, vocab_size
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args

if __name__ == '__main__':
    k = 0.2
    warmup_steps = 4000
    init_lr = 512 ** (-0.5)

    args = parse_args()

    encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout, pe_maxlen=args.pe_maxlen)
    decoder = Decoder(sos_id, eos_id, vocab_size,
                      args.d_word_vec, args.n_layers_dec, args.n_head,
                      args.d_k, args.d_v, args.d_model, args.d_inner,
                      dropout=args.dropout,
                      tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                      pe_maxlen=args.pe_maxlen)
    model = Transformer(encoder, decoder)

    optimizer = TransformerOptimizer(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        args.k,
        args.d_model,
        args.warmup_steps)

    print(args.k)
    print(args.d_model)
    print(args.warmup_steps)

    lr_list = []
    for step_num in range(1, 50000):
        # print(step_num)
        lr_1 = k * init_lr * min(step_num ** (-0.5), step_num * (warmup_steps ** (-1.5)))
        optimizer.step()
        lr_2 = optimizer.lr
        # print(lr_1)
        # print(lr_2)
        lr_list.append(lr_2)

        # if step_num > 20:
        #     break

    print(lr_list[:100])
    print(lr_list[-100:])

    plt.plot(lr_list)
    plt.show()
