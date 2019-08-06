import torch.nn as nn


class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        return pred, gold

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps
