import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, consensus_type, dim=1):
        ctx.consensus_type = consensus_type
        ctx.dim = dim
        ctx.shape = input_tensor.size()
        if consensus_type == 'avg':
            output = input_tensor.mean(dim=dim, keepdim=True)
        elif consensus_type == 'identity':
            output = input_tensor
        else:
            output = None
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None
        return grad_in, None, None



class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1, args=None):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim
        self.args = args

    def forward(self, input, lite_input=None):
        if self.consensus_type == "scsampler":  # TODO(yue)
            if lite_input is None:
                lite_input = input
            res = []
            ind = torch.topk(torch.max(lite_input.detach(), dim=2).values, dim=1, k=self.args.top_k).indices
            for bi in range(lite_input.shape[0]):
                res.append(torch.stack([input[bi, k, :] for k in ind[bi]]))
            output = torch.mean(torch.stack(res), dim=1, keepdim=True)

            if self.args.real_scsampler:
                return output, ind

            return output
        else:
            return SegmentConsensus.apply(input, self.consensus_type, self.dim)
