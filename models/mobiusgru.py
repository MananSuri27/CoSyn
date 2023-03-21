import itertools
import torch.nn
import torch.nn.functional
import math
import geoopt.manifolds.poincare.math as pmath
import geoopt
from mobius_utils import one_rnn_transform, mobius_gru_cell, mobius_gru_loop



class MobiusLinear(torch.nn.Linear):
    def __init__(self, *args, hyperbolic_input=True, nonlin=None ,c=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bias is not None:           
            self.ball = geoopt.PoincareBall(c=c)
            self.bias = geoopt.ManifoldParameter(self.bias, manifold=self.ball)
            with torch.no_grad():
                self.bias.set_(pmath.expmap0(self.bias.normal_(), c=c))
        else:
            self.ball = geoopt.PoincareBall(c=c)
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin

    def forward(self, input):
        if self.hyperbolic_input:
            output = pmath.mobius_matvec(self.weight, input, c=self.c)
        else:
            output = torch.nn.functional.linear(input, self.weight)
            output = pmath.expmap0(output, c=self.c)

        if self.bias is not None:
            output = pmath.mobius_add(output, self.bias, c=self.c)

        output = pmath.project(output, c=self.c)
        return output





class MobiusGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,  hyperbolic_input=True, hyperbolic_hidden_state0=True, c=1.0,):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=c)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.weight_ih = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )

        self.weight_hh = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )

        if bias:
            biases = []
            for i in range(num_layers):
                bias = torch.randn(3, hidden_size) * 1e-5
                bias = geoopt.ManifoldParameter(
                    pmath.expmap0(bias, c = self.ball.c), manifold=self.ball
                )
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)


        self.hyperbolic_input = hyperbolic_input

        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in itertools.chain.from_iterable([self.weight_ih, self.weight_hh]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0=None):

        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)

        if is_packed:
            input, batch_sizes = input[:2]
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)

        if h0 is None:
            h0 = input.new_zeros(
                self.num_layers, max_batch_size, self.hidden_size, requires_grad=False
            )

        h0 = h0.unbind(0)

        if self.bias is not None:
            biases = self.bias
        else:
            biases = (None,) * self.num_layers

        outputs = []
        last_states = []
        
        out = input
        for i in range(self.num_layers):
            out, h_last = mobius_gru_loop(
                input=out,
                h0=h0[i],
                weight_ih=self.weight_ih[i],
                weight_hh=self.weight_hh[i],
                bias=biases[i],
                c=self.ball.c,
                hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
                hyperbolic_input=self.hyperbolic_input or i > 0,
                batch_sizes=batch_sizes,
            )
            outputs.append(out)
            last_states.append(h_last)
        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack(last_states)

        return out, ht


    
