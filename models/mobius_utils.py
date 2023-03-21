import itertools
import torch.nn
import torch.nn.functional
import math
import geoopt.manifolds.poincare.math as pmath
import geoopt

def one_rnn_transform(W, h, U, x, b, c):
    W_otimes_h = pmath.mobius_matvec(W, h, c=c)
    U_otimes_x = pmath.mobius_matvec(U, x, c=c)
    Wh_plus_Ux = pmath.mobius_add(W_otimes_h, U_otimes_x, c=c)
    return pmath.mobius_add(Wh_plus_Ux, b, c=c)


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = pmath.logmap0(one_rnn_transform(W_hz, hx, W_iz, input, b_z, c), c=c).sigmoid()
    r_t = pmath.logmap0(one_rnn_transform(W_hr, hx, W_ir, input, b_r, c), c=c).sigmoid()

    rh_t = pmath.mobius_pointwise_mul(r_t, hx, c=c)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, b_h, c)

    if nonlin is not None:
        h_tilde = pmath.mobius_fn_apply(nonlin, h_tilde, c=c)
    delta_h = pmath.mobius_add(-hx, h_tilde, c=c)
    h_out = pmath.mobius_add(hx, pmath.mobius_pointwise_mul(z_t, delta_h, c=c), c=c)
    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    c: torch.Tensor,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
):
    if not hyperbolic_hidden_state0:
        hx = pmath.expmap0(h0, c=c)
    else:
        hx = h0
    if not hyperbolic_input:
        input = pmath.expmap0(input, c=c)
    outs = []
    if batch_sizes is None:
        input_unbinded = input.unbind(0)
        for t in range(input.size(0)):
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                c=c,
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
            hx = mobius_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                c=c,
            )
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t+1]], hx[batch_sizes[t+1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last