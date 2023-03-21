import torch
import torch.nn as nn
import torch.nn.functional as F


from .mobiusgru import MobiusGRU, MobiusLinear


def lorentz_activation(input_tensor):
    rr = torch.norm(input_tensor, p=2, dim=2)
    dd = input_tensor.permute(2,0,1) / rr
    cosh_r = torch.cosh(rr)
    sinh_r = torch.sinh(rr)
    output_tensor = torch.cat(((dd * sinh_r).permute(1, 2, 0), cosh_r.unsqueeze(0).permute(1, 2, 0)), dim=2)
    return output_tensor

def poincare_to_lorentz(input_tensor):
    eps = 1e-6
    rr = torch.norm(input_tensor, p=2, dim=2)
    output_tensor = torch.cat((2*input_tensor, (1+rr**2).unsqueeze(2)),dim=2).permute(2,0,1)/(1-rr**2+eps)
    return output_tensor.permute(1,2,0)

def lorentz_to_klien(input_tensor):
    eps = 1e-6
    dump = input_tensor[:, :, -1]
    dump = torch.clamp(dump, eps, 1.0e+16)
    return (input_tensor[:, :, :-1].permute(2, 0, 1)/dump).permute(1, 2, 0)

def arcosh(x):
    eps = 1e-6
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(torch.clamp(x * x - 1 + eps, min=0)) / x)
    return c0 + c1

def distance_lorentz(x, y):
    eps = 1e-6
    m = x * y
    prod_minus = -m[:, :, :-1].sum(dim=2) + m[:, :, -1]
    output_tensor = torch.clamp(prod_minus, 1.0 + eps, 1.0e+16)
    return arcosh(output_tensor)




class HFAN(nn.Module):
    def __init__(self, word_dim=768, hidden_dim=100,word_hidden_size=50, final_dim=768, num_classes=2):
        super(HFAN, self).__init__()
        self.sentence = HyperbolicPostAttention(hidden_dim,word_hidden_size,num_classes)
        self.reduce_dim1 = nn.Linear(word_dim,hidden_dim)
        self.reduce_dim2 = nn.Linear(2*hidden_dim, final_dim)
    
    def forward(self,x_):

        x_ = lorentz_to_klien(lorentz_activation(x_))

        res = []

        for x in x_:
            word_hidden_state = self.reduce_dim1(x)
            
            sentence_hidden_state, f = self.sentence(word_hidden_state)

            sentence_final = self.reduce_dim2(sentence_hidden_state)

            res.append(sentence_final)

        return torch.squeeze(torch.stack(res))


class HyperbolicPostAttention(nn.Module):
    def __init__(self, sent_hidden_size=100, word_hidden_size=50, num_classes=2):
        super().__init__()

        self.gru_forward = MobiusGRU(2*word_hidden_size, sent_hidden_size)
        self.gru_backward = MobiusGRU(2*word_hidden_size, sent_hidden_size)
        self.lorentz_centroid = nn.Parameter(torch.Tensor(2*sent_hidden_size))
        self.hyp_att_projector = MobiusLinear(2*sent_hidden_size, 2*sent_hidden_size, bias=True)

        self.beta = nn.Parameter(torch.Tensor(1))
        self.c = 1.0

        self.initialize_weights(0.0,0.05)

    def initialize_weights(self, mean=0.0, std=0.05):
        self.lorentz_centroid.data.normal_(mean, std)
        self.beta.data.normal_(mean, std)

    def forward(self, input):



        input = (input.permute(1,0)/(torch.sqrt(1 + torch.norm(input, p=2, dim=-1) ** 2))).permute(1,0)


        f_output1, h_output1 = self.gru_forward(input.view(-1,100,100))
        f_output2, h_output2 = self.gru_backward((torch.flip(input,(0,))).view(-1,100,100))


        f_output = torch.cat((f_output1, f_output2), -1)

        output = poincare_to_lorentz(f_output)
        output = lorentz_to_klien(output)


        att_projected = self.hyp_att_projector(f_output)     
        att_projected = poincare_to_lorentz(att_projected)
        centroid_effective = lorentz_activation(self.lorentz_centroid.unsqueeze(0).unsqueeze(0))
        att_projected = - self.beta * distance_lorentz(att_projected, centroid_effective) - self.c
        att_projected_normalised = att_projected - att_projected.max()
        att_projected_exponential = torch.exp(att_projected_normalised)

        eps = 1e-6


        res = att_projected_exponential
        inv_gamma = 1 - torch.norm(output, p=2, dim=2) ** 2
        gamma = 1/torch.sqrt(torch.clamp(inv_gamma, eps, 1 - eps))
        gamma = torch.clamp(gamma, 1.0 + eps, 1.0e+16)

        res = res * gamma
        res = res / (torch.sum(res, dim=0))    


        output = torch.sum(res * output.permute(2, 0, 1), dim=2).permute(1, 0)

        output = (output.permute(1,0) / (1 + torch.sqrt(1 + torch.norm(output, p=2, dim=1) ** 2))).permute(1, 0)

        return output, f_output.permute(1, 0, 2)









    