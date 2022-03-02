import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(0, 1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class AMER(BaseModel):
    def __init__(self, config):
        super().__init__()

    def initialize(self, config, device):

        n_classes = 9 if config['emo_type'] == 'primary' else 14
        D_e = config["model"]["args"]["D_e"]
        D_v = config["visual"]["dim_env"] + config["visual"]["dim_face"] + config["visual"]["dim_obj"]
        D_a = config["audio"]["feature_dim"]
        D_t = config["text"]["feature_dim"]
        D_p = config["personality"]["feature_dim"]
        
        self.attn = ScaledDotProductAttention((4 * D_e) ** 0.5, attn_dropout=0)

        self.enc_v = nn.Sequential(
            nn.Linear(D_v, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, D_e * 3),
            nn.ReLU(),
            nn.Linear(D_e * 3, 2 * D_e),
        )

        self.enc_a = nn.Sequential(
            nn.Linear(D_a, D_e * 8),
            nn.ReLU(),
            nn.Linear(D_e * 8, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )
        self.enc_t = nn.Sequential(
            nn.Linear(D_t, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.enc_p = nn.Sequential(
            nn.Linear(D_p, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(4 * D_e, 2 * D_e), 
            nn.ReLU(), 
            nn.Linear(2 * D_e, n_classes)
        )

        unified_d = 6 * D_e

        self.fusion_layer = nn.Linear(unified_d, 4 * D_e)

    def forward(self, U_v, U_a, U_t, U_p, M_v, M_a, M_t, seq_lengths, target_loc, seg_len, n_c):
        # Encoders
        V_e, A_e, T_e, P_e = self.enc_v(U_v), self.enc_a(U_a), self.enc_t(U_t), self.enc_p(U_p)

        U_all = []

        for i in range(M_v.shape[0]):
            target_moment, target_character = -1, -1
            for j in range(target_loc.shape[1]):
                if target_loc[i][j] == 1:
                    target_moment = j % int(seg_len[i].cpu().numpy())
                    target_character = int(j / seg_len[i].cpu().numpy())
                    break
            
            inp_V = V_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            inp_T = T_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)
            inp_P = P_e[i, : seq_lengths[i], :].reshape((n_c[i], seg_len[i], -1)).transpose(0, 1)

            mask_V = M_v[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
            mask_T = M_t[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)
            mask_A = M_a[i, : seq_lengths[i]].reshape((n_c[i], seg_len[i])).transpose(0, 1)

            

            U = []

            for k in range(n_c[i]):
                new_inp_T, new_inp_V = inp_T.clone(), inp_V.clone(),
                new_inp_VT = inp_V.clone()
                new_inp_TV = inp_T.clone()
                # Modality-level inter-personal attention
                for j in range(seg_len[i]):
                    att_V, _ = self.attn(inp_V[j, :], inp_V[j, :], inp_V[j, :], mask_V[j, :])
                    att_T, _ = self.attn(inp_T[j, :], inp_T[j, :], inp_T[j, :], mask_T[j, :])
                    
                    new_inp_V[j, :] = att_V + inp_V[j, :]
                    
                    new_inp_T[j, :] = att_T + inp_T[j, :]
                    
                for j in range(seg_len[i]):
                    att_V_T, _ = self.attn(new_inp_V[j, :], new_inp_T[j, :], new_inp_T[j, :], mask_V[j, :])
                    att_T_V, _ = self.attn(new_inp_T[j, :], new_inp_V[j, :], new_inp_V[j, :], mask_T[j, :])
                    new_inp_VT[j, :] = att_V_T + new_inp_V[j, :]
                    new_inp_TV[j, :] = att_T_V + new_inp_T[j, :]
                    
                    
                att_VT_,_ = self.attn(new_inp_VT[:, k], new_inp_VT[:, k], new_inp_VT[:, k], mask_V[:, k])
                att_TV_,_ = self.attn(new_inp_TV[:, k], new_inp_TV[:, k], new_inp_TV[:, k], mask_T[:, k])
                inner_VT_ = att_VT_ + new_inp_VT[:,k]
                inner_TV_ = att_TV_ + new_inp_TV[:,k]
                
                att_VT,_ = self.attn(inner_VT_, inner_TV_, inner_TV_, mask_V[:, k])
                att_TV,_ = self.attn(inner_TV_, inner_VT_, inner_VT_, mask_T[:, k])
                inner_VT = (att_VT[target_moment] + inner_VT_[target_moment]).squeeze()
                inner_TV = (att_TV[target_moment] + inner_TV_[target_moment]).squeeze()
                
                # Multimodal fusion
                inner_U = self.fusion_layer(torch.cat([inner_VT, inner_TV, inp_P[0][k]]))

                U.append(inner_U)

            if len(U) == 1:
                # Only one character in this sample
                U_all.append(U[0])
            else:
                # Person-level Inter-personal Attention
                U = torch.stack(U, dim=0)
                output, _ = self.attn(U, U, U)
                U = U + output
                U_all.append(U[target_character])

        U_all = torch.stack(U_all, dim=0)
        # Classification
        log_prob = self.out_layer(U_all)
        log_prob = F.log_softmax(log_prob)

        return log_prob