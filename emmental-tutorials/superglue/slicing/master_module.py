import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceMasterModule(nn.Module):
    def __init__(
        self,
        slice_ind_key="_slice_ind_",
        slice_pred_key="_slice_pred_",
        slice_pred_feat_key="_slice_feat_",
    ):
        super().__init__()

        self.slice_ind_key = slice_ind_key
        self.slice_pred_key = slice_pred_key
        self.slice_pred_feat_key = slice_pred_feat_key

    def forward(self, immediate_ouput_dict):
        slice_ind_names = sorted(
            [
                flow_name
                for flow_name in immediate_ouput_dict.keys()
                if self.slice_ind_key in flow_name
            ]
        )
        slice_pred_names = sorted(
            [
                flow_name
                for flow_name in immediate_ouput_dict.keys()
                if self.slice_pred_key in flow_name
            ]
        )

        if (
            len(
                [
                    F.softmax(immediate_ouput_dict[slice_ind_name][0])[:, 0].unsqueeze(
                        1
                    )
                    for slice_ind_name in slice_ind_names
                ]
            )
            == 0
        ):
            import pdb

            pdb.set_trace()

        Q = torch.cat(
            [
                F.softmax(immediate_ouput_dict[slice_ind_name][0])[:, 0].unsqueeze(1)
                for slice_ind_name in slice_ind_names
            ],
            dim=-1,
        )
        P = torch.cat(
            [
                F.softmax(immediate_ouput_dict[slice_pred_name][0])[:, 0].unsqueeze(1)
                for slice_pred_name in slice_pred_names
            ],
            dim=-1,
        )

        slice_feat_names = sorted(
            [
                flow_name
                for flow_name in immediate_ouput_dict.keys()
                if self.slice_pred_feat_key in flow_name
            ]
        )

        slice_reps = torch.cat(
            [
                immediate_ouput_dict[slice_feat_name][0].unsqueeze(1)
                for slice_feat_name in slice_feat_names
            ],
            dim=1,
        )

        A = F.softmax(Q * P, dim=1).unsqueeze(-1).expand([-1, -1, slice_reps.size(-1)])

        reweighted_rep = torch.sum(A * slice_reps, 1)

        return reweighted_rep
