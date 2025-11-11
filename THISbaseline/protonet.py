import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_sq_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: [N,D], b: [M,D] -> [N,M]
    """
    a2 = (a*a).sum(dim=1, keepdim=True)        # [N,1]
    b2 = (b*b).sum(dim=1, keepdim=True).T      # [1,M]
    ab = a @ b.T                                # [N,M]
    return a2 + b2 - 2*ab

class ProtoNetHead(nn.Module):
    """
    Computes class prototypes from support embeddings, then scores queries.
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature, dtype=torch.float32)))

    def forward(self, z_supp: torch.Tensor, y_supp: torch.Tensor,
                z_query: torch.Tensor) -> torch.Tensor:
        """
        z_supp:  [Ns, D]
        y_supp:  [Ns]
        z_query: [Nq, D]
        Returns logits [Nq, n_way]
        """
        classes = torch.unique(y_supp)
        n_way = classes.numel()
        # prototypes
        protos = []
        for c in classes:
            protos.append(z_supp[y_supp == c].mean(dim=0, keepdim=True))
        protos = torch.cat(protos, dim=0)            # [n_way, D]
        # distances -> logits
        d2 = pairwise_sq_euclidean(z_query, protos)  # [Nq, n_way]
        logits = -d2 / torch.clamp(self.log_temp.exp(), min=1e-6)
        return logits, classes
