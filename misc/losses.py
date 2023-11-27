import torch


class LossCombination(torch.nn.Module):
    def __init__(self, criterions):
        super().__init__()
        self.criterions = criterions

    def forward(self, embeddings, targets):
        losses = []
        for criterion in self.criterions:
            losses.append(criterion(embeddings, targets))

        return sum(losses)
