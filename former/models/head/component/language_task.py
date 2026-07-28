import torch


class LanguageOnehotEncoder:
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.unique_labels = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def forward(self, language, device):
        new_unique_labels = list(set(language))
        self.unique_labels.extend(new_unique_labels)
        self.unique_labels = list(set(self.unique_labels))

        batch_size = len(language)

        one_hot_array = torch.zeros((batch_size, self.hidden_dim), dtype=torch.get_default_dtype(), device=device)
        for i, label in enumerate(language):
            one_hot_array[i, self.unique_labels.index(label)] = 1.0

        one_hot_array = one_hot_array.unsqueeze(2)  # batch_size, hidden_dim, 1
        return one_hot_array
