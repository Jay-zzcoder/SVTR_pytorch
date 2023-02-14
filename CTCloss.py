import torch
import torch.nn as nn

def ctc1():
    # Target are to be padded
    T = 50      # Input sequence length
    C = 20      # Number of classes (including blank)
    N = 16      # Batch size
    S = 30      # Target sequence length of longest target in batch (padding length)
    S_min = 10  # Minimum target length, for demonstration purposes
    # Initialize random batch of input vectors, for *size = (T,N,C)
    input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
    # Initialize random batch of targets (0 = blank, 1:C = classes)
    target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
    ctc_loss = nn.CTCLoss()
    loss = ctc_loss(input, target, input_lengths, target_lengths)
    loss.backward()


def ctc2():
    # Target are to be un-padded
    T = 50      # Input sequence length
    C = 20      # Number of classes (including blank)
    N = 16      # Batch size
    # Initialize random batch of input vectors, for *size = (T,N,C)
    input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    # Initialize random batch of targets (0 = blank, 1:C = classes)
    target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
    target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
    print(target)
    print(target.shape)
    ctc_loss = nn.CTCLoss()
    loss = ctc_loss(input, target, input_lengths, target_lengths)
    loss.backward()



def ctc3():
    # Target are to be un-padded and unbatched (effectively N=1)
    T = 50      # Input sequence length
    C = 20      # Number of classes (including blank)
    # Initialize random batch of input vectors, for *size = (T,C)
    input = torch.randn(T, C).log_softmax(2).detach().requires_grad_()
    input_lengths = torch.tensor(T, dtype=torch.long)
    # Initialize random batch of targets (0 = blank, 1:C = classes)
    target_lengths = torch.randint(low=1, high=T, size=(), dtype=torch.long)
    target = torch.randint(low=1, high=C, size=(target_lengths,), dtype=torch.long)
    ctc_loss = nn.CTCLoss()
    loss = ctc_loss(input, target, input_lengths, target_lengths)
    loss.backward()

if __name__ == "__main__":
    m = nn.LogSoftmax(dim=0)
    input = torch.randn(2, 3)
    print(input)
    output = m(input)
    print(output)