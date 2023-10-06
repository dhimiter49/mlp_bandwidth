# MLP Bandwidth (Complexity) Setting with Dropout

This repo implements a `FixedDropout` torch layer and a `WeightDropLinear` linear layer. The latter drops weights an biases of Linear layer during the forward pass using the given probability whereas the former implements a dropout layer with variable probability specified during the forward pass.

Both bandwidth manipulation methods are implemented as part of a MLP with a single hidden layer for sinus function prediction. `FixedDropout` does not work as intended because dropping activation layers results in a non-continuous function prediction.
