using Flux

function construct_dropout_convnet()
    model = Chain(
    Conv((3,3), 1=>16, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((3,3), 32=>32, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x,4)),
    Dense(4608,10),
    Dropout(0.5),
    softmax,
    )
    return model
end

function construct_standard_convnet()
    model = Chain(
    Conv((3,3), 1=>16, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((3,3), 32=>32, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x,4)),
    Dense(4608,10),
    softmax,
    )
    return model
end
