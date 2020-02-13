using Flux

function dropout_convnet_1()
    model = Chain(
    Conv((3,3), 1=>16, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((3,3), 32=>32, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x,4)),
    Dense(2048,10),
    Dropout(0.5),
    softmax,
    )
    return model
end
function dropout_convnet_2()
    model = Chain(
    Conv((3,3), 1=>16, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((3,3), 32=>64, pad = (1,1), stride = (2,2), relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x,4)),
    Dense(9216,10),
    Dropout(0.5),
    softmax,
    )
    return model
end

function dropout_convnet_3()
    model = Chain(
    Conv((7,7), 1=>16, pad = (1,1),  relu),
    MaxPool((2,2)),
    Conv((7,7), 16=>32, pad = (1,1), relu),
    MaxPool((2,2)),
    Conv((7,7), 32=>32, pad = (1,1), relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x,4)),
    Dense(32768,10),
    softmax,
    )
    return model
end

function dropout_convnet_4()
    model = Chain(
    Conv((7,7), 1=>16, pad = (3,3), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((7,7), 16=>32, pad = (3,3), stride = (2,2), relu),
    MaxPool((2,2)),
    Conv((7,7), 32=>32, pad = (3,3), stride = (2,2), relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x,4)),
    Dense(2048,10),
    Dropout(0.5),
    softmax,
    )
    return model
end

function dropout_convnet_5()
    model = Chain(
    Conv((3,3), 1=>16, pad = (1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 16=>32, pad = (1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 32=>32, pad = (1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 32=>64, pad = (1,1), relu),
    MaxPool((2,2)),
    Conv((3,3), 64=>64, pad = (1,1), relu),
    MaxPool((2,2)),
    x -> reshape(x, :, size(x,4)),
    Dense(16384,10),
    softmax,
    )
    return model
end

function standard_convnet()
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
