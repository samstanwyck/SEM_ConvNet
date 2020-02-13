
function create_assignment_list(assignment_dict::Dict)
    assignment_list = Vector{String}(undef, 10)
    for (k,v) in assignment_dict
        assignment_list[v+1] = k
        end
    return assignment_list
end

function shuffle_data(train_images, train_labels, test_images, test_labels, generator = 1234)
    train_set = collect(zip(train_images, train_labels))
    test_set = collect(zip(test_images, test_labels));
    rng = MersenneTwister(generator);
    shuffle!(rng, train_set)
    shuffle!(rng, test_set)
    train_images = [x[1] for x in train_set]
    train_labels = [x[2] for x in train_set]
    test_images = [x[1] for x in test_set]
    test_labels = [x[2] for x in test_set]
    return train_images, train_labels, test_images, test_labels
end

function make_minibatches(train_images, train_labels, test_images, test_labels, batch_size = 32)
        mb_idxs = partition(1:length(train_images), batch_size)
        train_set = [make_minibatch(train_images, train_labels, i) for i in mb_idxs]
        test_set = make_minibatch(test_images, test_labels, 1:length(test_images))

        return train_set, test_set
end

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:,:,:,i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:9)
    return (X_batch, Y_batch)
end

function load_pretrained_model(path)
    model_dict = BSON.load(path)
    model = model_dict[:model]
    return model
end

function precompile_model(model, train_set)
    return model(train_set[1][1])
end
