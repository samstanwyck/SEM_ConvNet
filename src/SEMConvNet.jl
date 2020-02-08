ENV["JULIA_CUDA_SILENT"] = true

using ProgressBars
using Images, FileIO, QuartzImageIO
using Flux, Statistics
using Flux: onehotbatch, crossentropy, throttle, onecold
using Base.Iterators: partition
using BSON
using Random
using Plots
using Plots.PlotMeasures

include("LoadSEMData.jl")
include("Utils.jl")
include("Models.jl")

function loss(x,y)
    y_hat = model(x)
    return crossentropy(y_hat, y)
end

function accuracy(x,y)
     return mean(onecold(model(x)) .== onecold(y))
 end

 function train_model(loss = loss, model = model, train_set = train_set, test_set = test_set, opt = ADAM(0.001),
     save_name = "SEM_convnet.bson")
    best_acc = 0.0
    last_improvement = 0
    training_losses = Vector{Float32}()
    test_losses = Vector{Float32}()
    accuracies = Vector{Float32}()
    for epoch_idx in 1:100

        @eval Flux.istraining() = true
        Flux.train!(loss, params(model), train_set, opt)
        training_loss = sum(loss(train_set[i]...) for i in 1:length(train_set))
        @eval Flux.istraining() = false
        test_loss = loss(test_set...)
        acc = accuracy(test_set...)



        println("Epoch ", epoch_idx,": Training Loss = ", training_loss, ", Test accuracy = ", acc)
        append!(training_losses, training_loss)
        append!(accuracies, acc)
        append!(test_losses, test_loss)

        if acc >= 0.999
            break
        end

        if acc > best_acc
            println("New best accuracy")
            BSON.@save "sem_convnet_best.bson" model epoch_idx accuracies training_loss test_loss
            best_acc = acc
            last_improvement = epoch_idx
        end

        if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
            opt.eta /= 10
            println("Dropping learn rate to $(opt.eta)")
        end

        if epoch_idx - last_improvement >= 10
            println("Converged")
            break
        end
    end
    BSON.@save save_name model accuracies training_losses test_losses
    end

#train_images, train_labels, test_images, test_labels, assignment_dict = load_all("SEM_data/", 0.9, 500)
#train_images, train_labels, test_images, test_labels = shuffle_data(train_images, train_labels, test_images, test_labels)
#train_set, test_set = make_minibatches(train_images, train_labels, test_images, test_labels, 32)
#model = construct_dropout_convnet()
#test_model_compile = precompile_model(model, train_set[1][1])
#accuracy(test_set...)
#sum(loss(train_set[i]...) for i in 1:length(train_set))
#train_model(loss, model, train_set, test_set, ADAM(0.001), "dropout_atom_test.bson")
