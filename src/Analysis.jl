using BSON

function LoadModelData(path)
    model_dict = BSON.load(path)
    return model_dict
end

function ExtractModelParameters(model_dict)
    training_losses = model_dict[:training_losses]
    test_losses = model_dict[:test_losses]
    accs = model_dict[:accuracies]
    model = model_dict[:model]
    return model, accs, test_losses, training_losses
end

function ExtractBestModelParameters(model_dict)
    training_loss = model_dict[:training_loss]
    test_loss = model_dict[:test_loss]
    accs = model_dict[:accuracies]
    model = model_dict[:model]
    return model, accs, test_loss, training_loss
end
function PlotResults(accs, training_losses, test_losses)
    epochs = 1:length(training_losses)
    normalized_training_losses = training_losses./maximum(training_losses)
    normalized_test_losses = test_losses./maximum(test_losses)
    plotly()
    plot(epochs, [normalized_training_losses, normalized_test_losses, accs],
     title="SEM CNN Performance",
     label=["Normalized Training Loss" "Normalized Test Loss" "Test Accuracy"],
      left_margin = 5mm, xlabel = "Epoch", ylabel = "Value", lw=3)
  end

function ConfusionMatrix(model, test_set, k)
    model_predictions = onecold(model(test_set[1]))
    ground_truths = onecold(test_set[2])
    length(ground_truths) == length(model_predictions) || throw(DimensionMismatch("Ground truths and prediction vectors must
    be the same length"))
    R = zeros(Int, k, k)
    for i = 1:length(model_predictions)
        @inbounds g = ground_truths[i]
        @inbounds p = model_predictions[i]
        R[g, p] += 1
    end
    return R
end



function PlotConfusionMatrix(R, assignment_list)
    gr()
    heatmap(assignment_list, assignment_list, R, size = (700, 500), xrotation = 45)
end
