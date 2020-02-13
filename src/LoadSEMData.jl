using ProgressBars
using Images
using FileIO
using QuartzImageIO

export load_all


function train_test_split(images, labels, ratio::Float64)
    @assert length(images) == length(labels)
    @assert 0 < ratio < 1
    separator = trunc(Int, ratio*length(images))
    train_images = images[1:separator]
    train_labels = labels[1:separator]
    test_images = images[separator + 1:length(images)]
    test_labels = labels[separator + 1:length(images)]
    return train_images, train_labels, test_images, test_labels
    end

function create_training_and_test_data(path::String, label::Int64, ratio::Float64 = 0.9, max_group_size::Int64 = 100)
    images = []
    labels = Array{Int64}(undef, 0)
    files = readdir(path)
    for file in files
        if length(labels) >= max_group_size
            break
        end

        img = load(string(path, file))
        if size(img) == (768, 1024)
            push!(labels, label)
            gray_img = Gray.(img)
            gray_img_cropped = gray_img[1:512,1:512]
            push!(images, gray_img_cropped)
        end
    end
    train_images, train_labels, test_images, test_labels = train_test_split(images, labels, ratio)
    return train_images, train_labels, test_images, test_labels

    end

function load_all(path::String, ratio::Float64 = 0.9, max_group_size::Int64 = 100)
    label = 0
    train_images = []
    train_labels = Array{Int64}(undef, 0)
    test_images = []
    test_labels = Array{Int64}(undef, 0)
    assignment_dict = Dict()
    for folder in ProgressBar(readdir(path))
        if !startswith(folder, ".")
            folder_path = string(path,folder,"/")
            assignment_dict[folder] = label
            tmp_train_imgs, tmp_train_lbls, tmp_tst_imgs, tmp_tst_lbls = create_training_and_test_data(folder_path, label, ratio, max_group_size)
            train_images = vcat(train_images, tmp_train_imgs)
            train_labels = vcat(train_labels, tmp_train_lbls)
            test_images = vcat(test_images, tmp_tst_imgs)
            test_labels = vcat(test_labels, tmp_tst_lbls)
            label += 1
        end
    end
    return train_images, train_labels, test_images, test_labels, assignment_dict
end
