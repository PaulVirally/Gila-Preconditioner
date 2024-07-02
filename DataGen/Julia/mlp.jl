using Pkg
Pkg.add("Lux")
Pkg.add("Zygote")
Pkg.add("Statistics")
Pkg.add("Random")
Pkg.add("Optimisers")
Pkg.add("ADTypes")
Pkg.add("AMDGPU")
Pkg.add("LuxCUDA")
Pkg.add("CairoMakie")
Pkg.add("Printf")
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("Flux")


using Lux, ADTypes, AMDGPU, LuxCUDA, Optimisers, Printf, Random, Statistics, Zygote
using CairoMakie
using Flux.Data: DataLoader

# Extract input and output data for real parts
input_real = Matrix(df[:, 1:64]) |> Array{Float32, 2}
output_real = Matrix(df[:, 129:320]) |> Array{Float32, 2}

# Extract input and output data for imaginary parts
input_imag = Matrix(df[:, 65:128]) |> Array{Float32, 2}
output_imag = Matrix(df[:, 321:512]) |> Array{Float32, 2}

# Split data into training and test sets
train_ratio = 0.8
n_train = Int(floor(train_ratio * size(input_real, 1)))

train_input_real = input_real[1:n_train, :]
train_output_real = output_real[1:n_train, :]
test_input_real = input_real[n_train+1:end, :]
test_output_real = output_real[n_train+1:end, :]

train_input_imag = input_imag[1:n_train, :]
train_output_imag = output_imag[1:n_train, :]
test_input_imag = input_imag[n_train+1:end, :]
test_output_imag = output_imag[n_train+1:end, :]

# Define the MLP model architecture
function build_mlp()
    Lux.Chain(
        Dense(64, 128, relu),
        Dense(128, 256, relu),
        Dense(256, 192)
    )
end

input_size_real = size(train_input_real, 2)
output_size_real = size(train_output_real, 2)
model_real = build_mlp()

input_size_imag = size(train_input_imag, 2)
output_size_imag = size(train_output_imag, 2)
model_imag = build_mlp()

# Training parameters
epochs = 100
batch_size = 5

# Training function
function train_model(model, train_input, train_output)
    
    optimiser = Adam(0.001)
    loss_function = MSELoss()

    data = DataLoader((train_input, train_output), batchsize=batch_size, shuffle=true)

    for epoch in 1:epochs
        for (x, y) in data
            loss, grads = Flux.with_gradient(model) do model
                loss_function(model(x), y)
            end
            optimiser = Flux.update!(optimiser, model, grads)
        end
        println("Epoch $epoch complete")
    end
end

# Train the real model
println("Training the real model...")
train_model(model_real, train_input_real, train_output_real)

# Train the imaginary model
println("Training the imaginary model...")
train_model(model_imag, train_input_imag, train_output_imag)

println("Training complete")