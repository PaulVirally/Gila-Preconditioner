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


using Lux, ADTypes, AMDGPU, LuxCUDA, Optimisers, Printf, Random, Statistics, Zygote
using CairoMakie


rng = MersenneTwister()
#Random.seed!(rng, 12345)

#(x, y) = generate_data(rng)

df = CSV.read("data.csv", DataFrame)

# Extract input and output data for real parts
input_real = (Matrix(df[:, 1:64]) |> Array{Float32})'
output_real = (Matrix(df[:, 129:320]) |> Array{Float32})'

# Extract input and output data for imaginary parts
input_imag = (Matrix(df[:, 65:128]) |> Array{Float32})'
output_imag = (Matrix(df[:, 321:512]) |> Array{Float32})'

htanh(x) = min(max(x, -1), 1)

model = Chain(Dense(64 => 256, relu), Dense(256 => 256, relu), Dense(256 => 256, relu), Dense(256 => 192, htanh))

opt = Adam(0.001)

const loss_function = MSELoss()

tstate = Lux.Experimental.TrainState(rng, model, opt)

vjp_rule = AutoZygote()

function main(tstate::Lux.Experimental.TrainState, vjp, data, epochs)
    data = data .|> gpu_device()
    for epoch in 1:epochs
        _, loss, _, tstate = Lux.Experimental.single_train_step!(
            vjp, loss_function, data, tstate)
        if epoch % 50 == 1 || epoch == epochs
            @printf "Epoch: %3d \t Loss: %.5g\n" epoch loss
        end
    end
    return tstate
end

dev_cpu = cpu_device()
dev_gpu = gpu_device()

tstate = main(tstate, vjp_rule, (input_real, output_real), 250)
y_pred = dev_cpu(Lux.apply(tstate.model, dev_gpu(input_real), tstate.parameters, tstate.states)[1])

begin
    fig = Figure()
    ax = CairoMakie.Axis(fig[1, 1]; xlabel="Sample", ylabel="Value")

    for i in 1:5  # Plot first 5 samples for clarity
        lines!(ax, output_real[:, i]; color=:orange, label="Actual")
        lines!(ax, y_pred[:, i]; color=:blue, linestyle=:dash, label="Predicted")
    end

    axislegend(ax)
    fig
end