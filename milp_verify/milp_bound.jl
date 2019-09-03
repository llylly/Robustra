
@everywhere using MIPVerify
@everywhere using Gurobi
@everywhere using MAT

using ArgParse

function buildModel(dataset:: String, scale:: String, weights:: Dict)
    println("Building $dataset $scale...")
    if dataset == "MNIST" && scale == "small"
        nn = Sequential([
            get_conv_params(weights, "0", (4, 4, 1, 16), expected_stride = 2),
            ReLU(interval_arithmetic),
            get_conv_params(weights, "2", (4, 4, 16, 32), expected_stride = 2),
            ReLU(),
            Flatten([1, 3, 2, 4]),
            get_matrix_params(weights, "5", (1568, 100)),
            ReLU(),
            get_matrix_params(weights, "7", (100, 10))], "$(dataset)_$(scale)")
        return nn
    elseif dataset == "MNIST" && scale == "large"
        nn = Sequential([
            get_conv_params(weights, "0", (3, 3, 1, 32), expected_stride = 1),
            ReLU(interval_arithmetic),
            get_conv_params(weights, "2", (4, 4, 32, 32), expected_stride = 2),
            ReLU(),
            get_conv_params(weights, "4", (3, 3, 32, 64), expected_stride = 1),
            ReLU(),
            get_conv_params(weights, "6", (4, 4, 64, 64), expected_stride = 2),
            ReLU(),
            Flatten([1, 3, 2, 4]),
            get_matrix_params(weights, "9", (3136, 512)),
            ReLU(),
            get_matrix_params(weights, "11", (512, 512)),
            ReLU(),
            get_matrix_params(weights, "13", (512, 10))], "$(dataset)_$(scale)")
        return nn
    elseif dataset == "MNIST" && scale == "base"
        nn = Sequential([
            get_conv_params(weights, "0", (4, 4, 1, 32), expected_stride = 2),
            ReLU(interval_arithmetic),
            get_conv_params(weights, "2", (4, 4, 32, 64), expected_stride = 2),
            ReLU(),
            Flatten([1, 3, 2, 4]),
            get_matrix_params(weights, "5", (3136, 1024)),
            ReLU(),
            get_matrix_params(weights, "7", (1024, 10))], "$(dataset)_$(scale)")
        return nn
    elseif dataset == "CIFAR10" && scale == "small"
        nn = Sequential([
            get_conv_params(weights, "0", (4, 4, 3, 16), expected_stride = 2),
            ReLU(interval_arithmetic),
            get_conv_params(weights, "2", (4, 4, 16, 32), expected_stride = 2),
            ReLU(),
            Flatten([1, 3, 2, 4]),
            get_matrix_params(weights, "5", (2048, 100)),
            ReLU(),
            get_matrix_params(weights, "7", (100, 10))], "$(dataset)_$(scale)")
        return nn
    elseif dataset == "CIFAR10" && scale == "large"
        nn = Sequential([
            get_conv_params(weights, "0", (3, 3, 3, 32), expected_stride = 1),
            ReLU(interval_arithmetic),
            get_conv_params(weights, "2", (4, 4, 32, 32), expected_stride = 2),
            ReLU(),
            get_conv_params(weights, "4", (3, 3, 32, 64), expected_stride = 1),
            ReLU(),
            get_conv_params(weights, "6", (4, 4, 64, 64), expected_stride = 2),
            ReLU(),
            Flatten([1, 3, 2, 4]),
            get_matrix_params(weights, "9", (4096, 512)),
            ReLU(),
            get_matrix_params(weights, "11", (512, 512)),
            ReLU(),
            get_matrix_params(weights, "13", (512, 10))], "$(dataset)_$(scale)")
        return nn
    elseif dataset == "CIFAR10" && scale == "resnet"
        nn = SkipSequential([
            get_conv_params(weights, "0", (3, 3, 3, 16), expected_stride=1),
            ReLU(interval_arithmetic),
            get_conv_params(weights, "2/Ws/0", (3, 3, 16, 16), expected_stride=1),
            ReLU(),
            SkipBlock([
                get_conv_params(weights, "4/Ws/0", (1, 1, 16, 16), expected_stride=1),
                Zero(),
                get_conv_params(weights, "4/Ws/2", (3, 3, 16, 16), expected_stride=1)
            ]),
            ReLU(),
            get_conv_params(weights, "6/Ws/0", (3, 3, 16, 16), expected_stride=1),
            ReLU(),
            SkipBlock([
                get_conv_params(weights, "8/Ws/0", (1, 1, 16, 16), expected_stride=1),
                Zero(),
                get_conv_params(weights, "8/Ws/2", (3, 3, 16, 16), expected_stride=1)
            ]),
            ReLU(),
            get_conv_params(weights, "10/Ws/0", (4, 4, 16, 32), expected_stride=2),
            ReLU(),
            SkipBlock([
                get_conv_params(weights, "12/Ws/0", (2, 2, 16, 32), expected_stride=2),
                Zero(),
                get_conv_params(weights, "12/Ws/2", (3, 3, 32, 32), expected_stride=1)
            ]),
            ReLU(),
            get_conv_params(weights, "14/Ws/0", (4, 4, 32, 64), expected_stride=2),
            ReLU(),
            SkipBlock([
                get_conv_params(weights, "16/Ws/0", (2, 2, 32, 64), expected_stride=2),
                Zero(),
                get_conv_params(weights, "16/Ws/2", (3, 3, 64, 64), expected_stride=1)
            ]),
            ReLU(),
            Flatten([1, 3, 2, 4]),
            get_matrix_params(weights, "19", (4096, 1000)),
            ReLU(),
            get_matrix_params(weights, "21", (1000, 10))], "$(dataset)_$(scale)")
        return nn
    end
end

function initModels(path = "final_models":: String)
    files = [fname for fname in readdir(path) if endswith(fname, ".mat")]
    models = Dict{String, NeuralNet}()
    configs = Dict{String, Any}()
    for f in files
        things = [String(x) for x in split((rsplit(f, "."; limit=2)[1]), "_")]
        weights = matread(path * "/" * f)
        model = buildModel(things[1], things[3], weights)
        if model != nothing
            push!(models, f => model)
            push!(configs, f => things)
        end
    end
    return models, configs
end

@everywhere function get_label(y::Array{<:Real, 1}, test_index::Integer)::Int
    return y[test_index]
end

@everywhere function get_image(x::Array{T, 4}, test_index::Integer)::Array{T, 4} where {T<:Real}
    return x[test_index:test_index, :, :, :]
end

@everywhere function get_max_index(
    x::Array{<:Real, 1})::Integer
    return findmax(x)[2]
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--sampleNum"
            help = "number of samples"
            arg_type = Int
            default = 99999
        "--workers"
            help = "number of processes"
            arg_type = Int
            default = 1
        "dset"
            help = "dataset"
            arg_type = String
            required = true
        "epsilon"
            help = "epsilon"
            arg_type = String
            required = true
        "scale"
            help = "model scale / type"
            arg_type = String
            required = true
    end
    return parse_args(s)
end

@everywhere function work(work_id, dataset, datasetName, epsilon, m, l, r, tot)
    println("Proc. start $(work_id) $(l) $(r) $(tot)")

    tot_samples = 0
    clean_correct = 0
    robust_correct = 0
    tot_time = 0.0
    if l <= tot
        r = min(r, tot)
        tot_samples = max(r - l + 1, 0)
        for sample_index in l: r
            x0 = get_image(dataset.test.images, sample_index)
            if datasetName == "CIFAR10"
                # normalize
                mean = [0.485, 0.456, 0.406]
                std = [0.225, 0.225, 0.225]
                for i in 1: 3
                    x0[:, :, :, i] = (x0[:, :, :, i] - mean[i]) / std[i]
                end
                println(size(x0))
                println(minimum(x0))
                println(maximum(x0))
            end
            actual_label = get_label(dataset.test.labels, sample_index)
            actual_index = actual_label + 1
            predicted_output = x0 |> m
            predicted_index = predicted_output |> get_max_index
            sample_time = 0.0
            if predicted_index == actual_index
                clean_correct += 1
                try
                    d = find_adversarial_example(m, x0,
                        actual_index, GurobiSolver(TimeLimit=1200, BestObjStop=epsilon, OutputFlag=0, Threads=1),
                        invert_target_selection=true,
                        pp=MIPVerify.LInfNormBoundedPerturbationFamily(epsilon),
                        norm_order=Inf, tolerance=0.0, rebuild=false, tightening_algorithm=lp,
                        tightening_solver=GurobiSolver(TimeLimit=20, OutputFlag=0, Threads=1),
                        solve_if_predicted_in_targeted = false)
                    sample_time = float(d[:TotalTime])
                    if d[:SolveStatus] == :InfeasibleOrUnbounded
                        robust_correct += 1
                    end
                finally
                end
            end
            tot_time += sample_time
            println("\n\033[31m[worker #$(work_id)] [$(sample_index-l+1)/$(tot_samples)] Robust: $(robust_correct / (sample_index-l+1)) Clean: $(clean_correct / (sample_index-l+1)) Time: $(tot_time / (sample_index-l+1))\033[0m\n")
        end
    end
    return [tot_samples, clean_correct, robust_correct, tot_time]
end

# ----- main -----
function main()
    args = parse_commandline()

    io = open("$(args["dset"])_$(args["epsilon"])_$(args["scale"])_milp.test", "a")

    println("Start...")
    models, configs = initModels()

    num_samples = args["sampleNum"]
    m = nothing
    for (k, v) in models
        if configs[k][1] == args["dset"] && configs[k][2] == args["epsilon"] && configs[k][3] == args["scale"]
            m = v
            break
        end
    end
    if args["dset"] == "MNIST" && args["epsilon"] == "0.1" && args["scale"] == "baseline"
        m = MIPVerify.get_example_network_params("MNIST.WK17a_linf0.1_authors")
    end
    @assert m != nothing
    println("Now working on config $(args["dset"]) $(args["epsilon"]) $(args["scale"])")
    dataset = MIPVerify.read_datasets(lowercase(args["dset"]))
    num_samples = min(num_samples, MIPVerify.num_samples(dataset.test))

    workers = args["workers"]
    samplePerWorker = Int(ceil(num_samples / workers))
    runs = [@spawn work(i, dataset, args["dset"], float(args["epsilon"]), m, (i-1) * samplePerWorker + 1, i * samplePerWorker, num_samples) for i in 1: workers]
    anss = [fetch(item) for item in runs]

    print(anss)

    stat = [sum([anss[i][j] for i in 1: workers]) for j in 1: 4]
    @assert stat[1] == num_samples
    println("------")
    println("clean error: $(1.0 - stat[2] / num_samples)")
    println("robust error: $(1.0 - stat[3] / num_samples)")
    println("avg time: $(stat[4] / num_samples)")
    println("------")

    write(io, "$(num_samples)\nclean error: $(1.0 - stat[2] / num_samples)\nrobust error: $(1.0 - stat[3] / num_samples)\navg time: $(stat[4] / num_samples)\n")
    close(io)
end

main()
