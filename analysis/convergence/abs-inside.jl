using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = range(ComplexF64(-1.0), ComplexF64(1.0), 1000);

log_g(x, z) = log(z - x) * exp(x);
c_g(x, z) = exp(x) / (x - z);
r = [3:9; unique(floor.(Int, logrange(1e1, 1e4, 100)))];

P = Legendre();
f_N = expand(P, exp);

function logforward(n)
    vec = Vector{Float64}(undef, length(x))
    ff = Float64.(collect(f_N.args[2][1:n]))

    vec .= FixedLogKernel(n, x, ff)
    return vec
end;

function cauchyforward(n)
    vec = Vector{ComplexF64}(undef, length(x))
    ff = Float64.(collect(f_N.args[2][1:n]))

    vec .= abs.(FixedCauchy(n, x, ff))
    return vec
end;

function loginplace(n)
    vec = Vector{Float64}(undef, length(x))
    ff = Float64.(collect(f_N.args[2][1:n]))

    vec .= InplaceLogKernel(n, x, ff)
    return vec
end;

function cauchyinplace(n)
    vec = Vector{ComplexF64}(undef, length(x))
    ff = Float64.(collect(f_N.args[2][1:n]))

    vec .= abs.(InplaceCauchy(n, x, ff))
    return vec
end;

function logclenshaw(n)
    vec = Vector{Float64}(undef, length(x))
    ff = Float64.(collect(f_N.args[2][1:n]))

    vec .= ClenshawLogKernel(n, x, ff)
    return vec
end;

function cauchyclenshaw(n)
    vec = Vector{ComplexF64}(undef, length(x))
    ff = Float64.(collect(f_N.args[2][1:n]))

    vec .= abs.(ClenshawCauchy(n, x, ff))
    return vec
end;

baseline_log = [Float64.(log.(abs.(x₀ .- axes(P, 1)')) * f_N) for x₀ in x];
differencesf_log = Vector{Float64}(undef, length(r));
differencesi_log = Vector{Float64}(undef, length(r));
differencesc_log = Vector{Float64}(undef, length(r));
differencesg_log = Vector{Float64}(undef, length(r));

for (i, n) in enumerate(r)
    f = logforward(n)
    valid = .!(isnan.(f) .| isnan.(baseline_log))
    num_valid = length(valid)
    differencesf_log[i] = 1 / num_valid * norm(f[valid] .- baseline_log[valid])

    inp = loginplace(n)
    valid = .!(isnan.(inp) .| isnan.(baseline_log))
    num_valid = length(valid)
    differencesi_log[i] = 1 / num_valid * norm(inp[valid] .- baseline_log[valid])

    c = logclenshaw(n)
    valid = .!(isnan.(c) .| isnan.(baseline_log))
    num_valid = length(valid)
    differencesc_log[i] = 1 / num_valid * norm(c[valid] .- baseline_log[valid])

    x_g, w_g = gausslegendre(n)
    g = [Float64.(real(dot(w_g, log_g.(x_g, x₀)))) for x₀ in x]
    valid = .!(isnan.(g) .| isnan.(baseline_log))
    num_valid = length(valid)
    differencesg_log[i] = 1 / num_valid * norm(g[valid] .- baseline_log[valid])
end;

baseline_c = [Float64.(abs.((-inv.(x₀ .- axes(P, 1)') * f_N))) for x₀ in x];
differencesf_c = Vector{Float64}(undef, length(r));
differencesi_c = Vector{Float64}(undef, length(r));
differencesc_c = Vector{Float64}(undef, length(r));
differencesg_c = Vector{Float64}(undef, length(r));


for (i, n) in enumerate(r)
    f = cauchyforward(n)
    valid = .!(isnan.(f) .| isnan.(baseline_c))
    num_valid = length(valid)
    differencesf_c[i] = 1 / num_valid * norm(f[valid] .- baseline_c[valid])

    inp = cauchyinplace(n)
    valid = .!(isnan.(inp) .| isnan.(baseline_c))
    num_valid = length(valid)
    differencesi_c[i] = 1 / num_valid * norm(inp[valid] .- baseline_c[valid])

    c = cauchyclenshaw(n)
    valid = .!(isnan.(c) .| isnan.(baseline_c))
    num_valid = length(valid)
    differencesc_c[i] = 1 / num_valid * norm(c[valid] .- baseline_c[valid])

    x_g, w_g = gausslegendre(n)
    g = [abs.(dot(w_g, c_g.(x_g, x₀))) for x₀ in x]
    valid = .!(isnan.(g) .| isnan.(baseline_c))
    num_valid = length(valid)
    differencesg_c[i] = 1 / num_valid * norm(g[valid] .- baseline_c[valid])
end;

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=12,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.5inch, 5inch));

ax1 = Axis(
    fig[2, 1],
    xlabel=L"n",
    ylabel="Mean Squared Error",
    title="Log Transform",
    yscale=log10,
    xscale=log10,
);


f = lines!(ax1, r, differencesf_log);
# shift the colours
scatter!(ax1, [0], [0], visible=false);
lines!(ax1, [0], [0], visible=false);
lines!(ax1, [0], [0], visible=false);

i = scatter!(ax1, r[1:7:end], differencesi_log[1:7:end]);
c = scatter!(ax1, r[4:7:end], differencesc_log[4:7:end]);
g = lines!(ax1, r, differencesg_log);


ax2 = Axis(
    fig[1, 1],
    xlabel=L"n",
    ylabel="Mean Squared Error",
    title="Cauchy Transform",
    yscale=log10,
    xscale=log10,
);

lines!(ax2, r, differencesf_c);
# shift the colours
scatter!(ax2, [0], [0], visible=false);
lines!(ax2, [0], [0], visible=false);
lines!(ax2, [0], [0], visible=false);

scatter!(ax2, r[1:7:end], differencesi_c[1:7:end]);
scatter!(ax2, r[4:7:end], differencesc_c[4:7:end]);
lines!(ax2, r, differencesg_c);

Legend(
    fig[3, 1],
    [f, i, c, g],
    ["forward", "forward-inplace", "clenshaw", "gauss-legendre"],
    orientation=:horizontal,
    framevisible=false,
    groupgap=50
);

fig

save("abs-inside.svg", fig);
