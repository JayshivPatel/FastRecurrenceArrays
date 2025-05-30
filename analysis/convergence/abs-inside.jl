using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = range(ComplexF64(-1.0), ComplexF64(1.0), 1000);

log_g(x, z) = log(z - x) * exp(x);
c_g(x, z) = exp(x) / (x - z);
r = (3:25);

P = Legendre();
f_N = expand(P, exp);

function logforward(n)
    vec = Vector{Float64}(undef, length(x))
    ff = transform(P[:, 1:n], exp);

    vec .= FixedLogKernel(n, x, ff)
    return vec
end;

function cauchyforward(n)
    vec = Vector{ComplexF64}(undef, length(x))
    ff = transform(P[:, 1:n], exp);

    vec .= abs.(FixedCauchy(n, x, ff))
    return vec
end;

function loginplace(n)
    vec = Vector{Float64}(undef, length(x))
    ff = transform(P[:, 1:n], exp);

    vec .= InplaceLogKernel(n, x, ff)
    return vec
end;

function cauchyinplace(n)
    vec = Vector{ComplexF64}(undef, length(x))
    ff = transform(P[:, 1:n], exp);

    vec .= abs.(InplaceCauchy(n, x, ff))
    return vec
end;

function logclenshaw(n)
    vec = Vector{Float64}(undef, length(x))
    ff = transform(P[:, 1:n], exp);

    vec .= ClenshawLogKernel(n, x, ff)
    return vec
end;

function cauchyclenshaw(n)
    vec = Vector{ComplexF64}(undef, length(x))
    ff = transform(P[:, 1:n], exp);

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
    differencesf_log[i] = 1 / num_valid * norm(f[valid] .- baseline_log[valid], 1)

    inp = loginplace(n)
    valid = .!(isnan.(inp) .| isnan.(baseline_log))
    num_valid = length(valid)
    differencesi_log[i] = 1 / num_valid * norm(inp[valid] .- baseline_log[valid], 1)

    c = logclenshaw(n)
    valid = .!(isnan.(c) .| isnan.(baseline_log))
    num_valid = length(valid)
    differencesc_log[i] = 1 / num_valid * norm(c[valid] .- baseline_log[valid], 1)
end;

baseline_c = [Float64.(abs.((-inv.(x₀ .- axes(P, 1)') * f_N))) for x₀ in x];
differencesf_c = Vector{Float64}(undef, length(r));
differencesi_c = Vector{Float64}(undef, length(r));
differencesc_c = Vector{Float64}(undef, length(r));


for (i, n) in enumerate(r)
    f = cauchyforward(n)
    valid = .!(isnan.(f) .| isnan.(baseline_c))
    num_valid = length(valid)
    differencesf_c[i] = 1 / num_valid * norm(f[valid] .- baseline_c[valid], 1)

    inp = cauchyinplace(n)
    valid = .!(isnan.(inp) .| isnan.(baseline_c))
    num_valid = length(valid)
    differencesi_c[i] = 1 / num_valid * norm(inp[valid] .- baseline_c[valid], 1)

    c = cauchyclenshaw(n)
    valid = .!(isnan.(c) .| isnan.(baseline_c))
    num_valid = length(valid)
    differencesc_c[i] = 1 / num_valid * norm(c[valid] .- baseline_c[valid], 1)
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
    ylabel="Mean Absolute Error",
    title=L"\mathcal{L}[\exp](x):  x \in [-1,1]",
    yscale=log10,
);


f = lines!(ax1, r, differencesf_log);
# shift the colours
scatter!(ax1, [0], [0], visible=false);
lines!(ax1, [0], [0], visible=false);
lines!(ax1, [0], [0], visible=false);

i = scatter!(ax1, r[1:2:end], differencesi_log[1:2:end]);
c = scatter!(ax1, r[2:2:end], differencesc_log[2:2:end]);


ax2 = Axis(
    fig[1, 1],
    xlabel=L"n",
    ylabel="Mean Absolute Error",
    title=L"|\mathcal{C}[\exp](x)|:  x \in [-1,1]",
    yscale=log10,
);

lines!(ax2, r, differencesf_c);
# shift the colours
scatter!(ax2, [0], [0], visible=false);
lines!(ax2, [0], [0], visible=false);
lines!(ax2, [0], [0], visible=false);

scatter!(ax2, r[1:2:end], differencesi_c[1:2:end]);
scatter!(ax2, r[2:2:end], differencesc_c[2:2:end]);

Legend(
    fig[3, 1],
    [f, i, c],
    ["forward", "forward_inplace", "clenshaw"],
    orientation=:horizontal,
    framevisible=false,
    labelfont="TeX Gyre Cursor",
    groupgap=50
);

fig

save("abs-inside.svg", fig);
