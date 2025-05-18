using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals,
    ClassicalOrthogonalPolynomials, LinearAlgebra, CairoMakie, FastGaussQuadrature;

x = range(ComplexF64(1.0001), ComplexF64(10.0), 1000);

log_g(x, z) = log(z - x) * exp(x);
c_g(x, z) = exp(x) / (x - z);
r = [3:9; unique(floor.(Int, logrange(1e1, 1e4, 100)))];

P = Legendre();
f_N = expand(P, exp);

function logtransform(n)
    inplace = Vector{Float64}(undef, length(x))
    ff = Float64.(collect(f_N.args[2][1:n]))

    inplace .= InplaceLogKernel(n, x, ff)
    return inplace
end;

function cauchytransform(n)
    inplace = Vector{ComplexF64}(undef, length(x))
    ff = Float64.(collect(f_N.args[2][1:n]))

    inplace .= InplaceCauchy(n, x, ff)
    return inplace
end;

baseline_log = [Float64.(log.(abs.(x₀ .- axes(P, 1)')) * f_N) for x₀ in x];
differences1_log = Vector{Float64}(undef, length(r));
differences2_log = Vector{Float64}(undef, length(r));

for (i, n) in enumerate(r)
    lt = logtransform(n)
    valid = .!(isnan.(lt) .| isnan.(baseline_log))
    num_valid = length(valid)
    differences1_log[i] = 1 / num_valid * norm(lt[valid] .- baseline_log[valid])

    x_g, w_g = gausslegendre(n)
    g = [Float64.(real(dot(w_g, log_g.(x_g, x₀)))) for x₀ in x]
    valid = .!(isnan.(g) .| isnan.(baseline_log))
    num_valid = length(valid)
    differences2_log[i] = 1 / num_valid * norm(g[valid] .- baseline_log[valid])
end;

baseline_c = [(-inv(2π*im) * (inv.(x₀ .- axes(P, 1)') * f_N)) for x₀ in x];
differences1_c = Vector{Float64}(undef, length(r));
differences2_c = Vector{Float64}(undef, length(r));

for (i, n) in enumerate(r)
    c = cauchytransform(n)
    valid = .!(isnan.(c) .| isnan.(baseline_c))
    num_valid = length(valid)
    differences1_c[i] = 1 / num_valid * norm(c[valid] .- baseline_c[valid])

    x_g, w_g = gausslegendre(n)
    g = [inv(2π*im) * dot(w_g, c_g.(x_g, x₀)) for x₀ in x]
    valid = .!(isnan.(g) .| isnan.(baseline_c))
    num_valid = length(valid)
    differences2_c[i] = 1 / num_valid * norm(g[valid] .- baseline_c[valid])
end;

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
    figure_padding=12,
);

fig = Figure(size=(6.28inch, 6.28inch));

ax1 = Axis(
    fig[1, 1],
    xlabel=L"n",
    ylabel="Mean Squared Error",
    title="LogKernel convergence of forward-inplace vs\nFastGaussQuadrature outside the integral domain",
    yscale=log10,
    xscale=log10,
);

inplace_log = lines!(ax1, r, differences1_log, linewidth=2);
gauss_log = lines!(ax1, r, differences2_log, linewidth=2);

ax2 = Axis(
    fig[3, 1],
    xlabel=L"n",
    ylabel="Mean Squared Error",
    title="Cauchy convergence of forward-inplace vs\nFastGaussQuadrature outside the integral domain",
    yscale=log10,
    xscale=log10
);

inplace_c = lines!(ax2, r, differences1_c, linewidth=2);
gauss_c = lines!(ax2, r, differences2_c, linewidth=2);

Legend(
    fig[2, 1],
    [inplace_log, gauss_log],
    ["forward-inplace", "FastGaussQuadrature"],
    orientation=:horizontal,
    framevisible=false,
    groupgap=50
);

fig

save("abs-outside.svg", fig);