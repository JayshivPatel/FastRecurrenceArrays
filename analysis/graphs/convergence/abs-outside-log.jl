using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, GLMakie, FastGaussQuadrature;

x = range(ComplexF32(1.0001), ComplexF32(10.0), 1000);

f_g(x, z) = log(z-x)*exp(x); r = 3:100;

P = Legendre(); f_N = expand(P, exp);

function logtransform(n)
    fixed = Vector{Float32}(undef, length(x));
    ff = Float32.(collect(f_N.args[2][1:n]));

    fixed .= FixedLogKernel(n, x, ff);
    return fixed;
end;

baseline = [Float32.(log.(abs.(x₀ .- axes(P, 1)')) * f_N) for x₀ in x];
differences1 = Vector{Float32}(undef, length(r));
differences2 = Vector{Float32}(undef, length(r));

for i=r
    lt = logtransform(i);
    valid = .!(isnan.(lt) .| isnan.(baseline));
    num_valid = length(valid)
    differences1[i - 2] = 1/num_valid * norm(lt[valid] .- baseline[valid]);

    x_g, w_g = gausslegendre(i);
    g = [real(dot(w_g, f_g.(x_g, x₀))) for x₀ in x];
    valid = .!(isnan.(g) .| isnan.(baseline));
    differences2[i - 2] = 1/num_valid * norm(g[valid] .- baseline[valid]);
end;

fig = Figure(size = (1000, 1000));
ax1 = Axis(fig[1, 1], xlabel=L"n", ylabel="Average Absolute Difference", yscale=log10, xscale=log10);
fixed = lines!(ax1, r, differences1, color=(:red, 0.9), linewidth=2);
gauss = lines!(ax1, r, differences2, color=(:blue, 0.7), linewidth=2);
axislegend(ax1, [fixed, gauss], ["InplaceLogKernel", "FastGaussQuadrature"], position = :rt, orientation=:vertical);
set_theme!(fontsize = 25)
fig

save("abs-outside-log.png", fig);