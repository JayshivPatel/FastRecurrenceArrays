using FastRecurrenceArrays, RecurrenceRelationshipArrays, SingularIntegrals, 
    ClassicalOrthogonalPolynomials, LinearAlgebra, GLMakie, FastGaussQuadrature;

x_in = collect(range(ComplexF32(-1.0), ComplexF32(1.0), 100));

x_left = collect(range(Float32(-1.0001), Float32(-1.5), 100));

x_right = collect(range(Float32(1.0001), Float32(1.5), 100));


x = x_left;

M = length(x); T = 100;

P = Legendre(); rec_P = ClassicalOrthogonalPolynomials.recurrencecoefficients(P);
f(x) = exp(x); f_N = expand(P, exp); 

differences_fixed = Vector{Float32}(undef, T);
differences_gauss = Vector{Float32}(undef, T);
baseline = [Float32(real(inv.(x₀ .- axes(P, 1)') * f_N)) for x₀ in x];

for i=2:T
    ff = Float32.(collect(f_N.args[2][1:i])); fixed = Float32.(ForwardInplace(ff, rec_P, x))
    f_g(x, z) = f(x)/(z-x); x_g, w_g = gausslegendre(i); gauss = [dot(w_g, f_g.(x_g, x₀)) for x₀ in x];

    differences_fixed[i] = norm(fixed .- baseline);
    differences_gauss[i] = norm(gauss .- baseline);
end

@. differences_fixed = clamp(differences_fixed, 0.1, 1e4);
@. differences_gauss = clamp(differences_gauss, 0.1, 1e4);

fig = Figure(size = (1000, 1000));
ax = Axis(fig[1, 1], xlabel=L"n", ylabel=L"Absolute Error");
fixed = lines!(ax, (1:T), differences_fixed, color=:red, linewidth=2);
gauss = lines!(ax, (1:T), differences_gauss, color=(:blue, 0.6), linewidth=2);

fig