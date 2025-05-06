using FastRecurrenceArrays, ClassicalOrthogonalPolynomials, CairoMakie

side_length = 1000;

# Generate a 2D square mesh from -π/2 to π/2 in the complex plane
xs = range(-pi/2, pi/2, length=side_length);
ys = range(-pi/2, pi/2, length=side_length);

mesh = vec([x + y*im for x in xs, y in ys]);

N = 15;

# Choose f(x) = sin(x)²cos(x)²
P = Legendre(); x = axes(P, 1); f = expand(P, x -> (sin(x)^2 * cos(x)^2)); ff = collect(f.args[2][1:N-2]);

z1 = InplaceStieltjes(N, mesh, ff).f;
z2 = InplaceLogKernel(N, mesh, ff).f;

z1 = reshape(z1, side_length, side_length);
z2 = reshape(z2, side_length, side_length);

set_theme!(fontsize = 20)

fig = Figure(size = (1000, 800));

ax1 = Axis(fig[1, 1], xlabel=L"Re(z)", ylabel=L"Im(z)", title=L"\mathcal{L}[\sin(z)^2\cos(z)^2]");
contour!(ax1, xs, ys, z1, levels=15, colormap=:plasma);
lines!(ax1, [-1, 1], [0, 0], color = :black, linewidth = 1.75);

ax2 = Axis3(fig[1, 2], xlabel=L"Re(z)", ylabel=L"Im(z)", title=L"\mathcal{L}[\sin(z)^2\cos(z)^2]");
surface!(ax2, xs, ys, z1, colormap=:plasma);

ax3 = Axis(fig[2, 1], xlabel=L"Re(z)", ylabel=L"Im(z)", title=L"\mathcal{C}_{(1, 1)}[\sin(z)^2\cos(z)^2]");
contour!(ax3, xs, ys, z2, levels=15, colormap=:viridis);
lines!(ax3, [-1, 1], [0, 0], color = :black, linewidth = 1.75);

ax4 = Axis3(fig[2, 2], xlabel=L"Re(z)", ylabel=L"Im(z)", title=L"\mathcal{C}_{(1, 1)}[\sin(z)^2\cos(z)^2]");
surface!(ax4, xs, ys, z2, colormap=:viridis);

fig

save("sin^2cos^2.png", fig);
