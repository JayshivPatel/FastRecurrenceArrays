using FastRecurrenceArrays, ClassicalOrthogonalPolynomials, GLMakie

side_length = 500;

# Generate a 2D square mesh from -π/2 to π/2 in the complex plane
xs = range(-pi/2, pi/2, length=side_length);
ys = range(-pi/2, pi/2, length=side_length);

mesh = vec([x + y*im for x in xs, y in ys]);

N = 15;
set_theme!(fontsize = 20)

a = 1; b = 1; c = 1;
P = Legendre(); x = axes(P, 1);

function stieltjestransform(a, b, c)
    # Choose f(x) = ax² + bx + c
    f = expand(P, x -> a*x^2 + b*x + c);
    ff = collect(Float32.(f.args[2][1:N-2]));

    z = GPUInplaceStieltjes(N, mesh, ff).f;
    z = reshape(z, side_length, side_length);
    return z
end;

fig = Figure(size = (1000, 1000));

rg = -10.0f0:0.1f0:10.0f0;

sg = SliderGrid(
    fig[2, 1],
    (label = L"a", range = rg, format = "{:.1f}", startvalue = 1),
    (label = L"b", range = rg, format = "{:.1f}", startvalue = 0),
    (label = L"c", range = rg, format = "{:.1f}", startvalue = 1),
);

a = sg.sliders[1].value;
b = sg.sliders[2].value;
c = sg.sliders[3].value;

z = @lift stieltjestransform($a, $b, $c);


ax = Axis3(fig[1, 1], xlabel=L"Re(z)", ylabel=L"Im(z)", title=L"\mathcal{C}_{(-1, 1)}[az^2+bz+c]");
surface!(ax, xs, ys, z, colormap=:viridis);

fig

save("quadraticsurface.png", fig);
