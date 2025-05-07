using FastRecurrenceArrays, ClassicalOrthogonalPolynomials, GLMakie;

side_length = 100;

xs = range(-pi/2, pi/2, length=side_length);

N = 15;
set_theme!(fontsize = 20)

a = 1; b = 1; c = 1;
P = Legendre(); x = axes(P, 1);

function stieltjestransform(a, b, c)
    # Choose f(x) = ax² + bx + c
    f = expand(P, x -> a*x^2 + b*x + c);
    ff = collect(Float32.(f.args[2][1:N-2]));

    z = GPUInplaceStieltjes(N, xs .+ 0im, ff).f;
    return z
end;

function quadratic(a, b, c, t)
    map(x -> real((a*x^2 + b*x + c)/(t - x)), xs)
end;

fig = Figure(size = (1000, 1000));

rg = -1.0f0:0.1f0:1.0f0;

sg = SliderGrid(
    fig[2, 1:2],
    (label = L"a", range = rg, format = "{:.1f}", startvalue = 1),
    (label = L"b", range = rg, format = "{:.1f}", startvalue = 0),
    (label = L"c", range = rg, format = "{:.1f}", startvalue = 1),
    (label = L"t", range = -2:0.1:2, format = "{:.1f}", startvalue = 0),
);

a = sg.sliders[1].value;
b = sg.sliders[2].value;
c = sg.sliders[3].value;
t = sg.sliders[4].value;

fx = @lift quadratic($a, $b, $c, $t);

idxs = findall(x -> -1 ≤ x ≤ 1, xs);
xband = xs[idxs];
yband = @lift $fx[idxs];

ax1 = Axis(fig[1, 1], xlabel=L"Re(z)", ylabel=L"\int_{-1}^1\frac{az^2+bz+c}{z-t}");
band!(ax1, xband, yband, 0, color=(:dodgerblue, 0.2));
lines!(ax1, xs, fx, color=:dodgerblue);

z = @lift stieltjestransform($a, $b, $c);

t_idx = @lift argmin(abs.(xs .- $t));
marker_x = @lift xs[$t_idx];
marker_y = @lift $z[$t_idx];

ax2 = Axis(fig[1, 2], xlabel=L"Re(z)", ylabel=L"\mathcal{C}_{(1, 1)}[az^2+bz+c]");
lines!(ax2, xs, z, color=:red);
scatter!(ax2, marker_x, marker_y; color=:dodgerblue, marker=:x, markersize=20);

fig

save("quadratic.png", fig);