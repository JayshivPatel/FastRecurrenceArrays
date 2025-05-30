using FastRecurrenceArrays, ClassicalOrthogonalPolynomials, CairoMakie;

side_length = 1000;

xs = range(Float32(-π / 2), Float32(π / 2), length=side_length);
ys = range(Float32(-π / 2), Float32(π / 2), length=side_length);

mesh = vec([x + y * im for x in xs, y in ys]);

P = Legendre();
N = 15;
f = expand(P, x -> (cos(x)));
ff = collect(f.args[2][1:N-2]);

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=30,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

P = Legendre();

function cauchytransform()
    st = real.(ClenshawCauchy(N, mesh, ff))
    return reshape(st, side_length, side_length)
end;

function logtransform()
    lt = ClenshawLogKernel(N, mesh, ff)
    return reshape(lt, side_length, side_length)
end;

fig = Figure(size=(6.27inch, 3inch));

st = cauchytransform();
lt = logtransform();


ax1 = Axis3(fig[1, 1], xlabel=L"Re(z)", ylabel=L"Im(z)", zlabel="", title=L"\text{Re}(\mathcal{C}[\cos](z))");
surface!(ax1, xs, ys, st, colormap=:plasma);

ax2 = Axis3(fig[1, 2], xlabel=L"Re(z)", ylabel=L"Im(z)", zlabel="", title=L"\mathcal{L}[\cos](z)");
surface!(ax2, xs, ys, lt, colormap=:viridis);

fig

save("transforms.png", fig)