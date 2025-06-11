using FastRecurrenceArrays, ClassicalOrthogonalPolynomials, GLMakie, CUDA;

side_length = 1000;

xs = range(Float32(-π / 2), Float32(π / 2), length=side_length);
ys = range(Float32(-π / 2), Float32(π / 2), length=side_length);

mesh = vec([x + y * im for x in xs, y in ys]);

P = Legendre();
N = 15;

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(40pt),
    linewidth=2,
    markersize=13,
    figure_padding=30,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

P = Legendre();

function cauchytransform(k)
    f = expand(P, x -> (cos(k + x)));
    ff = Float32.(collect(f.args[2][1:N-2]));
    st = real(ClenshawCauchy(N, mesh, ff));

    return reshape(st, side_length, side_length);
end;

function logtransform(k)
    f = expand(P, x -> (cos(k + x)));
    ff = Float32.(collect(f.args[2][1:N-2]));
    lt = ClenshawLogKernel(N, mesh, ff);

    return reshape(lt, side_length, side_length);
end;

fig = Figure(size=(1920, 1080));

k = Observable(-π);

st = @lift cauchytransform($k);
lt = @lift logtransform($k);


ax1 = Axis3(fig[1, 1], xlabel=L"Re(z)", ylabel=L"Im(z)", zlabel="", title=L"\text{Re}(\mathcal{C}[\cos](kz))", xlabeloffset=100, ylabeloffset=100);
surface!(ax1, xs, ys, st, colormap=:plasma, rasterize = 10);

ax2 = Axis3(fig[1, 2], xlabel=L"Re(z)", ylabel=L"Im(z)", zlabel="", title=L"\mathcal{L}[\cos](kz)", xlabeloffset=100, ylabeloffset=100);
surface!(ax2, xs, ys, lt, colormap=:viridis, rasterize = 10);

fig

frames = 600;

GLMakie.record(fig, "stillanimated.mp4", 1:frames; framerate=60) do i
    k[] = range(Float32(-π), Float32(π), length=frames)[i];
end;