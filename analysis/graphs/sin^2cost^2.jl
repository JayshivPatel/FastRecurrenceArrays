using FastRecurrenceArrays, ClassicalOrthogonalPolynomials, GLMakie, CUDA

side_length = 100;

# Generate a 2D square mesh from -π/2 to π/2 in the complex plane
xs = range(Float32(-π/2), Float32(π/2), length=side_length);
ys = range(Float32(-π/2), Float32(π/2), length=side_length);

mesh = vec([x + y*im for x in xs, y in ys]);

N = 15;
set_theme!(fontsize = 25);
cuda = CUDA.has_cuda() && CUDA.has_cuda_gpu();

P = Legendre();

function stieltjestransform(k)
    f = expand(P, x -> (sin(x + k)^2 * cos(x + k)^2)); 
    ff = collect(f.args[2][1:N-2]);

    if cuda
        st = GPUInplaceStieltjes(N, mesh, ff).f;
    else
        st = InplaceStieltjes(N, mesh, ff).f;
    end

    return reshape(st, side_length, side_length);
end;

function logtransform(k)
    f = expand(P, x -> (sin(x + k)^2 * cos(x + k)^2)); 
    ff = collect(f.args[2][1:N-2]);

    if cuda
        lt = GPUInplaceLogKernel(N, mesh, ff).f;
    else
        lt = InplaceLogKernel(N, mesh, ff).f;
    end

    return reshape(lt, side_length, side_length);
end;

fig = Figure(size = (1500, 1000));

sg = SliderGrid(
    fig[3, 1:3],
    (
        label=L"k",
        range=-pi/2:0.01:pi/2,
        format="{:.2f}",
        startvalue=0,
        color_active=:grey30,
        color_active_dimmed=:grey60,
        color_inactive=:grey80,
    ),
);

k = sg.sliders[1].value;

st = @lift stieltjestransform($k);
lt = @lift logtransform($k);

ax1 = Axis(fig[1, 2], xlabel=L"Re(z)", ylabel=L"Im(z)");
contour!(ax1, xs, ys, st, levels=15, colormap=:plasma);
lines!(ax1, [-1, 1], [0, 0], color = :black, linewidth = 1.75);

ax2 = Axis3(fig[1, 3], xlabel=L"Re(z)", ylabel=L"Im(z)");
surface!(ax2, xs, ys, st, colormap=:plasma);

ax3 = Axis(fig[2, 2], xlabel=L"Re(z)", ylabel=L"Im(z)");
contour!(ax3, xs, ys, lt, levels=15, colormap=:viridis);
lines!(ax3, [-1, 1], [0, 0], color = :black, linewidth = 1.75);

ax4 = Axis3(fig[2, 3], xlabel=L"Re(z)", ylabel=L"Im(z)");
surface!(ax4, xs, ys, lt, colormap=:viridis);

label1 = Label(fig[1, 1], L"\int_{-1}^1\frac{sin^2(t + k)cos^2(t + k)}{z - t} dt")
label2 = Label(fig[2, 1], L"\int_{-1}^1sin^2(t + k)cos^2(t + k)\log |z - t| dt")

rowsize!(fig.layout, 1, 300)
rowsize!(fig.layout, 2, 300)

fig

save("sin^2cos^2.png", fig);