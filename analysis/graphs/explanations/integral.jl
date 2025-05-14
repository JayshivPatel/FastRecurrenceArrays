using FastRecurrenceArrays, ClassicalOrthogonalPolynomials, GLMakie, CUDA;

side_length = 100;

xs = range(Float32(-pi/2), Float32(pi/2), length=side_length);

N = 15;
set_theme!(fontsize = 25)
cuda = CUDA.has_cuda() && CUDA.has_cuda_gpu();

P = Legendre();

function stieltjestransform()
    f = expand(P, x -> (sin(x)^2 * cos(x)^2)); 
    ff = collect(f.args[2][1:N-2]);

    if cuda
        return real(Array(GPUInplaceStieltjes(N, ComplexF32.(xs .+ 0im), ff)));
    else
        return real(InplaceStieltjes(N, xs .+ 0im, ff));
    end
end;

function f(z)
    map(t -> (sin(t)^2 * cos(t)^2)/(z - t), xs)
end;

fig = Figure(size = (1000, 1000));

sg = SliderGrid(
    fig[2, 1:2],
    (
        label=L"z",
        range=-pi/2:0.01:pi/2,
        format="{:.2f}",
        startvalue=0,
        color_active=:grey30,
        color_active_dimmed=:grey60,
        color_inactive=:grey80
    ),
);

z = sg.sliders[1].value;
z_vis = @lift !isapprox($z, 0, atol=1e-3);

fz = @lift f($z);

idxs = findall(x -> -1 ≤ x ≤ 1, xs);
xband = xs[idxs];
yband = @lift $fz[idxs];

ax1 = Axis(fig[1, 1], xlabel=L"t", ylabel=L"f(t)");
int = band!(ax1, xband, yband, 0, color=(:chartreuse2, 0.2));
func = lines!(ax1, xs, fz, color=(:blue, 0.6), linewidth = 2);
vlines!(ax1, z, color=:white, visible=z_vis, linewidth=10);
vlines!(ax1, z, color=:black, visible=z_vis, linewidth=2, linestyle=:dot);
lines!(ax1, [-1, 1], [0, 0], color=:black, linewidth=1.5);
axislegend(ax1, [func, int], [L"f(t) = \frac{sin^2(t)cos^2(t)}{z-t}", L"\int_{-1}^1 f(t) dt"], position = :rb, orientation=:vertical);

ys = stieltjestransform();

z_idx = @lift argmin(abs.(xs .- $z));
marker_x = @lift xs[$z_idx];
marker_y = @lift ys[$z_idx];

ax2 = Axis(fig[1, 2], xlabel=L"z");
stieltjes = lines!(ax2, xs, ys, color=:red, linewidth=2);
value = hlines!(ax2, marker_y, color=:chartreuse2, linewidth=2, linestyle=:dash);
marker = scatter!(ax2, marker_x, marker_y; color=:grey30, marker=:x, markersize=20);
axislegend(ax2, [stieltjes, marker], [L"\int_{-1}^1\frac{sin^2(t + k)cos^2(t + k)}{z - t} dt", L"z"], position = :rb, orientation=:vertical);

fig

save("integral.png", fig);