using CairoMakie;

points = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6];

fixed_c = [1.00901e-7, 8.13085e-7, 8.98200e-6, 8.55330e-5, 9.09365e-4, 8.07700e-3];
gpu_c = [3.57985e-4, 3.56026e-4, 3.66089e-4, 3.73452e-4, 5.83019e-4, 1.63600e-3];
column_c = [3.65800e-6, 4.21800e-6, 5.57400e-6, 2.40350e-5, 2.23022e-4, 2.45000e-3];

fixed_i = [1.43681e-7, 7.15701e-7, 9.36000e-6, 7.13820e-5, 1.02100e-3, 8.63400e-3];
gpu_i = [1.11700e-3, 4.94688e-4, 5.38774e-4, 5.28650e-4, 8.82094e-4, 2.21600e-3];
column_i = [3.72600e-6, 3.85100e-6, 7.70500e-6, 3.02270e-5, 4.50289e-4, 4.79400e-3];

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
    figure_padding=1,
);
fig = Figure(size=(6.27inch, 4.5inch));
ax = Axis(
    fig[1, 1],
    title="Point scaling of clenshaw/forward-inplace with 10 recurrences",
    xlabel="Points",
    ylabel="Time [s]",
    yscale=log10,
    xscale=log10,
);

f_c = scatterlines!(ax, points, fixed_c, marker=:utriangle);
g_c = scatterlines!(ax, points, gpu_c);
c_c = scatterlines!(ax, points, column_c);

f_color = f_c.color;
g_color = g_c.color;
c_color = c_c.color;

f_i = scatterlines!(ax, points, fixed_i, color=f_color, linestyle=:dot, marker=:utriangle);
g_i = scatterlines!(ax, points, gpu_i, color=g_color, linestyle=:dot);
c_i = scatterlines!(ax, points, column_i, color=c_color, linestyle=:dot);

Legend(
    fig[2, 1],
    [[f_c, g_c, c_c], [f_i, g_i, c_i]],
    [["Control", "GPU", "Threaded (8)"], ["Control", "GPU", "Threaded (8)"]],
    ["clenshaw", "forward-inplace"],
    orientation=:horizontal,
    framevisible=false,
    nbanks=2,
    groupgap=50
);

save("clenshaw-inplace-point-scaling.svg", fig)