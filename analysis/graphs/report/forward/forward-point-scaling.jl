using CairoMakie;

points = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6];
fixed = [1.59294e-7, 1.77400e-6, 1.51670e-5, 1.55786e-4, 1.99000e-3, 2.35910e-2];
gpu = [2.34620e-4, 2.42656e-4, 2.45604e-4, 2.59646e-4, 5.84619e-4, 1.37700e-3];
column = [4.89500e-6, 6.46900e-6, 1.22290e-5, 6.08640e-5, 1.06300e-3, 1.94010e-2];
row = [1.42510e-5, 1.51040e-5, 2.07020e-5, 1.48709e-4, 5.79242e-4, 1.25900e-2];
distributed = [1.47810e-2, 1.10820e-2, 1.28140e-2, 7.80100e-3, 1.02930e-2, 3.73367e-2];

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(12pt),
    linewidth=2,
    markersize=13,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
    figure_padding=1,
);

fig = Figure(size=(6.27inch, 4inch));
ax = Axis(
    fig[1, 1],
    title="Point scaling of forward' with 10 recurrences",
    xlabel="Points",
    ylabel="Time [s]",
    yscale=log10,
    xscale=log10
);

f = scatterlines!(ax, points, fixed, linestyle=:dash);
g = scatterlines!(ax, points, gpu);
c = scatterlines!(ax, points, column);
r = scatterlines!(ax, points, row);
d = scatterlines!(ax, points, distributed);

axislegend(
    ax,
    [f, g, c, r, d],
    ["Control", "GPU", "Column (8)", "Row (2)", "Distributed (4)"],
    position=:rb,
    orientation=:vertical,
    backgroundcolor=(:white, 0.85)
);

fig

save("forward-point-scaling.svg", fig)