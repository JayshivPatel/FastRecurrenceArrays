using CairoMakie;

threads = [2, 4, 6, 8];
clenshaw_r = [1.31483e-1, 1.30664e-1, 1.30346e-1, 1.31425e-1]
inplace_r = [1.32023e-1, 1.23682e-1, 1.14882e-1, 1.30863e-3]
clenshaw_c = [4.64790e-2, 2.35220e-2, 1.74620e-2, 1.47110e-2];
inplace_c = [4.98700e-2, 2.59440e-2, 1.81290e-2, 1.43080e-2];

pt = 4/3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    fonts = (regular = "charter", bold = "charter bold", italic = "charter italic", bold_italic = "charter bold italic"),
    figure_padding=1,
);

fig = Figure(size = (6.2inch, 3inch));
ax = Axis(
    fig[1, 1],
    title="Thread scaling of clenshaw/forward-inplace with\n10⁴ recurrences at 10⁴ points",
    xlabel="Threads",
    ylabel="Time [s]",
    xticks=(2:2:8)
);

c_c = scatterlines!(ax, threads, clenshaw_c);
i_c = scatterlines!(ax, threads, inplace_c, linestyle=:dot);

c_r = scatterlines!(ax, threads, clenshaw_r);
i_r = scatterlines!(ax, threads, inplace_r, linestyle=:dot);

axislegend(ax, [c, i], ["clenshaw", "forward-inplace"], position = :rt, orientation=:vertical, backgroundcolor = (:white, 0.85));

fig

save("clenshaw-inplace-threaded-comparison.svg", fig)