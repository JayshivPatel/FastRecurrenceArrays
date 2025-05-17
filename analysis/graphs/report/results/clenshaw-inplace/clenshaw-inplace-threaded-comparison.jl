using CairoMakie;

threads = [2, 4, 6, 8];
clenshaw = [4.64790e-2, 2.35220e-2, 1.74620e-2, 1.47110e-2];
inplace = [4.98700e-2, 2.59440e-2, 1.81290e-2, 1.43080e-2];

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

c = scatterlines!(ax, threads, clenshaw);
i = scatterlines!(ax, threads, inplace);
axislegend(ax, [c, i], ["clenshaw", "forward-inplace"], position = :rt, orientation=:vertical, backgroundcolor = (:white, 0.85));

fig

save("clenshaw-inplace-threaded-comparison.svg", fig)