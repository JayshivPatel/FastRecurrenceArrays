using CairoMakie;

threads = [2, 4, 6, 8];
row = [1.23494e-1, 1.68765e-1, 1.99661e-1, 2.79667e-1];
column = [1.21960e-1, 1.15200e-1, 1.15560e-1, 1.11702e-1];

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(11pt),
    linewidth=2,
    markersize=13,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic")
);

fig = Figure(size=(6inch, 3inch));
ax = Axis(
    fig[1, 1],
    title="Thread scaling of forward' calculating 10⁴ recurrences at 10⁴ points",
    xlabel="Threads",
    ylabel="Time [s]",
    xticks=(2:2:8),
);

r = scatterlines!(ax, threads, row);
c = scatterlines!(ax, threads, column);
axislegend(ax, [r, c], ["Row-wise", "Column-wise"], position=:lt, orientation=:vertical, backgroundcolor=(:white, 0.85));

fig

save("forward-threaded-comparison.svg", fig)