using CairoMakie;

threads = [2, 4, 6, 8];
row = [1.23494e-1, 1.68765e-1, 1.99661e-1, 2.79667e-1];
column = [1.21960e-1, 1.15200e-1, 1.15560e-1, 1.11702e-1];

pt = 4/3;
inch = 96;

set_theme!(theme_latexfonts(), fontsize=round(11pt), linewidth=2, markersize=13);
fig = Figure(size = (6inch, 3inch));
ax = Axis(
    fig[1, 1],
    title=L"\textbf{Thread scaling of}  forward' \textbf{calculating $10^4$ recurrences at $10^4$ points}",
    xlabel=L"\textbf{Threads}",
    ylabel=L"\textbf{Time [s]}",
    xticks=(2:2:8),
);

r = scatterlines!(ax, threads, row);
c = scatterlines!(ax, threads, column);
axislegend(ax, [r, c], ["Row-wise", "Column-wise"], position = :lt, orientation=:vertical, backgroundcolor = (:white, 0.85));

fig

save("forward-threaded-comparison.svg", fig)