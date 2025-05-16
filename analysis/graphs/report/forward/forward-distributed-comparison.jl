using CairoMakie;

workers = [1, 2, 3, 4];
time = [2.62114e-1, 1.44850e-1, 1.06548e-1, 1.00103e-1];

pt = 4/3;
inch = 96;

set_theme!(theme_latexfonts(), fontsize=round(11pt));
fig = Figure(size = (6inch, 3inch));
ax = Axis(
    fig[1, 1],
    title=L"\textbf{Worker scaling of} forward' \textbf{calculating $10^4$ recurrences at $10^4$ points}",
    xlabel="Workers",
    ylabel="Time [s]",
    xticks=(1:4)
);

d = scatterlines!(ax, workers, time);
axislegend(ax, [d], ["Distributed"], position = :rt, orientation=:vertical, backgroundcolor = (:white, 0.85));

fig

save("forward-distributed-comparison.svg", fig)