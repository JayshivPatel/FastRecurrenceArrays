using CairoMakie;

workers = [1, 2, 3, 4];
time = [2.62114e-1, 1.44850e-1, 1.06548e-1, 1.00103e-1];

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
    figure_padding=1,
);

fig = Figure(size=(6.27inch, 3inch));
ax = Axis(
    fig[1, 1],
    title="Worker scaling of forward' with 10⁴ recurrences at 10⁴ points",
    xlabel="Workers",
    ylabel="Time [s]",
    xticks=(1:4)
);

d = scatterlines!(ax, workers, time);
axislegend(ax, [d], ["Distributed"], position=:rt, orientation=:vertical, backgroundcolor=(:white, 0.85));

fig

save("forward-distributed-comparison.svg", fig)