using CairoMakie;

recurrences = [1e1, 1e2, 1e3, 1e4];
fixed = [1.59294e-7, 1.62800e-6, 1.03350e-5, 2.11947e-4];
gpu = [2.34620e-4, 2.34500e-3, 2.39360e-2, 2.48020e-1];
column = [4.89500e-6, 4.69200e-6, 7.75700e-6, 6.73280e-5];
row = [1.42510e-5, 1.67207e-4, 2.41300e-3, 1.86570e-2];
distributed = [1.47810e-2, 1.25400e-2, 1.48820e-2, 1.52990e-2];

pt = 4/3;
inch = 96;

set_theme!(
    fontsize=round(11pt),
    linewidth=2,
    markersize=13,
    fonts = (regular = "charter", bold = "charter bold", italic = "charter italic", bold_italic = "charter bold italic")
);

fig = Figure(size = (6inch, 5inch));
ax = Axis(
    fig[1, 1],
    title="Recurrence scaling of forward' at 10 points",
    xlabel="Recurrences",
    ylabel="Time [s]",
    yscale=log10,
    xscale=log10
);

f = scatterlines!(ax, recurrences, fixed, linestyle=:dash);
g = scatterlines!(ax, recurrences, gpu);
c = scatterlines!(ax, recurrences, column);
r = scatterlines!(ax, recurrences, row);
d = scatterlines!(ax, recurrences, distributed);

axislegend(
    ax,
    [f, g, c, r, d],
    ["Control", "GPU", "Column (8)", "Row (2)", "Distributed (4)"],
    position = :rb,
    orientation=:vertical,
    backgroundcolor = (:white, 0.85)
);


fig

save("forward-recurrence-scaling.svg", fig)
