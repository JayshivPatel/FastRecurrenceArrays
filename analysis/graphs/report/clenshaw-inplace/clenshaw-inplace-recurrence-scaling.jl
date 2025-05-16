using CairoMakie;

recurrences = [1e1, 1e2, 1e3, 1e4];

fixed_c = [1.00901e-7, 8.29719e-7, 9.18400e-6, 9.37290e-5];
gpu_c = [3.57985e-4, 3.83200e-3, 4.18490e-2, 4.31168e-1];
column_c = [3.65800e-6, 3.51100e-6, 4.24100e-6, 2.14600e-5];

fixed_i = [1.43681e-7, 9.55452e-7, 9.72100e-6, 9.91440e-5];
gpu_i = [1.11700e-3, 5.27100e-3, 5.42410e-2, 5.67042e-1];
column_i = [3.72600e-6, 3.63000e-6, 4.64900e-6, 2.30490e-5];

pt = 4/3;
inch = 96;

set_theme!(theme_latexfonts(), fontsize=round(11pt));
fig = Figure(size = (6inch, 5inch));
ax = Axis(
    fig[1, 1],
    title=L"\textbf{Recurrence scaling of } clenshaw/forward-inplace \textbf{ at $10$ points}",
    xlabel="Recurrences",
    ylabel="Time [s]",
    yscale=log10,
    xscale=log10,
    limits=(nothing, (1e-9, 1))
);

f_c = scatterlines!(ax, recurrences, fixed_c);
g_c = scatterlines!(ax, recurrences, gpu_c);
c_c = scatterlines!(ax, recurrences, column_c);

f_color = f_c.color;
g_color = g_c.color;
c_color = c_c.color;

f_i = scatterlines!(ax, recurrences, fixed_i, color=f_color, linestyle=:dot);
g_i = scatterlines!(ax, recurrences, gpu_i, color=g_color, linestyle=:dot);
c_i = scatterlines!(ax, recurrences, column_i, color=c_color, linestyle=:dot);

axislegend(
    ax,
    [f_c, f_i, g_c, g_i, c_c, c_i],
    ["Fixed clenshaw", "Fixed forward-inplace", "GPU clenshaw", "GPU forward-inplace", "Threaded clenshaw", "Threaded forward-inplace"],
    position = :rb,
    orientation=:vertical,
    backgroundcolor = (:white, 0.85)
);

fig

save("clenshaw-inplace-recurrence-scaling.svg", fig)
