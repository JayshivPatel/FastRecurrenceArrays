using CairoMakie;

threads = [2, 4, 6, 8];
clenshaw = [4.64790e-2, 2.35220e-2, 1.74620e-2, 1.47110e-2];
inplace = [4.98700e-2, 2.59440e-2, 1.81290e-2, 1.43080e-2];

pt = 4/3;
inch = 96;

set_theme!(theme_latexfonts(), fontsize=round(11pt));
fig = Figure(size = (6inch, 3inch));
ax = Axis(
    fig[1, 1],
    title=L"\textbf{Thread scaling of} clenshaw/forward-inplace \textbf{calculating $10^4$ recurrences at $10^4$ points}",
    xlabel="Threads",
    ylabel="Time [s]",
    xticks=(2:2:8)
);

c = scatterlines!(ax, threads, clenshaw);
i = scatterlines!(ax, threads, inplace);
axislegend(ax, [c, i], ["Clenshaw", "Forward Inplace"], position = :rt, orientation=:vertical, backgroundcolor = (:white, 0.85));

fig

save("clenshaw-inplace-threaded-comparison.svg", fig)