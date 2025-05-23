using CairoMakie, CSV, DataFrames;

dfc = CSV.read("./analysis/benchmarks/clenshaw-inplace/clenshaw-thread-scaling-10e4-recurrences-points.csv", DataFrame);
dfi = CSV.read("./analysis/benchmarks/clenshaw-inplace/inplace-thread-scaling-10e4-recurrences-points.csv", DataFrame);

threads = dfc[!, "Threads"];
clenshaw_r = dfc[!, "Row-wise"];
clenshaw_c = dfc[!, "Column-wise"];

inplace_r = dfi[!, "Row-wise"];
inplace_c = dfi[!, "Column-wise"];

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=8,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.2inch, 3inch));
ax = Axis(
    fig[1, 1],
    xlabel="Threads",
    ylabel="Time [s]",
    xticks=(2:2:8),
    yscale=log10,
);

c_c = scatterlines!(ax, threads, clenshaw_c);
i_c = scatterlines!(ax, threads, inplace_c);

# shift the colours
for i = 1:5
    scatterlines!(ax, [0], [0], visible=false)
end

c_r = scatterlines!(ax, threads, clenshaw_r, linestyle=:dot);
i_r = scatterlines!(ax, threads, inplace_r, linestyle=:dot);

Legend(
    fig[2, 1],
    [[c_r, c_c], [i_r, i_c]],
    [["row-wise", "column-wise"], ["row-wise", "column-wise"]],
    ["clenshaw", "forward-inplace"],
    orientation=:horizontal,
    framevisible=false,
    groupgap=50
);

fig

save("clenshaw-inplace-threaded-comparison.svg", fig)