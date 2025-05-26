using CairoMakie, CSV, DataFrames;

dfc = CSV.read("./analysis/benchmarks/clenshaw-inplace/clenshaw-strong-scaling.csv", DataFrame);
dfi = CSV.read("./analysis/benchmarks/clenshaw-inplace/inplace-strong-scaling.csv", DataFrame);

threads = dfc[!, "Threads"];
clenshaw_r = dfc[!, "Row-wise"];
clenshaw_c = dfc[!, "Column-wise"];

inplace_r = dfi[!, "Row-wise"];
inplace_c = dfi[!, "Column-wise"];

clenshaw_r = clenshaw_r[1] ./ clenshaw_r
clenshaw_c = clenshaw_c[1] ./ clenshaw_c

inplace_r = inplace_r[1] ./ inplace_r
inplace_c = inplace_c[1] ./ inplace_c

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
    ylabel="Speedup",
    xticks=(2:2:8),
    limits=(nothing, (0, 5))
);

d = scatterlines!(ax, threads, threads, linestyle=:dot);
c_c = scatterlines!(ax, threads, clenshaw_c);
i_c = scatterlines!(ax, threads, inplace_c);

# shift the colours
for i = 1:5
    scatterlines!(ax, [0], [0], visible=false)
end

c_r = scatterlines!(ax, threads, clenshaw_r, linestyle=(:dot, :dense));
i_r = scatterlines!(ax, threads, inplace_r, linestyle=(:dot, :dense));

Legend(
    fig[2, 1],
    [[d], [c_r, c_c], [i_r, i_c]],
    [[""], ["row-wise", "column-wise"], ["row-wise", "column-wise"]],
    ["ideal", "clenshaw", "forward-inplace"],
    orientation=:horizontal,
    framevisible=false,
    groupgap=20
);

fig

save("clenshaw-inplace-strong-scaling.svg", fig)