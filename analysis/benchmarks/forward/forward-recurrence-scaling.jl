using CairoMakie, CSV, DataFrames;

df = CSV.read("./analysis/benchmarks/forward/recurrence-scaling-10-points.csv", DataFrame);

recurrences = parse.(Int, replace.(df[!, "Recurrences"], "_" => ""));
fixed = df[!, "Fixed"]
gpu = df[!, "GPU"]
column = df[!, "Column-wise (8)"]
row = df[!, "Row-wise (2)"]
distributed = df[!, "Distributed (4)"]


pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=1,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

fig = Figure(size=(6.3inch, 3inch));
ax = Axis(
    fig[1, 1],
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

Legend(
    fig[2, 1],
    [f, g, c, r, d],
    ["control", "GPU", "column-wise", "row-wise", "distributed"],
    orientation=:horizontal,
    framevisible=false
);

fig

save("forward-recurrence-scaling.svg", fig)
