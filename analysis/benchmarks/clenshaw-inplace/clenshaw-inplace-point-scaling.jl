using CairoMakie, CSV, DataFrames;

dfc = CSV.read("./analysis/benchmarks/clenshaw-inplace/clenshaw-point-scaling-10-recurrences.csv", DataFrame);
dfi = CSV.read("./analysis/benchmarks/clenshaw-inplace/inplace-point-scaling-10-recurrences.csv", DataFrame);

points = parse.(Int, replace.(dfc[!, "Points"], "_" => ""));

fixed_c = dfc[!, "Fixed"];
gpu_c = dfc[!, "GPU"];
row_c = dfc[!, "Row-wise (4)"];
column_c = dfc[!, "Column-wise (8)"];

fixed_i = dfi[!, "Fixed"];
gpu_i = dfi[!, "GPU"];
row_i = dfi[!, "Row-wise (4)"];
column_i = dfi[!, "Column-wise (8)"];

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=1,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);
fig = Figure(size=(6.27inch, 4inch));
ax = Axis(
    fig[1, 1],
    xlabel="Points",
    ylabel="Time [s]",
    yscale=log10,
    xscale=log10,
);

f_c = scatterlines!(ax, points, fixed_c, marker=:utriangle);
g_c = scatterlines!(ax, points, gpu_c);
r_c = scatterlines!(ax, points, row_c);
c_c = scatterlines!(ax, points, column_c);

f_color = f_c.color;
g_color = g_c.color;
r_color = r_c.color;
c_color = c_c.color;

f_i = scatterlines!(ax, points, fixed_i, color=f_color, linestyle=:dot, marker=:utriangle);
g_i = scatterlines!(ax, points, gpu_i, color=g_color, linestyle=:dot);
r_i = scatterlines!(ax, points, row_i, color=r_color, linestyle=:dot);
c_i = scatterlines!(ax, points, column_i, color=c_color, linestyle=:dot);

Legend(
    fig[2, 1],
    [[f_c, g_c, r_c, c_c], [f_i, g_i, r_i, c_i]],
    [["control", "GPU", "row-wise", "column-wise"], ["control", "GPU", "row-wise", "column-wise"]],
    ["clenshaw", "forward_inplace"],
    orientation=:horizontal,
    framevisible=false,
    nbanks=2,
    groupgap=30,
    titlefont="Courier"
)

save("clenshaw-inplace-point-scaling.svg", fig)