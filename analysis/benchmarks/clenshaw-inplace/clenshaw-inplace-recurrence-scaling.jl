using CairoMakie, CSV, DataFrames;

dfc = CSV.read("./analysis/benchmarks/clenshaw-inplace/clenshaw-recurrence-scaling-10-points.csv", DataFrame);
dfi = CSV.read("./analysis/benchmarks/clenshaw-inplace/inplace-recurrence-scaling-10-points.csv", DataFrame);

recurrences = parse.(Int, replace.(dfc[!, "Recurrences"], "_" => ""));

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
    theme_latexfonts(),
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=1,
);
fig = Figure(size=(6.27inch, 4.5inch));
ax = Axis(
    fig[1, 1],
    title=L"Recurrence scaling of clenshaw/forward-inplace at $10$ points",
    xlabel=L"Recurrences$$",
    ylabel=L"Time [s]$$",
    yscale=log10,
    xscale=log10,
    limits=(nothing, (1e-9, 1))
);

f_c = scatterlines!(ax, recurrences, fixed_c, marker=:utriangle);
g_c = scatterlines!(ax, recurrences, gpu_c);
r_c = scatterlines!(ax, recurrences, row_c);
c_c = scatterlines!(ax, recurrences, column_c);

f_color = f_c.color;
g_color = g_c.color;
r_color = r_c.color;
c_color = c_c.color;

f_i = scatterlines!(ax, recurrences, fixed_i, color=f_color, linestyle=:dot, marker=:utriangle);
g_i = scatterlines!(ax, recurrences, gpu_i, color=g_color, linestyle=:dot);
r_i = scatterlines!(ax, recurrences, row_i, color=r_color, linestyle=:dot);
c_i = scatterlines!(ax, recurrences, column_i, color=c_color, linestyle=:dot);

Legend(
    fig[2, 1],
    [[f_c, g_c, r_c, c_c], [f_i, g_i, r_i, c_i]],
    [[L"Control$$", L"GPU$$", L"Row-wise (4)$$", L"Column-wise (8)$$"], [L"Control$$", L"GPU$$", L"Row-wise (4)$$", L"Column-wise (8)$$"]],
    [L"clenshaw$$", L"forward-inplace$$"],
    orientation=:horizontal,
    framevisible=false,
    nbanks=2,
    groupgap=30
)

fig

save("clenshaw-inplace-recurrence-scaling.svg", fig)
