using CairoMakie;

table = (
    categories=[1, 1, 2, 2, 3, 3, 4, 4],
    height=[
        3.20440e-2, 3.70610e-2,
        4.88570e-2, 5.57510e-2,
        3.26098e-1, 1.95355e-1,
        5.29024e-1, 9.42860e-1,
    ],
    groups=[1, 2, 1, 2, 1, 2, 1, 2]
);

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(11pt),
    linewidth=2,
    markersize=13,
    fonts = (regular = "charter", bold = "charter bold", italic = "charter italic", bold_italic = "charter bold italic")
);

fig = Figure(size=(6inch, 3inch));
ax = Axis(
    fig[1, 1],
    title="Singular integral computation time by method",
    xlabel="Time [s]",
    yticks=(1:4, ["Forward Inplace", "Forward", "FastGaussQuadrature.jl", "SingularIntegrals.jl",]),
    xscale=log10,
);

c1 = Makie.wong_colors()[2];
c2 = Makie.wong_colors()[1];

b = barplot!(
    table.categories,
    table.height,
    dodge=table.groups,
    color=table.groups,
    direction=:x,
    colormap=[c1, c2],
);

legend = Legend(
    fig[1, 2],
    [PolyElement(color=c2), PolyElement(color=c1)],
    ["Stieltjes", "LogKernel"],
    orientation=:vertical,
);

fig

save("integrals.svg", fig)