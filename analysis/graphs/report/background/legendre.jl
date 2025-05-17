using CairoMakie;

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
    figure_padding=12,
);

fig = Figure(size=(6.27inch, 3.5inch));

ax = Axis(
    fig[1, 1],
    title="Visualisation of the first five Legendre Polynomials",
    xlabel=L"x",
    ylabel=L"P_n(x)",
    limits = ((-1, 1), (-1, 1))
);

xs = range(-1, 1, 1000);

p0(x) = 1
p1(x) = x
p2(x) = 1/2 * (3x^2 - 1)
p3(x) = 1/2 * (5x^3 - 3x) 
p4(x) = 1/8 * (35x^4 - 30x^2 + 3) 

p_0 = lines!(ax, xs, p0.(xs))
p_1 = lines!(ax, xs, p1.(xs))
p_2 = lines!(ax, xs, p2.(xs))
p_3 = lines!(ax, xs, p3.(xs))
p_4 = lines!(ax, xs, p4.(xs))


Legend(
    fig[2, 1],
    [p_0, p_1, p_2, p_3, p_4],
    [L"P_0(x)", L"P_1(x)", L"P_2(x)", L"P_3(x)", L"P_4(x)"],
    orientation=:horizontal,
    framevisible=false,
    groupgap=50
);

save("legendre.svg", fig)