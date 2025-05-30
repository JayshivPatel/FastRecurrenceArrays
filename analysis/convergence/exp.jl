using RecurrenceRelationshipArrays, SingularIntegrals, ClassicalOrthogonalPolynomials, CairoMakie;

pt = 4 / 3;
inch = 96;

set_theme!(
    fontsize=round(13pt),
    linewidth=2,
    markersize=13,
    figure_padding=12,
    fonts=(regular="charter", bold="charter bold", italic="charter italic", bold_italic="charter bold italic"),
);

n = 30;
P = Legendre();
ff = transform(P[:, 1:n], exp);

fig = Figure(size=(6.5inch, 3inch));

ax1 = Axis(
    fig[1, 1],
    xlabel=L"n",
    ylabel=L"|f_n|",
    yscale=log10,
);


f = lines!(ax1, 1:n, abs.(ff));

fig

save("exp.svg", fig);
