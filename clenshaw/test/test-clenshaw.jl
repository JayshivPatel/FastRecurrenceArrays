using Pkg; Pkg.activate("./clenshaw")

using FixedClenshawArrays, RecurrenceRelationshipArrays, LinearAlgebra;

X = SymTridiagonal(zeros(10), 1/2 * ones(10));

rec_U = 2 * ones(10), zeros(10), ones(11);

z = Clenshaw([1, 0.5], rec_U..., X)

y = FixedClenshaw([1, 0.5], rec_U..., X)

vec = [0.1, 0.2]

a = FixedClenshaw([1, 1/2, 1/3, 1/4], rec_U..., vec)