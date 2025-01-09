using RecurrenceRelationshipArrays, LinearAlgebra, FixedRecurrenceArrays;

X = SymTridiagonal(zeros(10), 1/2 * ones(10));

rec_U = 2 * ones(10), zeros(10), ones(10);

z = Clenshaw([1, 0.5], 2 * ones(10), zeros(10), ones(11), X)
y = FixedClenshaw([1, 0.5], 2 * ones(10), zeros(10), ones(11), X)