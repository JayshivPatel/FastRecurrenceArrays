module FastRecurrenceArrays

import Base: copy, size, show, string, getindex
import ClassicalOrthogonalPolynomials: _p0, orthogonalityweight, recurrencecoefficients, Legendre, OrthogonalPolynomial, Ultraspherical
import CUDA
import Distributed: workers, map, fetch
import DistributedData: save_at, get_val_from
import RecurrenceRelationships: _clenshaw_first, clenshaw, check_clenshaw_recurrences, clenshaw_next, forwardrecurrence_partial!, forwardrecurrence_next
import SingularIntegrals: stieltjes

export FixedRecurrenceArray, ThreadedRecurrenceArray, PartitionedRecurrenceArray, PartitionedArray, GPURecurrenceArray
export FixedClenshaw, ThreadedClenshaw, GPUClenshaw
export ForwardInplace, ThreadedInplace, GPUInplace
export FixedCauchy, FixedLogKernel, ClenshawCauchy, ThreadedClenshawCauchy, GPUClenshawCauchy, ClenshawLogKernel, ThreadedClenshawLogKernel, GPUClenshawLogKernel, InplaceCauchy, ThreadedInplaceCauchy, GPUInplaceCauchy, InplaceLogKernel, ThreadedInplaceLogKernel, GPUInplaceLogKernel

include("forward.jl")
include("clenshaw.jl")
include("forward_inplace.jl")
include("integrals.jl")

end # module FastRecurrenceArrays
