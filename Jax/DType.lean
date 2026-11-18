module

/-!
# Dtypes

The dtype of an array determines its entries and the dtype of its tangent.
The model follows the default configuration, `jax_enable_x64 = False`,
in which Python floats and ints become `float32` and `int32` arrays.

Values are exact (see `Jax/Scalar.lean`): the width of a dtype does not change any value.
The entries `DType.Elem` of an array, defined in `Jax/Scalar.lean`, have the width in their type.
The width matters only because `lax` primitives require their arguments to have the same dtype,
`check_same_dtypes`, `jax/_src/lax/lax.py:10343`:
```
  dtype = avals[0].dtype
  if any(aval.dtype != dtype for aval in avals[1:]):
    msg = "lax.{} requires arguments to have the same dtypes, got {}."
```
This compares dtypes and ignores weak types, and the `lax` functions do not promote their arguments.
So weak types are not modelled.
-/

@[expose] public section

namespace Jax

/-- The floating-point dtypes `bfloat16`, `float16` and `float32`. -/
inductive FloatWidth where
  | bf16
  | f16
  | f32
  deriving DecidableEq, Repr

/-- The signed integer dtypes `int8`, `int16` and `int32`.
Their values are exact integers: there is no overflow. Unsigned dtypes are not modelled. -/
inductive IntWidth where
  | i8
  | i16
  | i32
  deriving DecidableEq, Repr

/-- The dtype of an array. `float0` is the dtype of the tangents of non-floating-point arrays,
whose only value is the empty one. -/
inductive DType where
  | bool
  | int (w : IntWidth)
  | float (w : FloatWidth)
  | float0
  deriving DecidableEq, Repr

namespace DType

/-- The dtype of the tangents of an array of dtype `d`.
`core.primal_dtype_to_tangent_dtype`, `jax/_src/core.py:2683`:
```
def primal_dtype_to_tangent_dtype(primal_dtype):
  if isinstance(primal_dtype, dtypes.ExtendedDType):
    return primal_dtype._rules.tangent_dtype(primal_dtype)
  elif not dtypes.issubdtype(primal_dtype, np.inexact):
    return dtypes.float0
  else:
    return primal_dtype
```
Extended dtypes are not modelled. -/
def tangent : DType → DType
  | float w => float w
  | _ => float0

/-- `float32`, the dtype of a Python float. -/
abbrev f32 : DType := float .f32

/-- `int32`, the dtype of a Python int and of the indices that JAX creates. -/
abbrev i32 : DType := int .i32

end DType

end Jax
