module

public import Jax.Lax.Primal

/-!
# `convert_element_type`

Differentiation rules of the `convert_element_type` primitive of `jax/_src/lax/lax.py`,
between floating-point dtypes.

Values are exact, so a conversion between widths is the identity on values,
and both rules are correct everywhere.
They return arrays of the right width: the JVP rule converts the tangent to the new dtype,
and the transpose rule converts the cotangent back to the old one.
A conversion from `bool` or an integer dtype has a `float0` tangent, so its rules return symbolic zeros,
which are not modelled.
-/

@[expose] public section

namespace Jax.Lax

variable {n : ℕ} {s : Shape n} {w w' : FloatWidth}

/-- The JVP rule of `convert_element_type` between floating-point dtypes:
the tangent dtype of a floating-point dtype is the dtype itself (`DType.tangent`),
so the tangent is converted to `new_dtype`.
`standard_jvp2` passes the primal result to the rule, `jax/_src/interpreters/ad.py:989`.
`jax/_src/lax/lax.py:5814`:
```
def _convert_element_type_jvp_rule(tangent, primal_result, operand, *,
                                   new_dtype, weak_type, sharding):
  new_tangent_dtype = core.primal_dtype_to_tangent_dtype(new_dtype)
  if new_tangent_dtype == dtypes.float0:
    return ad_util.p2tz(primal_result)
  else:
    return convert_element_type_p.bind(tangent, new_dtype=new_tangent_dtype,
                                       weak_type=weak_type, sharding=sharding)
``` -/
def convertFloatJvp (tangent : Tensor (.float w) (Index s)) (_primalResult : Tensor (.float w') (Index s))
    (_operand : Tensor (.float w) (Index s)) : Tensor (.float w') (Index s) :=
  convertFloat tangent

/-- The transpose rule of `convert_element_type` between floating-point dtypes:
the cotangent is converted back to the dtype of the operand.
`jax/_src/lax/lax.py:5799`:
```
def _convert_element_type_transpose_rule(ct, operand, *, new_dtype, weak_type,
                                         sharding):
  assert ad.is_undefined_primal(operand)
  old_dtype = operand.aval.dtype
  old_weak_type = dtypes.is_weakly_typed(operand)
  if type(ct) is ad_util.Zero:
    return [ad_util.Zero(operand.aval)]
  elif core.primal_dtype_to_tangent_dtype(old_dtype) == dtypes.float0:
    return [ad_util.Zero(operand.aval.update(dtype=dtypes.float0, weak_type=False))]
  else:
    out = convert_element_type_p.bind(
        ct, new_dtype=old_dtype, weak_type=old_weak_type,
        sharding=operand.aval.sharding)
    return [out]
``` -/
def convertFloatTranspose (ct : Tensor (.float w') (Index s)) : Tensor (.float w) (Index s) :=
  convertFloat ct

/-! ## Correctness -/

/-- The JVP rule of `convert_element_type` computes the derivative of the conversion. -/
theorem convertFloatJvp_correct :
    IsJvpOf (fun g x => convertFloatJvp g (convertFloat (w' := w') x) x)
      (convertFloat (w := w) (w' := w') (ι := Index s)) :=
  isJvpOf_of_linear LinearMap.id fun _ => rfl

/-- The transpose rule of `convert_element_type` transposes the conversion. -/
theorem convertFloatTranspose_correct :
    IsTransposeOf (convertFloatTranspose (w := w) (w' := w') (s := s)) (convertFloat (w := w) (w' := w')) :=
  fun ct x => ⟨x, ct, rfl, .of_eq rfl, rfl⟩

end Jax.Lax
