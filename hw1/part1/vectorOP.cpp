#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  __pp_vec_int all_one_int = _pp_vset_int(1);
  __pp_vec_int all_zero_int = _pp_vset_int(0);
  __pp_vec_float all_99_float = _pp_vset_float(9.999999f);
  __pp_mask all_one_mask = _pp_init_ones();
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    __pp_vec_float cur_x, result=_pp_vset_float(1);
    __pp_vec_int cur_y, count;
    __pp_mask cur_mask=_pp_init_ones(0), tmp_mask=_pp_init_ones(0), whole_mask=_pp_init_ones(i+VECTOR_WIDTH<N?VECTOR_WIDTH:N-i);

    _pp_vload_float(cur_x, values+i, all_one_mask);
    _pp_vload_int(cur_y, exponents+i, all_one_mask);

    _pp_veq_int(cur_mask, cur_y, all_zero_int, whole_mask);
    cur_mask = _pp_mask_not(cur_mask);
    _pp_vmove_float(result, cur_x, cur_mask);
    _pp_vsub_int(count, cur_y, all_one_int, cur_mask);

    _pp_vgt_int(tmp_mask, count, all_zero_int, whole_mask);
    cur_mask = _pp_mask_and(cur_mask, tmp_mask);
    
    while(_pp_cntbits(cur_mask) > 0){
      _pp_vmult_float(result, result, cur_x, cur_mask);
      _pp_vsub_int(count, count, all_one_int, cur_mask);

      _pp_vgt_int(tmp_mask, count, all_zero_int, whole_mask);
      cur_mask = _pp_mask_and(cur_mask, tmp_mask);
    }

    _pp_vgt_float(tmp_mask, result, all_99_float, whole_mask);
    _pp_vset_float(result, 9.999999f, tmp_mask);

    _pp_vstore_float(output+i, result, whole_mask);
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  float sum = 0.0;
  // Assume VECTOR_WIDTH is even and a factor of N!
  for (int i = 0; i < N; i += VECTOR_WIDTH){
    int remain_num_cnt = i+VECTOR_WIDTH<N?VECTOR_WIDTH:N-i;
    __pp_vec_float cur_val;
    __pp_mask whole_mask = _pp_init_ones(remain_num_cnt);
    _pp_vload_float(cur_val, values+i, whole_mask);
    while(remain_num_cnt > 1){
      _pp_hadd_float(cur_val, cur_val);
      _pp_interleave_float(cur_val, cur_val);
      remain_num_cnt /= 2;
    }
    sum += cur_val.value[0];
  }
  return sum;
}