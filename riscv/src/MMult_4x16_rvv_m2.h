#include <stdio.h>
#include <riscv_vector.h>

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]


/* Routine for computing C = A * B + C */

void AddDot4x16( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  size_t vl = vsetvlmax_e32m2(); // 8

  vfloat32m2_t c_p0_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p1_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p2_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p3_sum = vfmv_v_f_f32m2(0.0f, vl);

  vfloat32m2_t c_p08_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p18_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p28_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p38_sum = vfmv_v_f_f32m2(0.0f, vl);

  for (int p = 0; p < k; ++p) {
    vfloat32m2_t  b_reg = vle32_v_f32m2 (&B(p, 0), vl);
    vfloat32m2_t  b_reg8 = vle32_v_f32m2 (&B(p, 8), vl);

    c_p0_sum = vfmacc_vf_f32m2(c_p0_sum, A(0, p),  b_reg, vl);
    c_p1_sum = vfmacc_vf_f32m2(c_p1_sum, A(1, p),  b_reg, vl);
    c_p2_sum = vfmacc_vf_f32m2(c_p2_sum, A(2, p),  b_reg, vl);
    c_p3_sum = vfmacc_vf_f32m2(c_p3_sum, A(3, p),  b_reg, vl);

    c_p08_sum = vfmacc_vf_f32m2(c_p08_sum, A(0, p),  b_reg8, vl);
    c_p18_sum = vfmacc_vf_f32m2(c_p18_sum, A(1, p),  b_reg8, vl);
    c_p28_sum = vfmacc_vf_f32m2(c_p28_sum, A(2, p),  b_reg8, vl);
    c_p38_sum = vfmacc_vf_f32m2(c_p38_sum, A(3, p),  b_reg8, vl);
  }

    vse32_v_f32m2(&C(0, 0), c_p0_sum, vl);
    vse32_v_f32m2(&C(1, 0), c_p1_sum, vl);
    vse32_v_f32m2(&C(2, 0), c_p2_sum, vl);
    vse32_v_f32m2(&C(3, 0), c_p3_sum, vl);

    vse32_v_f32m2(&C(0, 8), c_p08_sum, vl);
    vse32_v_f32m2(&C(1, 8), c_p18_sum, vl);
    vse32_v_f32m2(&C(2, 8), c_p28_sum, vl);
    vse32_v_f32m2(&C(3, 8), c_p38_sum, vl);
}


void MY_MMult_4x16_m2( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=16 ){
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */

      AddDot4x16( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}
