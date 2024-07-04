#include <stdio.h>
#include <riscv_vector.h>

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]


/* Routine for computing C = A * B + C */

void AddDot16x8( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  size_t vl = vsetvlmax_e32m2(); // 8

  vfloat32m2_t c_p0_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p1_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p2_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p3_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p4_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p5_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p6_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p7_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p8_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p9_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p10_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p11_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p12_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p13_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p14_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p15_sum = vfmv_v_f_f32m2(0.0f, vl);

  for (int p = 0; p < k; ++p) {
    vfloat32m2_t  b_reg = vle32_v_f32m2 (&B(p, 0), vl);
    c_p0_sum = vfmacc_vf_f32m2(c_p0_sum, A(0, p),  b_reg, vl);
    c_p1_sum = vfmacc_vf_f32m2(c_p1_sum, A(1, p),  b_reg, vl);
    c_p2_sum = vfmacc_vf_f32m2(c_p2_sum, A(2, p),  b_reg, vl);
    c_p3_sum = vfmacc_vf_f32m2(c_p3_sum, A(3, p),  b_reg, vl);
    c_p4_sum = vfmacc_vf_f32m2(c_p4_sum, A(4, p),  b_reg, vl);
    c_p5_sum = vfmacc_vf_f32m2(c_p5_sum, A(5, p),  b_reg, vl);
    c_p6_sum = vfmacc_vf_f32m2(c_p6_sum, A(6, p),  b_reg, vl);
    c_p7_sum = vfmacc_vf_f32m2(c_p7_sum, A(7, p),  b_reg, vl);
    c_p8_sum = vfmacc_vf_f32m2(c_p8_sum, A(8, p),  b_reg, vl);
    c_p9_sum = vfmacc_vf_f32m2(c_p9_sum, A(9, p),  b_reg, vl);
    c_p10_sum = vfmacc_vf_f32m2(c_p10_sum, A(10, p),  b_reg, vl);
    c_p11_sum = vfmacc_vf_f32m2(c_p11_sum, A(11, p),  b_reg, vl);
    c_p12_sum = vfmacc_vf_f32m2(c_p12_sum, A(12, p),  b_reg, vl);
    c_p13_sum = vfmacc_vf_f32m2(c_p13_sum, A(13, p),  b_reg, vl);
    c_p14_sum = vfmacc_vf_f32m2(c_p14_sum, A(14, p),  b_reg, vl);
    c_p15_sum = vfmacc_vf_f32m2(c_p15_sum, A(15, p),  b_reg, vl);

  }
    vse32_v_f32m2(&C(0, 0), c_p0_sum, vl);
    vse32_v_f32m2(&C(1, 0), c_p1_sum, vl);
    vse32_v_f32m2(&C(2, 0), c_p2_sum, vl);
    vse32_v_f32m2(&C(3, 0), c_p3_sum, vl);
    vse32_v_f32m2(&C(4, 0), c_p4_sum, vl);
    vse32_v_f32m2(&C(5, 0), c_p5_sum, vl);
    vse32_v_f32m2(&C(6, 0), c_p6_sum, vl);
    vse32_v_f32m2(&C(7, 0), c_p7_sum, vl);
    vse32_v_f32m2(&C(8, 0), c_p8_sum, vl);
    vse32_v_f32m2(&C(9, 0), c_p9_sum, vl);
    vse32_v_f32m2(&C(10, 0), c_p10_sum, vl);
    vse32_v_f32m2(&C(11, 0), c_p11_sum, vl);
    vse32_v_f32m2(&C(12, 0), c_p12_sum, vl);
    vse32_v_f32m2(&C(13, 0), c_p13_sum, vl);
    vse32_v_f32m2(&C(14, 0), c_p14_sum, vl);
    vse32_v_f32m2(&C(15, 0), c_p15_sum, vl);
}

/* Routine for computing C = A * B + C */

void AddDot16x8_packed16a( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  size_t vl = vsetvlmax_e32m2(); // 8

  vfloat32m2_t c_p0_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p1_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p2_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p3_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p4_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p5_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p6_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p7_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p8_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p9_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p10_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p11_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p12_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p13_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p14_sum = vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t c_p15_sum = vfmv_v_f_f32m2(0.0f, vl);
  int index = 0;

  for (int p = 0; p < k; ++p) {
    vfloat32m2_t  b_reg = vle32_v_f32m2 (&B(p, 0), vl);
    c_p0_sum = vfmacc_vf_f32m2(c_p0_sum, a[index++],  b_reg, vl);
    c_p1_sum = vfmacc_vf_f32m2(c_p1_sum, a[index++],  b_reg, vl);
    c_p2_sum = vfmacc_vf_f32m2(c_p2_sum, a[index++],  b_reg, vl);
    c_p3_sum = vfmacc_vf_f32m2(c_p3_sum,  a[index++],  b_reg, vl);
    c_p4_sum = vfmacc_vf_f32m2(c_p4_sum,  a[index++],  b_reg, vl);
    c_p5_sum = vfmacc_vf_f32m2(c_p5_sum,  a[index++],  b_reg, vl);
    c_p6_sum = vfmacc_vf_f32m2(c_p6_sum,  a[index++],  b_reg, vl);
    c_p7_sum = vfmacc_vf_f32m2(c_p7_sum,  a[index++],  b_reg, vl);
    c_p8_sum = vfmacc_vf_f32m2(c_p8_sum,  a[index++],  b_reg, vl);
    c_p9_sum = vfmacc_vf_f32m2(c_p9_sum,  a[index++],  b_reg, vl);
    c_p10_sum = vfmacc_vf_f32m2(c_p10_sum,  a[index++],  b_reg, vl);
    c_p11_sum = vfmacc_vf_f32m2(c_p11_sum,  a[index++],  b_reg, vl);
    c_p12_sum = vfmacc_vf_f32m2(c_p12_sum,  a[index++],  b_reg, vl);
    c_p13_sum = vfmacc_vf_f32m2(c_p13_sum,  a[index++],  b_reg, vl);
    c_p14_sum = vfmacc_vf_f32m2(c_p14_sum,  a[index++],  b_reg, vl);
    c_p15_sum = vfmacc_vf_f32m2(c_p15_sum,  a[index++],  b_reg, vl);
  }
    vse32_v_f32m2(&C(0, 0), c_p0_sum, vl);
    vse32_v_f32m2(&C(1, 0), c_p1_sum, vl);
    vse32_v_f32m2(&C(2, 0), c_p2_sum, vl);
    vse32_v_f32m2(&C(3, 0), c_p3_sum, vl);
    vse32_v_f32m2(&C(4, 0), c_p4_sum, vl);
    vse32_v_f32m2(&C(5, 0), c_p5_sum, vl);
    vse32_v_f32m2(&C(6, 0), c_p6_sum, vl);
    vse32_v_f32m2(&C(7, 0), c_p7_sum, vl);
    vse32_v_f32m2(&C(8, 0), c_p8_sum, vl);
    vse32_v_f32m2(&C(9, 0), c_p9_sum, vl);
    vse32_v_f32m2(&C(10, 0), c_p10_sum, vl);
    vse32_v_f32m2(&C(11, 0), c_p11_sum, vl);
    vse32_v_f32m2(&C(12, 0), c_p12_sum, vl);
    vse32_v_f32m2(&C(13, 0), c_p13_sum, vl);
    vse32_v_f32m2(&C(14, 0), c_p14_sum, vl);
    vse32_v_f32m2(&C(15, 0), c_p15_sum, vl);
}

// simple pack let m=lda k=ldb
void pack16a(float *a, float *packed, int m, int k, int lda) {
    int count = 0;
    for (int i = 0; i < m; i+=16) {
	for (int j = 0; j < k; j++) {
		packed[count++] = A(i, j);
		packed[count++] = A(i+1, j);
		packed[count++] = A(i+2, j);
		packed[count++] = A(i+3, j);
		packed[count++] = A(i+4, j);
		packed[count++] = A(i+5, j);
		packed[count++] = A(i+6, j);
		packed[count++] = A(i+7, j);
		packed[count++] = A(i+8, j);
		packed[count++] = A(i+9, j);
		packed[count++] = A(i+10, j);
		packed[count++] = A(i+11, j);
		packed[count++] = A(i+12, j);
		packed[count++] = A(i+13, j);
		packed[count++] = A(i+14, j);
		packed[count++] = A(i+15, j);

	}
    }

}



void AddDot4x16_packed4a( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  size_t vl = vsetvlmax_e32m4(); // 16
  int index = 0;

  vfloat32m4_t c_p0_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p1_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p2_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p3_sum = vfmv_v_f_f32m4(0.0f, vl);
  for (int p = 0; p < k; ++p) {
    vfloat32m4_t  b_reg = vle32_v_f32m4 (&B(p, 0), vl);
    c_p0_sum = vfmacc_vf_f32m4(c_p0_sum, a[index++],  b_reg, vl);
    c_p1_sum = vfmacc_vf_f32m4(c_p1_sum, a[index++],  b_reg, vl);
    c_p2_sum = vfmacc_vf_f32m4(c_p2_sum, a[index++],  b_reg, vl);
    c_p3_sum = vfmacc_vf_f32m4(c_p3_sum, a[index++],  b_reg, vl);
  }
    vse32_v_f32m4(&C(0, 0), c_p0_sum, vl);
    vse32_v_f32m4(&C(1, 0), c_p1_sum, vl);
    vse32_v_f32m4(&C(2, 0), c_p2_sum, vl);
    vse32_v_f32m4(&C(3, 0), c_p3_sum, vl);
}



void MY_MMult_16x8_m2( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j, x;

  for ( j=0; j<n; j+=8 ){
    for ( i=0; i<m; i+=16 ){        /* Loop over the rows of C */
      AddDot16x8_packed16a( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
//     AddDot16x8( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}
