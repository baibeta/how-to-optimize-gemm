#include <stdio.h>
#include <riscv_vector.h>

#define A(i,j) a[ (i)*lda + (j) ]
#define B(i,j) b[ (i)*ldb + (j) ]
#define C(i,j) c[ (i)*ldc + (j) ]


/* Routine for computing C = A * B + C */

void AddDot4x16( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  size_t vl = vsetvlmax_e32m4(); // 16

  vfloat32m4_t c_p0_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p1_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p2_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p3_sum = vfmv_v_f_f32m4(0.0f, vl);
  for (int p = 0; p < k; ++p) {
    vfloat32m4_t  b_reg = vle32_v_f32m4 (&B(p, 0), vl);
    c_p0_sum = vfmacc_vf_f32m4(c_p0_sum, A(0, p),  b_reg, vl);
    c_p1_sum = vfmacc_vf_f32m4(c_p1_sum, A(1, p),  b_reg, vl);
    c_p2_sum = vfmacc_vf_f32m4(c_p2_sum, A(2, p),  b_reg, vl);
    c_p3_sum = vfmacc_vf_f32m4(c_p3_sum, A(3, p),  b_reg, vl);
  }
    vse32_v_f32m4(&C(0, 0), c_p0_sum, vl);
    vse32_v_f32m4(&C(1, 0), c_p1_sum, vl);
    vse32_v_f32m4(&C(2, 0), c_p2_sum, vl);
    vse32_v_f32m4(&C(3, 0), c_p3_sum, vl);
}

// simple pack let m=lda k=ldb
void pack4a(float *a, float *packed, int m, int k, int lda) {
/**
 *  A:			Packed_A:
 *
 *  1 5 x x x ...	1 2 3 4 5 6 7 8 x x ...
 *  2 6 x x x ...	
 *  3 7 x x x ...
 *  4 8 x x x ...   ==> 
 *  x x x x x ...
 *  x x x x x ...
 *  ......
 */
    int count = 0;
    for (int i = 0; i < m; i+=4) {
	for (int j = 0; j < k; j++) {
		packed[count++] = A(i, j);
		packed[count++] = A(i+1, j);
		packed[count++] = A(i+2, j);
		packed[count++] = A(i+3, j);
	}
    }

}

void pack16b(float *b, float *packed, int n, int k, int ldb) {
/**
 *                        B:                                                     Packed_B:
 *  _____________________ N _____________________________
 * |                                                     |
 * 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 ......
 * 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 ......      ==>  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 ....
 * ......
*/
  int count = 0;
  for (int i = 0; i < n; i+=16) {
    for (int j = 0; j < k; j++) {
      for (int p = 0; p < 16; p++) {
        packed[count++] = B(j, i+p);
      }
    }
  }
}



void AddDot4x16_packed4a16b( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  size_t vl = vsetvlmax_e32m4(); // 16
  int index = 0;

  vfloat32m4_t c_p0_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p1_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p2_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p3_sum = vfmv_v_f_f32m4(0.0f, vl);
  
  for (int p = 0; p < k; ++p) {
    vfloat32m4_t  b_reg = vle32_v_f32m4 (&b[p*16], vl);
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


void AddDot4x16_packed16b( int k, float *a, int lda,  float *b, int ldb, float *c, int ldc )
{
  size_t vl = vsetvlmax_e32m4(); // 16

  vfloat32m4_t c_p0_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p1_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p2_sum = vfmv_v_f_f32m4(0.0f, vl);
  vfloat32m4_t c_p3_sum = vfmv_v_f_f32m4(0.0f, vl);
  for (int p = 0; p < k; ++p) {
    vfloat32m4_t  b_reg = vle32_v_f32m4 (&b[p*16], vl);
    c_p0_sum = vfmacc_vf_f32m4(c_p0_sum, A(0, p),  b_reg, vl);
    c_p1_sum = vfmacc_vf_f32m4(c_p1_sum, A(1, p),  b_reg, vl);
    c_p2_sum = vfmacc_vf_f32m4(c_p2_sum, A(2, p),  b_reg, vl);
    c_p3_sum = vfmacc_vf_f32m4(c_p3_sum, A(3, p),  b_reg, vl);
  }
    vse32_v_f32m4(&C(0, 0), c_p0_sum, vl);
    vse32_v_f32m4(&C(1, 0), c_p1_sum, vl);
    vse32_v_f32m4(&C(2, 0), c_p2_sum, vl);
    vse32_v_f32m4(&C(3, 0), c_p3_sum, vl);
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

void MY_MMult_4x16_m4( int m, int n, int k, float *a, int lda, 
                                    float *b, int ldb,
                                    float *c, int ldc )
{
  int i, j, x, y;

  for ( j=0; j<n; j+=16 ){
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      // x = i * lda;
       y = j * ldb;
//      AddDot4x16_packed4a( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
//      AddDot4x16_packed4a( k, &a[x], lda, &B( 0,j ), ldb, &C( i,j ), ldc );
//      AddDot4x16_packed4a16b( k, &A( i,0 ), lda, &b[y], ldb, &C( i,j ), ldc );
      AddDot4x16_packed16b( k, &A( i,0 ), lda, &b[y], ldb, &C( i,j ), ldc );
//     AddDot4x16( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}
